"""GradCAM heatmap generation for CNN and ViT backbones.

Supports timm backbones and AutoGluon MultiModalPredictor checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    EigenCAM,
    GradCAMPlusPlus,
    LayerCAM,
    AblationCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm.data import resolve_model_data_config
from torchvision.transforms.functional import InterpolationMode
import cv2

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcampp": GradCAMPlusPlus,
    "layercam": LayerCAM,
    "ablationcam": AblationCAM,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
}


def load_label_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns named 'image' and 'label'.")
    return df[["image", "label"]]


def load_model_from_args(
    load_ag: Optional[str],
    base_model: Optional[str],
    checkpoint_path: Optional[str],
    num_classes: Optional[int],
    pretrained: bool,
    device: torch.device,
) -> torch.nn.Module:
    if load_ag:
        try:
            from autogluon.multimodal import MultiModalPredictor
        except ImportError as exc:
            raise ImportError(
                "AutoGluon is not installed. pip install autogluon.multimodal"
            ) from exc
        predictor = MultiModalPredictor.load(load_ag)
        torch_model = predictor._learner._model.model
        logging.info("Loaded AutoGluon model from %s", load_ag)
    else:
        if base_model is None:
            raise ValueError("Either model_dir or base_model must be provided.")
        import timm

        torch_model = timm.create_model(
            base_model,
            pretrained=pretrained,
            num_classes=num_classes if num_classes is not None else None,
        )
        logging.info(
            "Instantiated timm backbone %s (pretrained=%s, num_classes=%s)",
            base_model,
            pretrained,
            num_classes,
        )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            torch_model.load_state_dict(state_dict, strict=False)
            logging.info("Loaded checkpoint weights from %s", checkpoint_path)
    torch_model.eval()
    torch_model.to(device)
    return torch_model


def build_eval_transforms(
    model: torch.nn.Module,
) -> Tuple[transforms.Compose, transforms.Compose]:
    cfg = resolve_model_data_config(model)
    crop_tuple = cfg.get("test_input_size", cfg.get("input_size"))
    crop_size = crop_tuple[1]
    crop_pct = cfg.get("test_crop_pct", cfg.get("crop_pct", 1.0))
    resize_shorter = int(round(crop_size / crop_pct))
    mean = cfg.get("mean")
    std = cfg.get("std")
    interpolation = cfg.get("interpolation", "bicubic").upper()
    interpolation = getattr(InterpolationMode, interpolation, InterpolationMode.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_shorter, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    display_transform = transforms.Compose(
        [
            transforms.Resize(resize_shorter, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
        ]
    )
    return preprocess, display_transform


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = model
    for attr in name.split("."):
        if not hasattr(module, attr):
            raise AttributeError(
                f"Module '{module.__class__.__name__}' has no attribute '{attr}'"
            )
        module = getattr(module, attr)
    return module


def find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found. Specify --target-layer-name manually."
        )
    return last_conv


def default_vit_target(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        block = model.blocks[-1]
        for candidate in ["norm1", "ln1", "ln"]:
            if hasattr(block, candidate):
                return getattr(block, candidate)
        # fallback to block itself
        return block
    raise RuntimeError(
        "Could not automatically find a ViT block. Specify --target-layer-name."
    )


def infer_architecture(base_model_name: str, model: torch.nn.Module) -> str:
    name = (base_model_name or model.__class__.__name__).lower()
    if "vit" in name or "transformer" in name:
        return "vit"
    return "cnn"


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT tokens (B, N, C) into feature maps (B, C, H, W)."""
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected ViT token tensor (B, N, C). Got shape {tensor.shape}."
        )
    tensor = tensor[:, 1:, :]  # drop CLS token
    batch, tokens, channels = tensor.shape
    spatial_dim = int(tokens**0.5)
    if spatial_dim * spatial_dim != tokens:
        raise ValueError("Token count cannot form a square grid.")
    tensor = tensor.permute(0, 2, 1).reshape(batch, channels, spatial_dim, spatial_dim)
    return tensor


def prepare_cam(
    model: torch.nn.Module,
    arch: str,
    target_layer_name: Optional[str],
    cam_name: str,
    device: torch.device,
    cam_batch_size: int,
) -> Tuple:
    if target_layer_name:
        target_layers = [get_module_by_name(model, target_layer_name)]
    else:
        target_layers = (
            [find_last_conv_module(model)]
            if arch == "cnn"
            else [default_vit_target(model)]
        )

    reshape_transform = vit_reshape_transform if arch == "vit" else None
    cam_kwargs = {
        "model": model,
        "target_layers": target_layers,
        "reshape_transform": reshape_transform,
    }
    if cam_name == "ablationcam" and arch == "vit":
        cam_kwargs["ablation_layer"] = AblationLayerVit()
    cam = CAM_METHODS[cam_name](**cam_kwargs)
    if isinstance(cam, BaseCAM):
        cam.batch_size = cam_batch_size  # ScoreCAM/EigenCAM
    return cam, target_layers, reshape_transform


def prepare_output_dirs(out_dir: Path) -> Dict[str, Path]:
    fig_dir = out_dir / "figures"
    array_dir = out_dir / "arrays"
    for d in [fig_dir, array_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"fig": fig_dir, "array": array_dir}


def process_image(
    img_path: Path,
    label: str,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    display_transform: transforms.Compose,
    cam_extractor,
    device: torch.device,
    fig_dir: Path,
    array_dir: Path,
    image_weight: float,
    fig_format: str,
    save_npy: bool,
) -> Dict[str, str]:
    pil_img = Image.open(img_path).convert("RGB")
    original_img = pil_img.copy()
    rgb_display = np.array(original_img).astype(np.float32) / 255.0

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits, dim=1)
        pred_score = float(probs[0, pred_idx].cpu())

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

    cam_norm = grayscale_cam - grayscale_cam.min()
    if cam_norm.max() > 0:
        cam_norm = cam_norm / cam_norm.max()
    else:
        cam_norm = np.zeros_like(cam_norm)

    cam_on_full = cv2.resize(
        cam_norm,
        (original_img.width, original_img.height),
        interpolation=cv2.INTER_LINEAR,
    )

    overlay = show_cam_on_image(
        rgb_display,
        cam_on_full,
        use_rgb=True,
        image_weight=image_weight,
    )
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    combined = Image.new("RGB", (original_img.width * 2, original_img.height))
    combined.paste(original_img, (0, 0))
    combined.paste(overlay_img, (original_img.width, 0))

    fig_path = fig_dir / f"{img_path.stem}_cam.{fig_format}"
    combined.save(fig_path)

    cam_array_path = ""
    if save_npy:
        npy_path = array_dir / f"{img_path.stem}.npy"
        np.save(npy_path, cam_norm.astype(np.float32))
        cam_array_path = str(npy_path)

    return {
        "image": img_path.name,
        "label": label,
        "pred_class": pred_idx,
        "pred_prob": pred_score,
        "figure_path": str(fig_path),
        "cam_array_path": cam_array_path,
    }


def run_cam(
    *,
    label_csv: Path,
    images_dir: Path,
    out_dir: Path,
    model_dir: Optional[Path],
    base_model: Optional[str],
    checkpoint_path: Optional[Path],
    num_classes: Optional[int],
    pretrained: bool,
    cam_method: str,
    arch: Optional[str],
    target_layer_name: Optional[str],
    image_weight: float,
    fig_format: str,
    save_npy: bool,
    max_images: Optional[int],
    cam_batch_size: int,
    device: torch.device,
) -> None:
    """Run CAM heatmap generation for all images in label_csv."""
    out_dirs = prepare_output_dirs(out_dir)
    df = load_label_file(label_csv)
    model = load_model_from_args(
        load_ag=str(model_dir) if model_dir else None,
        base_model=base_model,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )
    preprocess, display_transform = build_eval_transforms(model)
    inferred_arch = arch or infer_architecture(base_model or "", model)
    cam_extractor, target_layers, reshape_transform = prepare_cam(
        model, inferred_arch, target_layer_name, cam_method, device, cam_batch_size
    )

    logging.info("Architecture inferred as %s", inferred_arch)
    logging.info(
        "Using target layer(s): %s",
        ", ".join([layer.__class__.__name__ for layer in target_layers]),
    )
    if reshape_transform:
        logging.info("Enabled ViT reshape_transform for CAM.")

    records = []
    for idx, row in df.iterrows():
        if max_images and idx >= max_images:
            break
        img_path = (images_dir / row["image"]).resolve()
        if not img_path.exists():
            logging.error("Image not found: %s", img_path)
            continue
        try:
            record = process_image(
                img_path=img_path,
                label=str(row["label"]),
                model=model,
                preprocess=preprocess,
                display_transform=display_transform,
                cam_extractor=cam_extractor,
                device=device,
                fig_dir=out_dirs["fig"],
                array_dir=out_dirs["array"],
                image_weight=image_weight,
                fig_format=fig_format,
                save_npy=save_npy,
            )
            records.append(record)
        except Exception as exc:
            logging.exception("Failed on %s: %s", img_path, exc)
    if records:
        pd.DataFrame(records).to_csv(out_dir / "cam_summary.csv", index=False)
