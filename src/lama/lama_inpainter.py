import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

# Ensure saicinpainting can be imported
lama_dir = os.path.dirname(__file__)
if lama_dir not in sys.path:
    sys.path.insert(0, lama_dir)

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.refinement import refine_predict

import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger('root').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')


class LaMaInpainter:
    """LaMa (Large Mask Inpainting) WACV 2022 inpainting wrapper.
    
    Supports CPU inference for projects that don't have GPU access.
    """

    def __init__(self, device: str = "auto", checkpoint_path: str = None, refine: bool = False):
        """Initialize LaMa inpainter.
        
        Args:
            device: Device for inference ("auto", "cpu", "cuda", "mps")
            checkpoint_path: Path to model checkpoint directory (containing config.yaml and models/best.ckpt)
            refine: Enable refinement mode for better results
        """
        self.device = self._select_device(device)
        self.checkpoint_path = checkpoint_path
        self.refine = refine
        self.model = self._load_model()

    def _select_device(self, device: str) -> torch.device:
        """Select appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif device == "cpu":
            return torch.device("cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            return torch.device("mps")
        else:
            raise ValueError(f"Unknown device: {device}")

    def _load_model(self):
        """Load LaMa model checkpoint."""
        if self.checkpoint_path:
            # Use provided checkpoint path
            model_path = self.checkpoint_path
        else:
            # Default path
            lama_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(os.path.dirname(lama_dir))
            model_path = os.path.join(project_root, 'models', 'big-lama')
        
        checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')
        config_path = os.path.join(model_path, 'config.yaml')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        
        model = load_checkpoint(
            train_config, 
            checkpoint_path, 
            strict=False, 
            map_location='cpu'
        )
        model.freeze()
        model.to(self.device)
        
        return model

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint the image using LaMa.
        
        Args:
            image: Input image (H, W, 3) in uint8 RGB format
            mask: Binary mask where 255=area to inpaint, 0=valid
        
        Returns:
            Repaired image (H, W, 3) in uint8 RGB format
        """
        assert image.ndim == 3 and image.shape[2] == 3, "Image must be HWC with 3 channels"
        assert mask.ndim == 2 or (mask.ndim == 3 and mask.shape[2] == 1), "Mask must be HW or HWC"
        
        if mask.ndim == 3:
            mask = mask.squeeze(2)
        
        mask = mask.astype(np.uint8)
        
        lama_mask = (mask > 0).astype(np.uint8)
        
        image_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_torch = torch.from_numpy(lama_mask).unsqueeze(0).unsqueeze(0).float()
        
        image_torch = image_torch.to(self.device)
        mask_torch = mask_torch.to(self.device)
        
        unpad_to_size = tuple(image.shape[:2])
        
        with torch.no_grad():
            batch = {
                'image': image_torch,
                'mask': mask_torch
            }
            batch['mask'] = (batch['mask'] > 0) * 1
            
            if self.refine:
                batch['unpad_to_size'] = unpad_to_size
                result = refine_predict(batch, self.model, refine_refine=True)
                result = result['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                batch['image'] = pad_tensor_to_modulo(batch['image'], 64)
                batch['mask'] = pad_tensor_to_modulo(batch['mask'], 64)
                result = self.model(batch)
                result = result['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
                
                result = result[:unpad_to_size[0], :unpad_to_size[1], :]
        
        result = np.clip(result * 255, 0, 255).astype('uint8')
        result = result[:, :, ::-1]  # BGR to RGB
        
        return result
