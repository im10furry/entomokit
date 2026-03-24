"""Shared utilities for the classify command group."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Valid augment preset names and their corresponding transform lists
AUGMENT_TRANSFORMS = {
    "none": ["resize_shorter_side", "center_crop"],
    "light": ["resize_shorter_side", "center_crop", "random_horizontal_flip"],
    "medium": [
        "resize_shorter_side",
        "center_crop",
        "random_horizontal_flip",
        "color_jitter",
        "trivial_augment",
    ],
    "heavy": [
        "random_resize_crop",
        "random_horizontal_flip",
        "random_vertical_flip",
        "color_jitter",
        "trivial_augment",
        "randaug",
    ],
}

# All valid transform string names accepted by AutoGluon
VALID_TRANSFORM_NAMES = {
    "resize_to_square",
    "resize_shorter_side",
    "center_crop",
    "random_resize_crop",
    "random_horizontal_flip",
    "random_vertical_flip",
    "color_jitter",
    "affine",
    "randaug",
    "trivial_augment",
}


def resolve_augment(augment: str) -> List[str]:
    """Parse --augment value to a list of transform strings.

    Accepts preset names (none/light/medium/heavy) or a JSON array string.
    Raises ValueError on invalid preset or unknown transform names in JSON.
    """
    if augment in AUGMENT_TRANSFORMS:
        return AUGMENT_TRANSFORMS[augment]

    # Try JSON
    try:
        transforms = json.loads(augment)
    except json.JSONDecodeError:
        valid = ", ".join(sorted(AUGMENT_TRANSFORMS))
        raise ValueError(
            f"Invalid --augment value: {augment!r}. "
            f"Use one of [{valid}] or a JSON array of transform names."
        )

    if not isinstance(transforms, list):
        raise ValueError(
            "--augment JSON must be an array, e.g. '[\"random_resize_crop\"]'"
        )

    unknown = [t for t in transforms if t not in VALID_TRANSFORM_NAMES]
    if unknown:
        valid_names = ", ".join(sorted(VALID_TRANSFORM_NAMES))
        raise ValueError(
            f"Unknown transform name(s) in --augment: {unknown}. "
            f"Valid names: [{valid_names}]"
        )

    return transforms


def select_device(device_str: str) -> torch.device:
    """Resolve 'auto'/cuda/mps/cpu to a torch.device, with fallback."""
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    if device_str == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("Warning: MPS not available, falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_str)


def set_num_threads(num_threads: int) -> None:
    """Set PyTorch CPU thread count. 0 = let PyTorch decide."""
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def load_image_csv(csv_path: Path, require_label: bool = False) -> "pd.DataFrame":
    """Load and validate a CSV with at least an 'image' column."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"CSV must have an 'image' column: {csv_path}")
    if require_label and "label" not in df.columns:
        raise ValueError(f"CSV must have a 'label' column: {csv_path}")
    return df


def ag_device_map(device: torch.device) -> str:
    """Convert torch.device to AutoGluon device string."""
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"
