"""Compile augmentation policy dicts into albumentations Compose pipelines."""

from __future__ import annotations

from typing import Optional

import albumentations as A

from src.augment.policy import PRESETS


def _build_transform(cfg: dict) -> A.BasicTransform:
    """Instantiate a single albumentations transform from a config dict."""
    name = cfg.get("name")
    if not name:
        raise ValueError("Transform config must have a 'name' key.")

    transform_cls = getattr(A, name, None)
    if transform_cls is None:
        raise ValueError(
            f"unknown transform: {name!r}. Must be a valid albumentations class name."
        )

    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return transform_cls(**kwargs)


def build_pipeline(
    preset: Optional[str] = "light",
    custom: Optional[dict] = None,
    args: Optional[dict] = None,
) -> A.Compose:
    """Build an albumentations Compose pipeline."""
    if preset is not None and custom is not None:
        raise ValueError(
            "Provide preset or custom, not both. "
            "Use preset=None to pass a custom policy."
        )
    if preset is None and custom is None:
        raise ValueError("Must provide either preset or custom policy dict.")

    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(
                f"unknown preset: {preset!r}. Available: {sorted(PRESETS)}"
            )
        transform_cfgs = PRESETS[preset]
    else:
        transform_cfgs = custom.get("transforms", [])

    if args:
        transform_cfgs = [
            {**cfg, **{k: v for k, v in args.items() if k not in cfg}}
            for cfg in transform_cfgs
        ]

    transforms = [_build_transform(cfg) for cfg in transform_cfgs]
    return A.Compose(transforms)
