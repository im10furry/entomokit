"""AutoGluon MultiModalPredictor training logic."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def train(
    train_csv: Path,
    images_dir: Path,
    base_model: str,
    out_dir: Path,
    augment_transforms: List[str],
    max_epochs: int,
    time_limit_hours: float,
    focal_loss: bool,
    focal_loss_gamma: float,
    device: str,  # 'cpu'/'cuda'/'mps' (already resolved)
    batch_size: int,
    num_workers: int,
    num_threads: int,
) -> Path:
    """Train an AutoGluon MultiModalPredictor for image classification.

    Returns:
        Path to the saved predictor directory.
    """
    from autogluon.multimodal import MultiModalPredictor
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = pd.read_csv(train_csv)
    df["image"] = df["image"].apply(lambda p: str(images_dir / p))

    model_dir = out_dir / "AutogluonModels" / base_model
    model_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "model.timm_image.checkpoint_name": base_model,
        "model.timm_image.train_transforms": augment_transforms,
        "env.num_workers": num_workers,
        "env.batch_size": batch_size,
    }
    if focal_loss:
        hyperparameters["optimization.loss_function"] = "focal_loss"
        hyperparameters["optimization.focal_loss.alpha"] = -1
        hyperparameters["optimization.focal_loss.gamma"] = focal_loss_gamma

    predictor = MultiModalPredictor(
        label="label",
        problem_type="multiclass",
        path=str(model_dir),
    )
    predictor.fit(
        train_data=df,
        hyperparameters=hyperparameters,
        max_epochs=max_epochs,
        time_limit=int(time_limit_hours * 3600),
    )

    # Save processed CSV for traceability
    processed_csv = out_dir / "train.processed.csv"
    df.to_csv(processed_csv, index=False)

    return model_dir
