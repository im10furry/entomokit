"""Classification evaluation — accuracy, F1, MCC, AUC."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np


def evaluate(
    test_csv: Path,
    images_dir: Path,
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    num_threads: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate AutoGluon predictor. Returns dict of metric name → value."""
    from autogluon.multimodal import MultiModalPredictor
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
    )
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = pd.read_csv(test_csv)
    df["image"] = df["image"].apply(lambda p: str(images_dir / p))

    predictor = MultiModalPredictor.load(str(model_dir))
    predictions = predictor.predict(df).values
    labels = df["label"].values

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "precision_micro": precision_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall_micro": recall_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }

    # ROC-AUC (OVR, requires probabilities)
    try:
        from sklearn.metrics import roc_auc_score

        proba = predictor.predict_proba(df).values
        metrics["roc_auc_ovo"] = roc_auc_score(
            labels, proba, multi_class="ovo", average="macro"
        )
    except Exception:
        metrics["roc_auc_ovo"] = float("nan")

    return metrics


def evaluate_onnx(
    test_csv: Path,
    images_dir: Path,
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
) -> Dict[str, float]:
    """Evaluate ONNX model using predict_onnx + sklearn metrics."""
    from src.classification.predictor import predict_onnx
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    df = pd.read_csv(test_csv)
    result = predict_onnx(df, images_dir, onnx_path, batch_size, num_threads)

    labels = df["label"].values
    predictions = result["prediction"].values

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }
