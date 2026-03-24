"""Image classification inference — AutoGluon and ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def predict_ag(
    input_df: pd.DataFrame,
    images_dir: Optional[Path],
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    num_threads: int,
    device: str,
) -> pd.DataFrame:
    """Run inference with AutoGluon predictor.
    Returns DataFrame with columns: image, prediction, proba_*.
    """
    from autogluon.multimodal import MultiModalPredictor
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = input_df.copy()
    if images_dir:
        df["image"] = df["image"].apply(lambda p: str(images_dir / p))

    predictor = MultiModalPredictor.load(str(model_dir))
    proba = predictor.predict_proba(df)
    predictions = predictor.predict(df)

    result = input_df.copy()
    result["prediction"] = predictions.values
    for cls in proba.columns:
        result[f"proba_{cls}"] = proba[cls].values
    return result


def predict_onnx(
    input_df: pd.DataFrame,
    images_dir: Optional[Path],
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
) -> pd.DataFrame:
    """Run inference with ONNX model.
    Returns DataFrame with columns: image, prediction, proba_*.
    """
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import torchvision.transforms as T

    sess_options = ort.SessionOptions()
    if num_threads > 0:
        sess_options.intra_op_num_threads = num_threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options)

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_name = sess.get_inputs()[0].name
    all_proba = []

    paths = input_df["image"].tolist()
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        tensors = []
        for p in batch_paths:
            full_p = (images_dir / p) if images_dir else Path(p)
            img = Image.open(full_p).convert("RGB")
            tensors.append(transform(img).numpy())
        batch = np.stack(tensors)
        logits = sess.run(None, {input_name: batch})[0]
        # softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        all_proba.append(proba)

    all_proba_arr = np.vstack(all_proba)
    n_classes = all_proba_arr.shape[1]

    result = input_df.copy()
    result["prediction"] = all_proba_arr.argmax(axis=1)
    for i in range(n_classes):
        result[f"proba_{i}"] = all_proba_arr[:, i]
    return result
