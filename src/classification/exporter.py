"""ONNX export using AutoGluon's native MultiModalPredictor.export_onnx()."""

from __future__ import annotations

from pathlib import Path


def export_onnx(
    model_dir: Path,
    out_dir: Path,
    opset: int = 17,
    input_size: int = 224,
) -> Path:
    """Export AutoGluon predictor to ONNX format.

    Returns:
        Path to the exported model.onnx file.
    """
    from autogluon.multimodal import MultiModalPredictor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = MultiModalPredictor.load(str(model_dir))
    predictor.export_onnx(
        save_path=str(out_dir),
        opset=opset,
        input_size=(input_size, input_size),
    )

    onnx_path = out_dir / "model.onnx"
    return onnx_path
