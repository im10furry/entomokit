"""ONNX export using AutoGluon's native MultiModalPredictor.export_onnx()."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd


_DUMMY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\x1dc\xf8\xff\xff?\x00\x05"
    b"\xfe\x02\xfe\x89\x99\x1d\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _infer_image_column(model_dir: Path) -> str:
    assets_path = model_dir / "assets.json"
    if not assets_path.exists():
        return "image"

    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    column_types = assets.get("column_types", {})
    for name, dtype in column_types.items():
        if dtype == "image_path":
            return name

    return "image"


def _ensure_trace_image(sample_image: Path | None) -> Path:
    if sample_image is not None:
        sample_image = Path(sample_image)
        if not sample_image.is_file():
            raise FileNotFoundError(f"--sample-image does not exist: {sample_image}")
        return sample_image

    fd, path = tempfile.mkstemp(prefix="entomokit_onnx_trace_", suffix=".png")
    Path(path).write_bytes(_DUMMY_PNG)
    return Path(path)


def _resolve_exported_model_path(out_dir: Path, exported_path: object) -> Path:
    target = out_dir / "model.onnx"
    candidates: list[Path] = []

    if isinstance(exported_path, str):
        p = Path(exported_path)
        if p.is_file():
            candidates.append(p)
        if p.is_dir():
            candidates.append(p / "model.onnx")

    candidates.extend(
        [
            out_dir / "model.onnx",
            out_dir / "model.onnx" / "model.onnx",
        ]
    )
    candidates.extend(out_dir.glob("**/model.onnx"))

    existing = next((p for p in candidates if p.is_file()), None)
    if existing is None:
        raise FileNotFoundError(f"No exported model.onnx found under {out_dir}")

    if existing == target:
        return target

    if target.exists() and target.is_dir() and existing.is_relative_to(target):
        blob = existing.read_bytes()
        shutil.rmtree(target)
        target.write_bytes(blob)
        return target

    if target.exists() and target.is_file():
        target.unlink()
    if target.exists() and target.is_dir():
        shutil.rmtree(target)

    shutil.move(str(existing), str(target))

    parent = existing.parent
    if parent != out_dir:
        try:
            parent.rmdir()
        except OSError:
            pass

    return target


def _write_label_metadata(predictor: object, out_dir: Path) -> None:
    class_labels = getattr(predictor, "class_labels", None)
    if class_labels is None:
        return

    labels = [str(x) for x in list(class_labels)]
    (out_dir / "label_classes.json").write_text(
        json.dumps({"class_labels": labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_onnx(
    model_dir: Path,
    out_dir: Path,
    opset: int = 17,
    sample_image: Path | None = None,
) -> Path:
    """Export AutoGluon predictor to ONNX format.

    Returns:
        Path to the exported model.onnx file.
    """
    from autogluon.multimodal import MultiModalPredictor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_trace = out_dir / "_onnx_trace_input.png"
    if legacy_trace.exists():
        legacy_trace.unlink()

    legacy_nested = out_dir / "model.onnx"
    if legacy_nested.is_dir():
        shutil.rmtree(legacy_nested)

    model_dir = Path(model_dir)
    image_column = _infer_image_column(model_dir)
    trace_image = _ensure_trace_image(sample_image=sample_image)
    trace_is_temp = sample_image is None

    predictor = MultiModalPredictor.load(str(model_dir))
    try:
        data = pd.DataFrame([{image_column: str(trace_image)}])
        exported_path = predictor.export_onnx(
            data=data,
            path=str(out_dir),
            opset_version=opset,
        )
        _write_label_metadata(predictor=predictor, out_dir=out_dir)
    finally:
        if trace_is_temp and trace_image.exists():
            trace_image.unlink()

    return _resolve_exported_model_path(out_dir=out_dir, exported_path=exported_path)
