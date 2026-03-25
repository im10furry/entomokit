"""Tests for classify export-onnx CLI and exporter behavior."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def test_export_onnx_parser_accepts_sample_image() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "classify",
            "export-onnx",
            "--model-dir",
            "model",
            "--out-dir",
            "out",
            "--sample-image",
            "sample.jpg",
        ]
    )

    assert args.sample_image == "sample.jpg"


def test_exporter_uses_current_autogluon_export_kwargs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.classification import exporter

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "assets.json").write_text(
        json.dumps({"column_types": {"specimen_path": "image_path"}}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "onnx"
    sample_image = tmp_path / "sample.jpg"
    sample_image.write_bytes(b"fake")

    captured: dict[str, object] = {}

    class FakePredictor:
        class_labels = ["cls0", "cls1"]

        @classmethod
        def load(cls, _path: str):
            return cls()

        def export_onnx(self, **kwargs):
            captured.update(kwargs)
            nested = out_dir / "model.onnx"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "model.onnx").write_bytes(b"onnx")

    fake_mm = types.ModuleType("autogluon.multimodal")
    fake_mm.MultiModalPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", fake_mm)

    onnx_path = exporter.export_onnx(
        model_dir=model_dir,
        out_dir=out_dir,
        opset=19,
        sample_image=sample_image,
    )

    assert onnx_path == out_dir / "model.onnx"
    assert onnx_path.exists()
    assert captured["path"] == str(out_dir)
    assert captured["opset_version"] == 19
    assert "save_path" not in captured
    assert "input_size" not in captured

    data = captured["data"]
    assert data["specimen_path"].tolist() == [str(sample_image)]
    label_meta = out_dir / "label_classes.json"
    assert label_meta.exists()


def test_exporter_cleans_up_auto_generated_trace_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.classification import exporter

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "assets.json").write_text(
        json.dumps({"column_types": {"image": "image_path"}}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "onnx"

    class FakePredictor:
        @classmethod
        def load(cls, _path: str):
            return cls()

        def export_onnx(self, **_kwargs):
            (out_dir / "model.onnx").write_bytes(b"onnx")

    fake_mm = types.ModuleType("autogluon.multimodal")
    fake_mm.MultiModalPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", fake_mm)

    exporter.export_onnx(
        model_dir=model_dir,
        out_dir=out_dir,
        opset=17,
        sample_image=None,
    )

    assert not (out_dir / "_onnx_trace_input.png").exists()
