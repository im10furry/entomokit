"""Tests for ONNX prediction input/output handling."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


def test_predict_onnx_feeds_required_valid_num_and_selects_logits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.classification.predictor import predict_onnx

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"fake")
    (images_dir / "b.jpg").write_bytes(b"fake")
    (tmp_path / "model.onnx").write_bytes(b"placeholder")

    input_df = pd.DataFrame({"image": ["a.jpg", "b.jpg"]})

    class FakeInput:
        def __init__(self, name: str):
            self.name = name

    class FakeSession:
        def __init__(self, _path: str, sess_options=None):
            self.sess_options = sess_options

        def get_inputs(self):
            return [
                FakeInput("timm_image_image"),
                FakeInput("timm_image_image_valid_num"),
            ]

        def run(self, _out_names, input_feed):
            batch = input_feed["timm_image_image"]
            valid_num = input_feed["timm_image_image_valid_num"]

            assert batch.shape == (2, 1, 3, 224, 224)
            assert valid_num.shape == (2,)
            assert valid_num.dtype == np.int64
            assert np.all(valid_num == 1)

            emb = np.zeros((2, 384), dtype=np.float32)
            logits = np.array([[0.2, 1.4], [3.0, 1.0]], dtype=np.float32)
            return [emb, logits]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.SessionOptions = type("SessionOptions", (), {})
    fake_ort.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def numpy(self):
            return self._arr

    class Compose:
        def __init__(self, _ops):
            pass

        def __call__(self, _img):
            return FakeTensor(np.zeros((3, 224, 224), dtype=np.float32))

    fake_transforms = types.SimpleNamespace(
        Compose=Compose,
        Resize=lambda *_args, **_kwargs: None,
        CenterCrop=lambda *_args, **_kwargs: None,
        ToTensor=lambda *_args, **_kwargs: None,
        Normalize=lambda *_args, **_kwargs: None,
    )
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.transforms = fake_transforms
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", fake_transforms)

    class _FakeImage:
        def convert(self, _mode: str):
            return self

    fake_image_mod = types.SimpleNamespace(open=lambda _p: _FakeImage())
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_image_mod
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_mod)

    result = predict_onnx(
        input_df=input_df,
        images_dir=images_dir,
        onnx_path=tmp_path / "model.onnx",
        batch_size=2,
        num_threads=0,
    )

    assert result["prediction"].tolist() == [1, 0]
    assert "proba_0" in result.columns
    assert "proba_1" in result.columns
