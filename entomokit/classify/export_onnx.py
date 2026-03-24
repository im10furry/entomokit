"""entomokit classify export-onnx — export AutoGluon model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "export-onnx",
        help="Export an AutoGluon predictor to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, help="AutoGluon predictor directory.")
    p.add_argument("--out-dir", required=True, help="Output directory for model.onnx.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    p.add_argument(
        "--input-size", type=int, default=224, help="Model input image size (square)."
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.classification.exporter import export_onnx
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    save_log(out_dir, args)

    onnx_path = export_onnx(
        model_dir=Path(args.model_dir),
        out_dir=out_dir,
        opset=args.opset,
        input_size=args.input_size,
    )
    print(f"ONNX model saved to: {onnx_path}")
