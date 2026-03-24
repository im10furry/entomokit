"""classify command group — AutoGluon image classification."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "classify",
        help="Image classification commands (AutoGluon).",
    )
    sub = p.add_subparsers(dest="subcommand", metavar="<subcommand>")
    sub.required = True

    from entomokit.classify import train, predict, evaluate, embed, cam, export_onnx

    train.register(sub)
    predict.register(sub)
    evaluate.register(sub)
    embed.register(sub)
    cam.register(sub)
    export_onnx.register(sub)
