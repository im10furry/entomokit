"""classify command group — implemented in Phase 3."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the classify command group (stub)."""
    p = subparsers.add_parser(
        "classify",
        help="Image classification commands (AutoGluon). Coming in Phase 3.",
    )
    sub = p.add_subparsers(dest="subcommand", metavar="<subcommand>")
    sub.required = True
    # Subcommands registered in Phase 3
