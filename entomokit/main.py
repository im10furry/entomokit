"""entomokit — unified CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="entomokit",
        description="A toolkit for building insect image datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # Lazy imports keep startup fast and avoid heavy optional deps at import time
    from entomokit import segment as _segment
    from entomokit import extract_frames as _extract_frames
    from entomokit import clean as _clean
    from entomokit import split_csv as _split_csv
    from entomokit import synthesize as _synthesize
    from entomokit.classify import register as _register_classify

    _segment.register(subparsers)
    _extract_frames.register(subparsers)
    _clean.register(subparsers)
    _split_csv.register(subparsers)
    _synthesize.register(subparsers)
    _register_classify(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
