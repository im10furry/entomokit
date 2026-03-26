"""entomokit doctor — environment diagnostics."""

from __future__ import annotations

import argparse

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "doctor",
        help="Check environment and dependency health.",
        description=with_examples(
            "Diagnose runtime environment and package readiness.",
            ["entomokit doctor"],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from src.doctor.service import run_doctor

    report = run_doctor()
    print("\n=== entomokit doctor ===")
    print(f"python: {report['python']}")

    print("devices:")
    for name, value in report["devices"].items():
        print(f"  {name}: {value}")

    print("packages:")
    for name, info in report["packages"].items():
        min_v = info.get("min_version")
        suffix = f", min={min_v}" if min_v else ""
        print(f"  {name}: {info['version']} ({info['status']}{suffix})")

    print("recommendations:")
    for item in report["recommendations"]:
        print(f"  - {item}")
    print("")
