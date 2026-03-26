"""Doctor service: checks environment and dependency health."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PackageRule:
    name: str
    min_version: str | None = None
    required: bool = True
    note: str = ""


PACKAGE_RULES = [
    PackageRule(
        "torch", required=True, note="core inference and classify device checks"
    ),
    PackageRule(
        "opencv-python", required=True, note="extract-frames / clean / augment"
    ),
    PackageRule("scikit-image", required=False, note="segment / synthesize extras"),
    PackageRule("pandas", required=False, note="split-csv"),
    PackageRule("imagehash", required=False, note="clean --dedup-mode phash"),
    PackageRule("albumentations", required=False, note="augment"),
    PackageRule("timm", required=False, note="classify"),
    PackageRule("onnxruntime", required=False, note="classify predict --onnx-model"),
    PackageRule(
        "autogluon.multimodal",
        min_version="1.4.0",
        required=False,
        note="classify (AutoMM)",
    ),
    PackageRule(
        "autogluon",
        min_version="1.4.0",
        required=False,
        note="optional umbrella package",
    ),
]


def _check_pkg_version(name: str) -> str:
    """Return version string or 'NOT INSTALLED' for a package."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass

    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return "NOT INSTALLED"


def _version_tuple(version: str) -> tuple[int, ...]:
    clean = version.split("+")[0]
    tokens = clean.replace("-", ".").split(".")
    values: list[int] = []
    for token in tokens:
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        values.append(int(digits))
    return tuple(values)


def _is_below_min(version: str, min_version: str | None) -> bool:
    if version == "NOT INSTALLED" or min_version is None:
        return False
    return _version_tuple(version) < _version_tuple(min_version)


def _device_report() -> dict[str, str]:
    devices: dict[str, str] = {"cpu": "available"}
    try:
        import torch

        if torch.cuda.is_available():
            devices["cuda"] = torch.version.cuda or "available"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices["mps"] = "available"
    except ImportError:
        pass
    return devices


def run_doctor() -> dict:
    """Run environment health checks and recommendations."""
    packages: dict[str, dict[str, str | bool]] = {}
    recommendations: list[str] = []

    for rule in PACKAGE_RULES:
        version = _check_pkg_version(rule.name)
        status = "ok"
        if version == "NOT INSTALLED":
            status = "missing"
        elif _is_below_min(version, rule.min_version):
            status = "outdated"

        pkg_info: dict[str, str | bool] = {
            "version": version,
            "status": status,
            "required": rule.required,
        }
        if rule.min_version is not None:
            pkg_info["min_version"] = rule.min_version
        if rule.note:
            pkg_info["note"] = rule.note
        packages[rule.name] = pkg_info

    automm_version = str(packages["autogluon.multimodal"]["version"])
    if automm_version == "NOT INSTALLED":
        recommendations.append(
            "Install classify dependency: pip install 'autogluon.multimodal>=1.4.0'"
        )
    elif _is_below_min(automm_version, "1.4.0"):
        recommendations.append(
            "Upgrade AutoMM: pip install -U 'autogluon.multimodal>=1.4.0'"
        )

    if str(packages["albumentations"]["version"]) == "NOT INSTALLED":
        recommendations.append("Install augment dependency: pip install albumentations")

    if str(packages["imagehash"]["version"]) == "NOT INSTALLED":
        recommendations.append("Install phash support: pip install imagehash")

    if not recommendations:
        recommendations.append("Environment looks good for current entomokit features.")

    return {
        "python": sys.version,
        "devices": _device_report(),
        "packages": packages,
        "recommendations": recommendations,
    }
