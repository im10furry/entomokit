"""Tests for src/classification/utils.py"""

import pytest
from src.classification.utils import resolve_augment, AUGMENT_TRANSFORMS


def test_preset_none():
    assert resolve_augment("none") == AUGMENT_TRANSFORMS["none"]


def test_preset_medium():
    assert resolve_augment("medium") == AUGMENT_TRANSFORMS["medium"]


def test_preset_heavy():
    assert resolve_augment("heavy") == AUGMENT_TRANSFORMS["heavy"]


def test_json_custom_valid():
    result = resolve_augment('["random_resize_crop", "color_jitter"]')
    assert result == ["random_resize_crop", "color_jitter"]


def test_invalid_preset_raises():
    with pytest.raises(ValueError, match="Invalid --augment"):
        resolve_augment("ultra")


def test_invalid_json_raises():
    with pytest.raises(ValueError, match="Invalid --augment"):
        resolve_augment("not_json_or_preset")


def test_unknown_transform_in_json_raises():
    with pytest.raises(ValueError, match="Unknown transform name"):
        resolve_augment('["random_resize_crop", "super_augment"]')


def test_json_not_array_raises():
    with pytest.raises(ValueError, match="must be an array"):
        resolve_augment('{"key": "value"}')
