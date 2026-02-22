# tests/test_utils.py
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image
from src.utils import (
    apply_mask_with_alpha,
    load_image,
    save_image_rgba,
)

def test_load_image_returns_numpy_array():
    """Test that load_image returns a numpy array."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGB', (100, 100), color='red')
        img.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        loaded = load_image(tmp_path)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == (100, 100, 3)
        assert loaded[0, 0, 0] > 250  # Red channel should be near 255
    finally:
        os.unlink(tmp_path)

def test_apply_mask_with_alpha_creates_rgba():
    """Test that apply_mask_with_alpha creates RGBA image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    
    result = apply_mask_with_alpha(img, mask)
    
    assert result.shape == (100, 100, 4)
    assert result.dtype == np.uint8

def test_apply_mask_with_alpha_applies_transparency():
    """Test that mask properly applies transparency."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Square in center
    
    result = apply_mask_with_alpha(img, mask)
    
    # Check corners are transparent
    assert result[0, 0, 3] == 0
    assert result[99, 99, 3] == 0
    # Check center is opaque
    assert result[50, 50, 3] == 255


def test_apply_mask_with_alpha_with_float_mask():
    """Test mask normalization with float values in 0-1 range."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0  # Square in center with float values
    
    result = apply_mask_with_alpha(img, mask)
    
    # Check corners are transparent
    assert result[0, 0, 3] == 0
    # Check center is opaque
    assert result[50, 50, 3] == 255
