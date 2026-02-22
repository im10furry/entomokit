import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from synthesize import (
    calculate_scale_factor,
    random_position_with_constraint,
    paste_with_alpha,
    load_image,
    save_image,
    match_lab_histograms,
    _is_region_black
)


class TestCalculateScaleFactor:
    """Test scale factor calculation."""
    
    def test_scale_factor_basic(self):
        """Test basic scale factor calculation."""
        bg_shape = (1000, 1000, 3)
        mask_area = 2500  # 50x50
        scale_ratio = 0.10
        
        scale_factor = calculate_scale_factor(bg_shape, mask_area, scale_ratio)
        
        expected = np.sqrt((1000000 * 0.10) / 2500)
        assert 6.0 < scale_factor < 7.0
    
    def test_scale_factor_different_ratios(self):
        """Test scale factor for different ratios."""
        bg_shape = (800, 600, 3)
        mask_area = 10000
        
        scale_factor_05 = calculate_scale_factor(bg_shape, mask_area, 0.05)
        scale_factor_10 = calculate_scale_factor(bg_shape, mask_area, 0.10)
        
        assert scale_factor_05 < scale_factor_10
    
    def test_scale_factor_zero_ratio(self):
        """Test scale factor with zero ratio."""
        bg_shape = (1000, 1000, 3)
        mask_area = 5000
        
        scale_factor = calculate_scale_factor(bg_shape, mask_area, 0.0)
        assert scale_factor == 0.0


class TestIsRegionBlack:
    """Test black region detection."""
    
    def test_pure_black_region(self):
        """Test detection of pure black region."""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        
        is_black = _is_region_black(bg, 10, 10, 50, 50)
        assert is_black == True
    
    def test_non_black_region(self):
        """Test detection of non-black region."""
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        is_black = _is_region_black(bg, 10, 10, 50, 50)
        assert is_black == False
    
    def test_mixed_region(self):
        """Test detection of mixed region."""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        bg[20:80, 20:80] = [100, 100, 100]
        
        is_black = _is_region_black(bg, 10, 10, 50, 50)
        assert is_black == False
    
    def test_black_with_alpha(self):
        """Test black detection with alpha channel."""
        bg = np.zeros((100, 100, 4), dtype=np.uint8)
        bg[:, :, 3] = 255
        
        is_black = _is_region_black(bg, 10, 10, 50, 50)
        assert is_black == True


class TestRandomPositionConstraint:
    """Test position constraint algorithm."""
    
    def test_position_within_margin(self):
        """Test position stays within margin."""
        bg = np.ones((200, 200, 3), dtype=np.uint8) * 255
        target_shape = (50, 50, 3)
        
        x, y = random_position_with_constraint(bg, target_shape, edge_margin=0.1)
        
        margin_x = int(200 * 0.1)
        margin_y = int(200 * 0.1)
        
        assert x >= margin_x
        assert y >= margin_y
    
    def test_avoid_black_enabled(self):
        """Test avoiding black regions when enabled."""
        bg = np.zeros((200, 200, 3), dtype=np.uint8)
        bg[50:150, 50:150] = [255, 255, 255]
        
        target_shape = (30, 30, 3)
        
        for _ in range(10):
            x, y = random_position_with_constraint(bg, target_shape, avoid_black=True)
            region = bg[y:y+30, x:x+30]
            assert not _is_region_black(bg, x, y, 30, 30)
    
    def test_avoid_black_disabled(self):
        """Test when black avoidance is disabled."""
        bg = np.zeros((200, 200, 3), dtype=np.uint8)
        target_shape = (50, 50, 3)
        
        x, y = random_position_with_constraint(bg, target_shape, avoid_black=False)
        assert isinstance(x, int)
        assert isinstance(y, int)


class TestPasteWithAlpha:
    """Test alpha blending."""
    
    def test_paste_with_full_alpha(self):
        """Test pasting with full opacity."""
        bg = np.ones((200, 200, 3), dtype=np.uint8) * 100
        target = np.ones((50, 50, 4), dtype=np.uint8) * 255
        
        result = paste_with_alpha(bg, target, x=50, y=50)
        
        assert result.shape == (200, 200, 3)
        region = result[50:100, 50:100]
        assert np.all(region == 255)
    
    def test_paste_with_zero_alpha(self):
        """Test pasting with transparent alpha."""
        bg = np.ones((200, 200, 3), dtype=np.uint8) * 100
        target = np.ones((50, 50, 4), dtype=np.uint8) * 255
        target[:, :, 3] = 0
        
        result = paste_with_alpha(bg, target, x=50, y=50)
        
        assert result.shape == (200, 200, 3)
        region = result[50:100, 50:100]
        assert np.all(region == 100)
    
    def test_paste_clipping(self):
        """Test pasting at edges (clipping)."""
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 100
        target = np.ones((50, 50, 4), dtype=np.uint8) * 255
        
        result = paste_with_alpha(bg, target, x=-10, y=-10)
        
        assert result.shape == (100, 100, 3)
    
    def test_paste_outside_bounds(self):
        """Test pasting completely outside bounds."""
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 100
        target = np.ones((50, 50, 4), dtype=np.uint8) * 255
        
        result = paste_with_alpha(bg, target, x=200, y=200)
        
        assert result.shape == (100, 100, 3)
        assert np.all(result == 100)


class TestMatchLabHistograms:
    """Test LAB histogram matching."""
    
    def test_no_matching_strength(self):
        """Test with zero matching strength."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        ref = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        result = match_lab_histograms(img, ref, strength=0.0)
        
        assert np.all(result == img)
    
    def test_full_matching_strength(self):
        """Test with full matching strength."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        ref = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        result = match_lab_histograms(img, ref, strength=1.0)
        
        assert result.shape == img.shape
    
    def test_partial_matching(self):
        """Test with partial matching strength."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        ref = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        result = match_lab_histograms(img, ref, strength=0.5)
        
        assert result.shape == img.shape
        assert not np.all(result == img)
        assert not np.all(result == ref)


class TestIntegration:
    """Integration tests."""
    
    def test_full_synthesize_workflow(self):
        """Test full synthesis workflow with real images."""
        target_dir = Path("outputs/out0_black/cleaned_images")
        background_dir = Path("outputs/out0_black/repaired_images")
        
        if not target_dir.exists() or not background_dir.exists():
            pytest.skip("Test data not available")
        
        target_images = []
        for img_path in list(target_dir.glob("*.png"))[:2]:
            target_images.append(load_image(img_path))
        
        background_images = []
        for img_path in list(background_dir.glob("*"))[:2]:
            try:
                background_images.append(load_image(img_path))
            except:
                continue
        
        assert len(target_images) > 0
        assert len(background_images) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
