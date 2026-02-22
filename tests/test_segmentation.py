# tests/test_segmentation.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.segmentation import SegmentationProcessor


def test_segmentation_processor_init():
    """Test processor initialization."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor(
            sam3_checkpoint="fake.pt",
            device="cpu"
        )
        
        assert processor.device == "cpu"
        assert processor.sam_wrapper is not None
        assert processor.hint == "insect"
        assert processor.repair_strategy is None
        assert processor.metadata_manager is not None
        assert processor.insect_category_id > 0


def test_process_single_insect():
    """Test processing single insect image."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        # Setup mock
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[30:70, 30:70] = 255
        mock_wrapper.predict_with_scores.return_value = {
            'masks': [mock_mask],
            'scores': [0.95]
        }
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu")
        
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img,
                output_dir=tmpdir,
                base_name="test"
            )
            
            assert result is not None
            assert 'masks' in result
            assert 'output_files' in result
            assert len(result['masks']) == 1
            assert len(result['output_files']) == 1
            assert result['output_files'][0].endswith('.png')
            assert 'cleaned_images' in result['output_files'][0]


def test_confidence_threshold_filtering():
    """Test filtering masks by confidence threshold."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_mask1 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask1[30:70, 30:70] = 255
        mock_mask2 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask2[50:80, 50:80] = 255
        mock_wrapper.predict_with_scores.return_value = {
            'masks': [mock_mask1, mock_mask2],
            'scores': [0.95, 0.45]
        }
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu", confidence_threshold=0.7)
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img,
                output_dir=tmpdir,
                base_name="filtered"
            )
            
            assert result is not None
            assert len(result['masks']) == 1
            assert len(result['output_files']) == 1
            assert 'cleaned_images' in result['output_files'][0]


def test_process_multiple_masks():
    """Test processing with multiple masks."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_mask1 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask1[10:30, 10:30] = 255
        mock_mask2 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask2[50:70, 50:70] = 255
        mock_wrapper.predict_with_scores.return_value = {
            'masks': [mock_mask1, mock_mask2],
            'scores': [0.95, 0.85]
        }
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu")
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img,
                output_dir=tmpdir,
                base_name="multi"
            )
            
            assert len(result['masks']) == 2
            assert len(result['output_files']) == 2
            assert all('cleaned_images' in f for f in result['output_files'])


def test_process_empty_masks():
    """Test processing with no masks found."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_wrapper.predict_with_scores.return_value = {
            'masks': [],
            'scores': []
        }
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu")
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img,
                output_dir=tmpdir,
                base_name="empty"
            )
            
            assert result is not None
            assert len(result['masks']) == 0
            assert len(result['output_files']) == 0


def test_process_image_metadata():
    """Test that metadata is properly populated."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[20:50, 30:70] = 255
        mock_wrapper.predict_with_scores.return_value = {
            'masks': [mock_mask],
            'scores': [0.95]
        }
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu")
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.process_image(
                image=img,
                output_dir=tmpdir,
                base_name="meta_test",
                original_path="/path/to/original.jpg"
            )
            
            # Check metadata was added
            assert len(processor.metadata_manager.images) == 1
            assert len(processor.metadata_manager.annotations) == 1
            
            img_meta = processor.metadata_manager.images[0]
            assert img_meta['file_name'] == 'cleaned_images/meta_test.png'
            
            ann_meta = processor.metadata_manager.annotations[0]
            assert ann_meta['category_id'] == processor.insect_category_id
            assert ann_meta['bbox'] == [30, 20, 40, 30]  # x, y, w, h (list for JSON serialization)


def test_process_directory():
    """Test processing directory of images."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[30:70, 30:70] = 255
        mock_wrapper.predict_with_scores.return_value = {
            'masks': [mock_mask],
            'scores': [0.95]
        }
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            img_dir = Path(tmpdir) / "input"
            out_dir = Path(tmpdir) / "output"
            img_dir.mkdir()
            
            # Create dummy image files
            for i in range(3):
                (img_dir / f"image{i:03d}.png").write_text("fake image data")
            
            # Mock load_image to return test image
            with patch('src.segmentation.load_image') as mock_load:
                test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                mock_load.return_value = test_img
                
                result = processor.process_directory(
                    input_dir=img_dir,
                    output_dir=out_dir
                )
                
                assert result is not None
                assert result['processed'] == 3
                assert result['failed'] == 0
                assert len(result['output_files']) == 3


def test_e2e_segment_real_insect_image():
    """End-to-end test: segment real insect image using SAM3 model."""
    # This is a real end-to-end test that uses actual insect images
    # Skip if SAM3 is not available
    pytest.importorskip("sam3", reason="SAM3 not installed")
    
    # Use real insect image
    test_image_path = Path("data/insects_raw/female_dor_1_Lucanus_brivioi.jpg")
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        processor = SegmentationProcessor(
            sam3_checkpoint="models/sam3.pt",
            device="cpu",
            hint="insect"
        )
        
        # Load real image
        from src.utils import load_image
        image = load_image(test_image_path)
        
        # Process single image
        result = processor.process_image(
            image=image,
            output_dir=tmpdir,
            base_name="test_insect"
        )
        
        # Verify results
        assert result is not None
        assert len(result['masks']) >= 1  # Should find at least one mask
        assert len(result['output_files']) >= 1
        
        # Verify output file exists and is valid
        output_path = Path(result['output_files'][0])
        assert output_path.exists()
        assert output_path.suffix == '.png'
        
        # Verify image has alpha channel
        from src.utils import load_image as load_rgba
        output_img = load_rgba(output_path)
        assert output_img.shape[2] == 4  # RGBA


def test_e2e_segment_directory_real_images():
    """End-to-end test: process directory of real insect images."""
    pytest.importorskip("sam3", reason="SAM3 not installed")
    
    # Use real insect images
    test_images = list(Path("data/insects_raw").glob("*.jpg"))
    if len(test_images) < 2:
        pytest.skip(f"Need at least 2 test images, found {len(test_images)}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        
        # Copy first 2 images to input dir
        for img_path in test_images[:2]:
            import shutil
            shutil.copy(img_path, input_dir)
        
        processor = SegmentationProcessor(
            sam3_checkpoint="models/sam3.pt",
            device="cpu",
            hint="insect"
        )
        
        result = processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        # Verify results
        assert result is not None
        assert result['processed'] == 2
        assert result['failed'] == 0
        assert len(result['output_files']) >= 2  # At least one per image
        
        # Verify metadata was saved
        metadata_path = output_dir / "annotations.json"
        assert metadata_path.exists()
        
        # Verify metadata structure
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert 'images' in metadata
        assert 'annotations' in metadata
        assert 'categories' in metadata
        assert len(metadata['categories']) > 0
        assert metadata['categories'][0]['name'] == 'insect'
