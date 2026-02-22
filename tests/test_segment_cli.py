# tests/test_segment_cli.py
"""Tests for segment.py CLI."""
import pytest
import sys
import tempfile
import os
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test data path
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "insects"


def test_parse_args_basic():
    """Test basic argument parsing."""
    from scripts.segment import parse_args

    args = parse_args([
        '--input_dir', '/input',
        '--out_dir', '/output',
        '--sam3-checkpoint', 'model.pt'
    ])

    assert args.input_dir == '/input'
    assert args.out_dir == '/output'
    assert args.sam3_checkpoint == 'model.pt'
    assert args.device == 'auto'
    assert args.hint == 'insect'
    assert args.threads == 8
    assert args.lama_mask_dilate == 0


def test_parse_args_verbose():
    """Test verbose flag parsing."""
    from scripts.segment import parse_args

    args = parse_args([
        '--input_dir', '/input',
        '--out_dir', '/output',
        '--sam3-checkpoint', 'model.pt',
        '--verbose'
    ])

    assert args.verbose is True


def test_parse_args_custom_device():
    """Test custom device argument."""
    from scripts.segment import parse_args

    args = parse_args([
        '--input_dir', '/input',
        '--out_dir', '/output',
        '--sam3-checkpoint', 'model.pt',
        '--device', 'cpu'
    ])

    assert args.device == 'cpu'


def test_parse_args_custom_threads():
    """Test custom threads argument."""
    from scripts.segment import parse_args

    args = parse_args([
        '--input_dir', '/input',
        '--out_dir', '/output',
        '--sam3-checkpoint', 'model.pt',
        '--threads', '12'
    ])

    assert args.threads == 12


def test_parse_args_all_options():
    """Test all optional arguments."""
    from scripts.segment import parse_args

    args = parse_args([
        '--input_dir', '/input',
        '--out_dir', '/output',
        '--sam3-checkpoint', 'model.pt',
        '--device', 'mps',
        '--hint', 'beetle',
        '--repair-strategy', 'opencv',
        '--out-image-format', 'png',
        '--threads', '4',
        '--verbose',
        '--lama-mask-dilate', '2'
    ])

    assert args.input_dir == '/input'
    assert args.out_dir == '/output'
    assert args.sam3_checkpoint == 'model.pt'
    assert args.device == 'mps'
    assert args.hint == 'beetle'
    assert args.repair_strategy == 'opencv'
    assert args.out_image_format == 'png'
    assert args.threads == 4
    assert args.verbose is True
    assert args.lama_mask_dilate == 2


def test_setup_logging_verbose():
    """Test setup_logging with verbose."""
    import logging
    logging.root.handlers = []
    
    from pathlib import Path
    from src.common.cli import setup_logging
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_logging(Path(tmpdir), verbose=True)
    
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG


def test_setup_logging_default():
    """Test setup_logging with default level."""
    import logging
    logging.root.handlers = []
    
    from pathlib import Path
    from src.common.cli import setup_logging
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_logging(Path(tmpdir), verbose=False)
    
    logger = logging.getLogger()
    assert logger.level == logging.INFO


def test_save_log():
    """Test save_log function."""
    from src.common.cli import save_log
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create mock args
        mock_args = Mock()
        mock_args.input_dir = '/input'
        mock_args.out_dir = '/output'
        mock_args.sam3_checkpoint = 'model.pt'
        
        save_log(output_dir, mock_args)
        
        log_file = output_dir / 'log.txt'
        assert log_file.exists()
        
        content = log_file.read_text()
        assert 'Command:' in content
        assert 'Timestamp:' in content
        assert 'Arguments:' in content


def test_segment_cli_main_success():
    """Test main() with mock processor."""
    from scripts.segment import main
    import argparse
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        
        # Create a test image
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(input_dir / "test.jpg")
        
        output_dir = Path(tmpdir) / "output"
        
        with patch('scripts.segment.SegmentationProcessor') as MockProcessor:
            mock_processor = MagicMock()
            mock_processor.process_directory.return_value = {
                'processed': 1,
                'failed': 0,
                'output_files': ['output/test.png']
            }
            MockProcessor.return_value = mock_processor
            
            with patch('sys.argv', [
                'segment.py',
                '--input_dir', str(input_dir),
                '--out_dir', str(output_dir),
                '--sam3-checkpoint', 'model.pt',
                '--lama-mask-dilate', '3'
            ]):
                main()

            call_args = MockProcessor.call_args
            assert call_args is not None
            assert call_args.kwargs['lama_mask_dilate'] == 3


def test_parse_args_with_test_data():
    """Test parsing with actual test data path."""
    from scripts.segment import parse_args
    
    args = parse_args([
        '--input_dir', str(TEST_DATA_DIR),
        '--out_dir', '/output',
        '--sam3-checkpoint', 'model.pt'
    ])
    
    assert args.input_dir == str(TEST_DATA_DIR)
    assert Path(args.input_dir).exists()


def test_segment_cli_with_real_test_image():
    """Test segment.py with real test image from insects_raw."""
    import tempfile
    from scripts.segment import main
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = TEST_DATA_DIR
        output_dir = Path(tmpdir) / "output"
        
        # Mock SegmentationProcessor to avoid needing actual model
        with patch('scripts.segment.SegmentationProcessor') as MockProcessor:
            mock_processor = MagicMock()
            # Simulate processing one image
            mock_processor.process_directory.return_value = {
                'processed': 1,
                'failed': 0,
                'output_files': [str(output_dir / "test.png")]
            }
            MockProcessor.return_value = mock_processor
            
            # Mock sys.argv for main()
            with patch('sys.argv', [
                'segment.py',
                '--input_dir', str(input_dir),
                '--out_dir', str(output_dir),
                '--sam3-checkpoint', 'models/sam3.pt'
            ]):
                main()
            
            # Verify processor was initialized with correct checkpoint
            call_args = MockProcessor.call_args
            assert call_args is not None
            assert call_args.kwargs['sam3_checkpoint'] == 'models/sam3.pt'
            assert call_args.kwargs['device'] == 'auto'
            assert call_args.kwargs['hint'] == 'insect'


def test_parse_args_required_args_fail():
    """Test that missing required args raises error."""
    from scripts.segment import parse_args
    
    with pytest.raises(SystemExit):
        parse_args(['--input_dir', '/input'])
