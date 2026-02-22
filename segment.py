#!/usr/bin/env python3
"""
Segmentation script for insect extraction.

Usage:
    # SAM3 with alpha channel
    python segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_clean/ \\
        --sam3-checkpoint models/sam3_hq_vit_h.pt \\
        --segmentation-method sam3 \\
        --device auto \\
        --hint "insect" \\
        --threads 12
    
    # SAM3 with bounding box only
    python segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_bbox/ \\
        --sam3-checkpoint models/sam3_hq_vit_h.pt \\
        --segmentation-method sam3-bbox \\
        --out-image-format jpg
    
    # Otsu method (no SAM3 checkpoint required)
    python segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_otsu/ \\
        --segmentation-method otsu \\
        --out-image-format jpg
    
    # GrabCut method (no SAM3 checkpoint required)
    python segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_grabcut/ \\
        --segmentation-method grabcut \\
        --repair-strategy sam3-fill
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from datetime import datetime

# Global shutdown flag for graceful exit on Ctrl+C
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) by setting shutdown flag."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested. Finishing current image and exiting gracefully...", file=sys.stderr)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.segmentation import SegmentationProcessor


def parse_args(args=None):
    """Parse command line arguments.
    
    Args:
        args: List of arguments. If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(
        description='Segment insects from images using SAM3',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--hint', '-t',
        default='insect',
        help='Text prompt for segmentation (default: "insect")'
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Input directory containing images'
    )
    
    parser.add_argument(
        '--out_dir', '-o',
        required=True,
        help='Output directory for segmented images'
    )
    
    parser.add_argument(
        '--segmentation-method',
        default='sam3',
        choices=['sam3', 'sam3-bbox', 'otsu', 'grabcut'],
        help='''Segmentation method:
  sam3       - SAM3 segmentation with alpha channel (transparent background)
  sam3-bbox  - SAM3 segmentation with cropped bounding box (no alpha)
  otsu       - Otsu thresholding (no SAM3 checkpoint required)
  grabcut    - GrabCut algorithm (no SAM3 checkpoint required)
'''
    )
    
    parser.add_argument(
        '--sam3-checkpoint', '-c',
        required=False,
        help='Path to SAM3 checkpoint file (required for sam3/sam3-bbox methods)'
    )
    
    parser.add_argument(
        '--device', '-d',
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for inference (default: auto)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.0,
        help='Minimum confidence score for masks (default: 0.0 = no filtering)'
    )
    
    parser.add_argument(
        '--repair-strategy', '-r',
        default=None,
        choices=['opencv', 'sam3-fill'],
        help='Repair strategy for filling holes (optional)'
    )
    
    parser.add_argument(
        '--padding-ratio',
        type=float,
        default=0.0,
        help='Padding ratio for bounding box (default: 0.0 = no padding)'
    )
    
    parser.add_argument(
        '--out-image-format', '-f',
        default='png',
        choices=['png', 'jpg'],
        help='Output image format (default: png)'
    )
    
    parser.add_argument(
        '--threads', '-n',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args(args)


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    log_path = output_dir / 'log.txt'
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode='a')  # Append mode to preserve command log
        ]
    )


def save_log(output_dir: Path, args) -> None:
    """Save command log to file."""
    log_path = output_dir / 'log.txt'
    with open(log_path, 'a') as f:
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")


def main():
    """Main entry point."""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()

    # Validate thread count
    max_threads = os.cpu_count() or 8
    if args.threads > max_threads * 2:
        print(f"Warning: Thread count {args.threads} exceeds recommended maximum {max_threads * 2}", file=sys.stderr)

    # Validate input directory exists
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Check if directory has images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    has_images = any(
        f.suffix.lower() in image_extensions 
        for f in input_dir.iterdir() 
        if f.is_file()
    )
    if not has_images:
        print(f"Warning: No image files found in input directory: {input_dir}", file=sys.stderr)
    
    # Create output directory first for logging
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command log FIRST (before setting up logging handlers)
    # This writes initial metadata without overwriting later processing logs
    save_log(output_dir, args)
    
    # Setup logging with output_dir for file handler (appends to the log file)
    setup_logging(output_dir, args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting segmentation process")
    try:
        processor = SegmentationProcessor(
            sam3_checkpoint=args.sam3_checkpoint,
            device=args.device,
            hint=args.hint,
            repair_strategy=args.repair_strategy,
            confidence_threshold=args.confidence_threshold,
            padding_ratio=args.padding_ratio,
            segmentation_method=args.segmentation_method
        )
    except Exception:
        logger.exception("Failed to initialize processor")
        sys.exit(1)
    
    # Process directory
    try:
        results = processor.process_directory(
            input_dir=args.input_dir,
            output_dir=output_dir,
            num_workers=args.threads,
            disable_tqdm=True,
            output_format=args.out_image_format,
            shutdown_flag=lambda: _shutdown_requested
        )
        
        logger.info(f"Processing complete!")
        logger.info(f"  Successfully processed: {results['processed']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Output files: {len(results['output_files'])}")
        
    except Exception:
        logger.exception("Processing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
