#!/usr/bin/env python3
"""
Synthesis script for compositing target objects onto background images.

Usage:
    # Basic synthesis with 10 variations per target
    python scripts/synthesize.py \\
        --target-dir outputs/out0_black/cleaned_images/ \\
        --background-dir outputs/out0_black/repaired_images/ \\
        --out-dir outputs/synthesized/ \\
        --num-syntheses 10 \\
        --area-ratio-min 0.05 \\
        --area-ratio-max 0.20 \\
        --color-match-strength 0.5 \\
        --avoid-black-regions \\
        --threads 4
    
    # With COCO annotations (unified, default)
    python scripts/synthesize.py \\
        --target-dir outputs/out0_black/cleaned_images/ \\
        --background-dir outputs/out0_black/repaired_images/ \\
        --out-dir outputs/synthesized/ \\
        --num-syntheses 10 \\
        --annotation-output-format coco
    
    # With COCO annotations (separate files)
    python scripts/synthesize.py \\
        --target-dir outputs/out0_black/cleaned_images/ \\
        --background-dir outputs/out0_black/repaired_images/ \\
        --out-dir outputs/synthesized/ \\
        --num-syntheses 10 \\
        --annotation-output-format coco \\
        --coco-output-mode separate
    
    # With VOC Pascal annotations
    python scripts/synthesize.py \\
        --target-dir outputs/out0_black/cleaned_images/ \\
        --background-dir outputs/out0_black/repaired_images/ \\
        --out-dir outputs/synthesized/ \\
        --num-syntheses 10 \\
        --annotation-output-format voc
    
    # With VOC format and polygon segmentation
    python scripts/synthesize.py \\
        --target-dir outputs/out0_black/cleaned_images/ \\
        --background-dir outputs/out0_black/repaired_images/ \\
        --out-dir outputs/synthesized/ \\
        --num-syntheses 10 \\
        --annotation-output-format voc \\
        --avoid-black-regions
    
    # With YOLO annotations
    python scripts/synthesize.py \\
        --target-dir outputs/out0_black/cleaned_images/ \\
        --background-dir outputs/out0_black/repaired_images/ \\
        --out-dir outputs/synthesized/ \\
        --num-syntheses 10 \\
        --annotation-output-format yolo
    
    # With YOLO format and polygon segmentation
    python scripts/synthesize.py \\
        --target-dir outputs/out0_black/cleaned_images/ \\
        --background-dir outputs/out0_black/repaired_images/ \\
        --out-dir outputs/synthesized/ \\
        --num-syntheses 10 \\
        --annotation-output-format yolo \\
        --avoid-black-regions
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

if sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import setup_shutdown_handler, get_shutdown_flag, setup_logging, save_log
from src.synthesis.processor import SynthesisProcessor


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Composite target objects onto background images',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--target-dir', '-t',
        required=True,
        help='Target object images directory (cleaned, with alpha channel)'
    )

    parser.add_argument(
        '--background-dir', '-b',
        required=True,
        help='Background images directory'
    )

    parser.add_argument(
        '--out-dir', '-o',
        required=True,
        help='Output directory'
    )

    parser.add_argument(
        '--num-syntheses', '-n',
        type=int,
        default=10,
        help='Number of syntheses per target image (default: 10)'
    )

    parser.add_argument(
        '--area-ratio-min', '-a',
        type=float,
        default=0.05,
        help='Minimum area ratio (target area / background area), 0.01-0.50 (default: 0.05)'
    )

    parser.add_argument(
        '--area-ratio-max', '-x',
        type=float,
        default=0.20,
        help='Maximum area ratio (target area / background area), 0.01-0.50 (default: 0.20)'
    )

    parser.add_argument(
        '--color-match-strength', '-c',
        type=float,
        default=0.5,
        help='Color matching strength 0-1 (default: 0.5)'
    )

    parser.add_argument(
        '--avoid-black-regions', '-A',
        action='store_true',
        help='Avoid pure black regions in background'
    )

    parser.add_argument(
        '--rotate', '-r',
        type=float,
        default=0.0,
        help='Maximum random rotation degrees (default: 0, use 0 for no rotation)'
    )

    parser.add_argument(
        '--out-image-format', '-f',
        default='png',
        choices=['png', 'jpg'],
        help='Output image format (default: png)'
    )

    parser.add_argument(
        '--annotation-output-format',
        default='coco',
        choices=['coco', 'voc', 'yolo'],
        help='Output format for annotations: coco (default), voc, yolo'
    )

    parser.add_argument(
        '--coco-output-mode',
        default='unified',
        choices=['unified', 'separate'],
        help='COCO output mode: unified (single annotations.json) or separate (per-image JSON files)'
    )

    parser.add_argument(
        '--threads', '-d',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()

    max_threads = os.cpu_count() or 8
    if args.threads > max_threads * 2:
        print(
            f"Warning: Thread count {args.threads} exceeds recommended maximum {max_threads * 2}",
            file=sys.stderr
        )

    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        print(f"Error: Target directory does not exist: {target_dir}", file=sys.stderr)
        sys.exit(1)
    if not target_dir.is_dir():
        print(f"Error: Target path is not a directory: {target_dir}", file=sys.stderr)
        sys.exit(1)

    background_dir = Path(args.background_dir)
    if not background_dir.exists():
        print(f"Error: Background directory does not exist: {background_dir}", file=sys.stderr)
        sys.exit(1)
    if not background_dir.is_dir():
        print(f"Error: Background path is not a directory: {background_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_log(output_dir, args)
    logger = setup_logging(output_dir, args.verbose)

    logger.info("Starting synthesis process")

    try:
        processor = SynthesisProcessor(
            output_format=args.out_image_format,
            area_ratio_min=args.area_ratio_min,
            area_ratio_max=args.area_ratio_max,
            color_match_strength=args.color_match_strength,
            avoid_black_regions=args.avoid_black_regions,
            rotate_degrees=args.rotate,
            annotation_format=args.annotation_output_format,
            coco_output_mode=args.coco_output_mode,
        )
    except Exception:
        logger.exception("Failed to initialize processor")
        sys.exit(1)

    try:
        results = processor.process_directory(
            target_dir=target_dir,
            background_dir=background_dir,
            output_dir=output_dir,
            num_syntheses=args.num_syntheses,
            disable_tqdm=False,
            threads=args.threads,
        )

        logger.info("Processing complete!")
        logger.info(f"  Successfully processed: {results['processed']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Output files: {results['output_files']}")

    except Exception:
        logger.exception("Processing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
