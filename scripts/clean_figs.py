#!/usr/bin/env python3
"""
Image cleaning and deduplication script.

Usage:
    python scripts/clean_figs.py --input_dir images/raw/ --out_dir images/cleaned/
"""

import argparse
import sys
from pathlib import Path

from src.common.cli import setup_shutdown_handler, get_shutdown_flag
from src.cleaning.processor import ImageCleaner


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate images"
    )
    
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder containing output images.")

    parser.add_argument("--out_short_size", type=int, default=512, 
                       help="Shorter size of output images. Use -1 to keep original size. Default: 512")
    parser.add_argument("--out_image_format", type=str, default="jpg", 
                       choices=["jpg", "png", "tif", "pdf"], help="Output image format. Default: jpg")

    parser.add_argument("--threads", type=int, default=12, help="Number of threads to use. Default: 12")

    parser.add_argument("--keep_exif", action="store_true", help="Keep EXIF data in output images.")

    parser.add_argument("--dedup_mode", type=str, default="md5", 
                       choices=["none", "md5", "phash"], help="Deduplication mode. Default: md5")
    parser.add_argument("--phash_threshold", type=int, default=5, help="Phash threshold. Default: 5")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()
    
    try:
        cleaner = ImageCleaner(
            input_dir=args.input_dir,
            output_dir=args.out_dir,
            out_short_size=args.out_short_size,
            out_image_format=args.out_image_format,
            dedup_mode=args.dedup_mode,
            phash_threshold=args.phash_threshold,
            threads=args.threads,
            keep_exif=args.keep_exif
        )
        
        results = cleaner.process_directory(log_path="log.txt")
        
        print(f"Done. Processed {results['processed']} images, {results['errors']} errors.")
        print(f"Log saved to: log.txt")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
