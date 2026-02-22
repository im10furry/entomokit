#!/usr/bin/env python3
"""
合成脚本:将干净的目标物体图片合成到背景图片上

Usage:
    python synthesize.py \
        --target-dir outputs/out0_black/cleaned_images/ \
        --background-dir outputs/out0_black/repaired_images/ \
        --out-dir outputs/synthesized/ \
        --num-syntheses 10 \
        --scale-min 0.10 \
        --scale-max 0.50 \
        --color-match-strength 0.5 \
        --avoid-black-regions \
        --threads 4
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Union
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from skimage import exposure
from skimage.color import rgb2lab, lab2rgb

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='合成目标物体到背景图片'
    )
    
    parser.add_argument(
        '--target-dir', '-t',
        required=True,
        help='目标物体图片目录 (干净的分割图片,带alpha通道)'
    )
    
    parser.add_argument(
        '--background-dir', '-b',
        required=True,
        help='背景图片目录'
    )
    
    parser.add_argument(
        '--out-dir', '-o',
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--num-syntheses', '-n',
        type=int,
        default=10,
        help='每个目标物体合成的数量 (default: 10)'
    )
    
    parser.add_argument(
        '--scale-min', '-s',
        type=float,
        default=0.10,
        help='目标尺度最小比例 (default: 0.10)'
    )
    
    parser.add_argument(
        '--scale-max', '-x',
        type=float,
        default=0.50,
        help='目标尺度最大比例 (default: 0.50)'
    )
    
    parser.add_argument(
        '--color-match-strength', '-c',
        type=float,
        default=0.5,
        help='颜色匹配强度 0-1 (default: 0.5)'
    )
    
    parser.add_argument(
        '--avoid-black-regions', '-a',
        action='store_true',
        help='避让纯黑色区域'
    )
    
    parser.add_argument(
        '--out-image-format', '-f',
        default='png',
        choices=['png', 'jpg'],
        help='输出图片格式 (default: png)'
    )
    
    parser.add_argument(
        '--threads', '-d',
        type=int,
        default=1,
        help='并行线程数 (default: 1)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def save_log(output_dir: Path, args) -> None:
    """Save command log to file."""
    log_path = output_dir / 'log.txt'
    with open(log_path, 'w') as f:
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array (H, W, C)
    """
    image_path = Path(image_path)
    
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            return np.array(img)
        elif img.mode == 'RGB':
            return np.array(img)
        else:
            img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        raise ValueError(f"Could not load image: {image_path}: {e}")


def load_images_from_directory(
    directory: Union[str, Path],
    desc: str = "Loading images"
) -> List[np.ndarray]:
    """
    Load all images from directory.
    
    Args:
        directory: Directory containing images
        desc: Progress bar description
    
    Returns:
        List of image arrays
    """
    directory = Path(directory)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [
        p for p in directory.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    images = []
    for img_path in tqdm(image_paths, desc=desc, leave=False):
        try:
            img = load_image(img_path)
            images.append(img)
        except Exception as e:
            logging.warning(f"Failed to load {img_path}: {e}")
    
    return images


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    format: str = 'png',
    quality: int = 90
) -> None:
    """
    Save image to file.
    
    Args:
        image: Image array
        output_path: Output path
        format: Output format ('png' or 'jpg')
        quality: Compression quality (0-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if image.shape[2] == 4 and format.lower() == 'jpg':
        img_pil = Image.fromarray(image[:, :, :3], mode='RGB')
    else:
        img_pil = Image.fromarray(image, mode='RGBA' if image.shape[2] == 4 else 'RGB')
    
    if format.lower() == 'jpg':
        img_pil.save(output_path, 'JPEG', quality=quality, optimize=True)
    else:
        img_pil.save(output_path, 'PNG', optimize=True, compress_level=9 - (quality // 11))


def calculate_scale_factor(
    background_shape: Tuple[int, ...],
    mask_area: int,
    scale_ratio: float
) -> float:
    """
    Calculate scale factor to achieve target area ratio.
    
    Args:
        background_shape: Background image shape (H, W, C)
        mask_area: Area of target mask in pixels
        scale_ratio: Target ratio of target area to background area
    
    Returns:
        Scale factor to apply to target
    """
    bg_area = background_shape[0] * background_shape[1]
    target_pixel_area = int(bg_area * scale_ratio)
    
    scale_factor = np.sqrt(target_pixel_area / mask_area)
    
    return float(scale_factor)


def _is_region_black(
    background: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int
) -> bool:
    """
    Check if region is pure black [0, 0, 0].
    
    Args:
        background: Background image
        x, y: Top-left coordinates
        w, h: Width and height
    
    Returns:
        Whether the region is mostly black
    """
    region = background[y:y+h, x:x+w]
    
    if region.shape[2] == 4:
        rgb = region[:, :, :3]
    else:
        rgb = region
    
    is_black = np.all(rgb == [0, 0, 0], axis=-1)
    black_ratio = np.sum(is_black) / is_black.size
    
    return black_ratio > 0.5


def random_position_with_constraint(
    background: np.ndarray,
    target_shape: Tuple[int, ...],
    edge_margin: float = 0.1,
    avoid_black: bool = False
) -> Tuple[int, int]:
    """
    Find random position with constraint (avoid edges and optionally black regions).
    
    Args:
        background: Background image
        target_shape: Target shape (H, W) or (H, W, C)
        edge_margin: Minimum distance from edges (as ratio of bg size)
        avoid_black: Whether to avoid black regions
    
    Returns:
        (x, y) position for top-left corner
    """
    bg_h, bg_w = background.shape[:2]
    target_h, target_w = target_shape[:2]
    
    margin_x = int(bg_w * edge_margin)
    margin_y = int(bg_h * edge_margin)
    
    min_x = margin_x
    max_x = bg_w - target_w - margin_x
    min_y = margin_y
    max_y = bg_h - target_h - margin_y
    
    if max_x <= min_x:
        min_x = 0
        max_x = max(1, bg_w - target_w)
    if max_y <= min_y:
        min_y = 0
        max_y = max(1, bg_h - target_h)
    
    max_attempts = 100
    for _ in range(max_attempts):
        x = np.random.randint(min_x, max_x) if max_x > min_x else min_x
        y = np.random.randint(min_y, max_y) if max_y > min_y else min_y
        
        if not avoid_black:
            return x, y
        
        if not _is_region_black(background, x, y, target_w, target_h):
            return x, y
    
    return min_x, min_y


def paste_with_alpha(
    background: np.ndarray,
    target_rgba: np.ndarray,
    x: int,
    y: int
) -> np.ndarray:
    """
    Paste target with alpha blending onto background.
    
    Args:
        background: Background image (H, W, 3)
        target_rgba: Target with alpha (h, w, 4)
        x, y: Top-left coordinates
    
    Returns:
        Blended image (H, W, 3)
    """
    result = background.copy()
    
    h, w = target_rgba.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)
    
    src_x1 = x1 - x
    src_y1 = y1 - y
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)
    
    if x1 >= x2 or y1 >= y2:
        return result
    
    bg_region = result[y1:y2, x1:x2]
    target_region = target_rgba[src_y1:src_y2, src_x1:src_x2]
    
    alpha = target_region[:, :, 3:4].astype(np.float32) / 255.0
    
    blended = (alpha * target_region[:, :, :3] + (1 - alpha) * bg_region).astype(np.uint8)
    
    result[y1:y2, x1:x2] = blended
    
    return result


def match_lab_histograms(
    image: np.ndarray,
    reference: np.ndarray,
    strength: float = 0.5
) -> np.ndarray:
    """
    Match LAB histograms between image and reference.
    
    Args:
        image: Image to adjust (H, W, 3)
        reference: Reference image for histogram matching
        strength: Matching strength 0-1 (0=no change, 1=full match)
    
    Returns:
        Adjusted image
    """
    if strength <= 0:
        return image
    
    image_lab = rgb2lab(image)
    reference_lab = rgb2lab(reference)
    
    result_lab = image_lab.copy()
    
    for channel in range(3):
        image_channel = image_lab[:, :, channel]
        ref_channel = reference_lab[:, :, channel]
        
        matched = exposure.match_histograms(image_channel, ref_channel)
        
        result_lab[:, :, channel] = (
            (1 - strength) * image_channel +
            strength * matched
        )
    
    result = (lab2rgb(result_lab) * 255).astype(np.uint8)
    
    return result


def synthesize_single(
    target_image: np.ndarray,
    background: np.ndarray,
    scale_ratio: float,
    color_match_strength: float,
    avoid_black: bool = False
) -> Optional[np.ndarray]:
    """
    Perform single synthesis.
    
    Args:
        target_image: Target image with alpha (H, W, 4)
        background: Background image (H, W, 3)
        scale_ratio: Scale ratio (0-1)
        color_match_strength: Color matching strength
        avoid_black: Whether to avoid black regions
    
    Returns:
        Synthesized image or None if failed
    """
    try:
        if target_image.shape[2] == 4:
            mask = target_image[:, :, 3]
        else:
            mask = np.ones(target_image.shape[:2], dtype=np.uint8) * 255
        
        mask_area = int(np.sum(mask > 0))
        
        scale_factor = calculate_scale_factor(
            background.shape,
            mask_area,
            scale_ratio
        )
        
        new_h = int(target_image.shape[0] * scale_factor)
        new_w = int(target_image.shape[1] * scale_factor)
        
        if new_h < 10 or new_w < 10:
            logging.warning(f"Target too small after scaling: {new_w}x{new_h}")
            return None
        
        target_resized = cv2.resize(
            target_image, (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        x, y = random_position_with_constraint(
            background,
            target_resized.shape,
            edge_margin=0.05,
            avoid_black=avoid_black
        )
        
        result = paste_with_alpha(background, target_resized, x, y)
        
        if color_match_strength > 0:
            result = match_lab_histograms(result, background, color_match_strength)
        
        return result
    
    except Exception as e:
        logging.error(f"Synthesis failed: {e}")
        return None


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting synthesis process")
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_log(output_dir, args)
    
    logger.info("Loading target images...")
    target_images = load_images_from_directory(
        args.target_dir,
        desc="Loading targets"
    )
    
    logger.info("Loading background images...")
    background_images = load_images_from_directory(
        args.background_dir,
        desc="Loading backgrounds"
    )
    
    if not target_images:
        logger.error("No target images found!")
        sys.exit(1)
    
    if not background_images:
        logger.error("No background images found!")
        sys.exit(1)
    
    logger.info(f"Targets: {len(target_images)}, Backgrounds: {len(background_images)}")
    
    synthesis_id = 0
    total_syntheses = len(target_images) * args.num_syntheses
    
    with tqdm(total=total_syntheses, desc="Synthesizing") as pbar:
        for target in target_images:
            for i in range(args.num_syntheses):
                background = random.choice(background_images)
                
                scale_ratio = random.uniform(args.scale_min, args.scale_max)
                
                result = synthesize_single(
                    target,
                    background,
                    scale_ratio,
                    args.color_match_strength,
                    avoid_black=args.avoid_black_regions
                )
                
                if result is None:
                    continue
                
                output_path = output_dir / f"synth_{synthesis_id:06d}.{args.out_image_format}"
                save_image(result, output_path, format=args.out_image_format)
                
                synthesis_id += 1
                pbar.update(1)
    
    logger.info(f"Synthesis complete! Created {synthesis_id} images")


if __name__ == '__main__':
    main()
