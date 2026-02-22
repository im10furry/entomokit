# Insect Synthesizer

A Python-based tool for segmenting insects from clean backgrounds using multiple segmentation methods (SAM3, Otsu thresholding, GrabCut) and compositing them onto complex backgrounds for data augmentation.

## Overview

This project provides two main scripts:
- `segment.py` - Segments insects from clean background images using multiple segmentation methods
- `synthesize.py` - Composites segmented insects onto complex background images

## Features

- **Multiple Segmentation Methods**: SAM3 (with alpha channel), SAM3-bbox (cropped), Otsu thresholding, GrabCut
- **Flexible Repair Strategies**: OpenCV morphological operations or SAM3-based hole filling
- **Input Validation**: Validates input directories, image files, and parameter constraints
- **Graceful Shutdown**: Handles Ctrl+C to finish current image before exiting
- **Parallel Processing**: Multi-threaded image processing with configurable worker count
- **Comprehensive Logging**: Detailed logging with verbose mode and log file output

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Project Structure

```
.
├── src/                  # Source code
│   └── segmentation.py   # Core segmentation logic
├── tests/                # Test files
├── data/                 # Data directory (large files ignored)
├── models/               # Model weights (large files ignored)
├── outputs/              # Output files (ignored)
├── segment.py            # Segmentation script
├── synthesize.py         # Synthesis script
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## Usage

### Segment Script

#### Basic Usage

```bash
# SAM3 with alpha channel (transparent background)
python segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --segmentation-method sam3 \
    --device auto \
    --hint "insect"
```

#### All Parameters

**Required Parameters:**
- `--input_dir`, `-i`: Input directory containing images
- `--out_dir`, `-o`: Output directory for segmented images

**Optional Parameters:**
- `--hint`, `-t`: Text prompt for segmentation (default: "insect")
- `--segmentation-method`: Segmentation method
  - `sam3` - SAM3 segmentation with alpha channel (transparent background)
  - `sam3-bbox` - SAM3 segmentation with cropped bounding box (no alpha)
  - `otsu` - Otsu thresholding (no SAM3 checkpoint required)
  - `grabcut` - GrabCut algorithm (no SAM3 checkpoint required)
- `--sam3-checkpoint`, `-c`: Path to SAM3 checkpoint file (required for sam3/sam3-bbox methods)
- `--device`, `-d`: Device for inference (`auto`, `cpu`, `cuda`, `mps`)
- `--confidence-threshold`: Minimum confidence score for masks (0.0-1.0, default: 0.0)
- `--repair-strategy`, `-r`: Repair strategy for filling holes
  - `opencv` - OpenCV morphological operations
  - `sam3-fill` - SAM3-based hole filling
- `--padding-ratio`: Padding ratio for bounding box (0.0-0.5, default: 0.0)
- `--out-image-format`, `-f`: Output image format (`png`, `jpg`)
- `--threads`, `-n`: Number of parallel workers (default: 8)
- `--verbose`, `-v`: Enable verbose logging

#### Usage Examples

**1. SAM3 with Alpha Channel (Transparent Background)**

```bash
python segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --segmentation-method sam3 \
    --device auto \
    --hint "insect" \
    --threads 12
```

**2. SAM3 with Bounding Box Only (No Alpha)**

```bash
python segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/insects_bbox/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --segmentation-method sam3-bbox \
    --out-image-format jpg
```

**3. Otsu Thresholding (No SAM3 Checkpoint Required)**

```bash
python segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/insects_otsu/ \
    --segmentation-method otsu \
    --out-image-format jpg
```

**4. GrabCut Algorithm (No SAM3 Checkpoint Required)**

```bash
python segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/insects_grabcut/ \
    --segmentation-method grabcut \
    --repair-strategy sam3-fill
```

**5. With Repair Strategy and Padding**

```bash
python segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/insects_repaired/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --segmentation-method sam3 \
    --repair-strategy opencv \
    --padding-ratio 0.1 \
    --confidence-threshold 0.5 \
    --verbose
```

### Synthesize Script

```bash
python synthesize.py \
    --target_dir outputs/insects_clean/ \
    --background_dir images/backgrounds/ \
    --out_dir outputs/synthesized/ \
    --num_syntheses 100
```

## Input Validation and Error Handling

The segment script includes comprehensive input validation:

- **Directory Validation**: Checks if input directory exists and contains images
- **Parameter Validation**: Validates thread count, padding ratios, and confidence thresholds
- **Checkpoint Validation**: Ensures SAM3 checkpoint exists when required
- **Image Format Validation**: Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- **Graceful Error Handling**: Detailed error messages with logging

## Graceful Shutdown

Press `Ctrl+C` at any time to trigger a graceful shutdown:
- Current image processing completes
- Results are saved
- Clean exit with status message

## Logging

Logs are automatically saved to `log.txt` in the output directory:
- Command used and timestamp
- All parameter values
- Processing progress and results
- Errors and warnings

Enable verbose mode with `--verbose` for detailed debugging information.

## Model Requirements

For SAM3-based methods (`sam3`, `sam3-bbox`), you need:
- SAM3 checkpoint file (e.g., `sam3_hq_vit_h.pt`)
- Download from [SAM repository](https://github.com/facebookresearch/segment-anything)

For `otsu` and `grabcut` methods, no external model is required.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
