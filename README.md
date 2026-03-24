# Insect Dataset Toolkit

A Python-based toolkit for building insect image datasets. Provides a unified `entomokit` CLI with commands for segmentation, frame extraction, image cleaning, dataset splitting, image synthesis, and AutoGluon image classification.

## Overview

All functionality is accessed through a single entry point:

```
entomokit <command> [options]
```

| Command | Description |
|---------|-------------|
| `segment` | Segment insects from images (SAM3, Otsu, GrabCut) |
| `extract-frames` | Extract frames from video files |
| `clean` | Clean and deduplicate images |
| `split-csv` | Split datasets into train/val/test CSVs |
| `synthesize` | Composite insects onto background images |
| `classify train` | Train an AutoGluon image classifier |
| `classify predict` | Run inference (AutoGluon or ONNX) |
| `classify evaluate` | Evaluate model performance |
| `classify embed` | Extract embeddings + UMAP + quality metrics |
| `classify cam` | Generate GradCAM heatmaps |
| `classify export-onnx` | Export model to ONNX format |

## Features

- **Unified CLI**: Single `entomokit` entry point — no more per-script invocations
- **Multiple Segmentation Methods**: SAM3 (with alpha channel), SAM3-bbox (cropped), Otsu thresholding, GrabCut
- **Flexible Repair Strategies**: OpenCV morphological operations, SAM3-based or LaMa hole filling
- **Annotation Output**: COCO JSON, VOC Pascal XML, YOLO TXT (detcli-compatible layout)
- **Video Frame Extraction**: Multithreaded extraction with time range support
- **Image Cleaning**: Resize, deduplicate (MD5/Phash), and standardize image naming; recursive mode
- **Dataset Splitting**: Ratio or count-based train/val/test splits with stratification
- **Image Synthesis**: Advanced compositing with rotation, color matching, and black region avoidance
- **AutoGluon Classification**: Train, predict, evaluate, embed, GradCAM, and ONNX export
- **Embedding Quality Metrics**: NMI, ARI, Recall@K, kNN accuracy, mAP@R, Silhouette, UMAP visualization
- **Input Validation**: Validates input directories, image files, and parameter constraints
- **Graceful Shutdown**: Handles Ctrl+C to finish current image before exiting
- **Parallel Processing**: Multi-threaded image processing with configurable worker count
- **Comprehensive Logging**: Detailed logging with verbose mode and log file output

## Installation

```bash
pip install -e .
```

For classification commands (AutoGluon, timm, GradCAM, UMAP):

```bash
pip install -e ".[classify]"
```

## Project Structure

```
.
├── entomokit/            # Unified CLI package
│   ├── main.py           # Entry point dispatcher
│   ├── segment.py        # entomokit segment
│   ├── extract_frames.py # entomokit extract-frames
│   ├── clean.py          # entomokit clean
│   ├── split_csv.py      # entomokit split-csv
│   ├── synthesize.py     # entomokit synthesize
│   └── classify/         # entomokit classify *
│       ├── train.py
│       ├── predict.py
│       ├── evaluate.py
│       ├── embed.py
│       ├── cam.py
│       └── export_onnx.py
├── src/
│   ├── common/           # Shared utilities (CLI, annotation_writer, logging)
│   ├── classification/   # AutoGluon classification logic
│   ├── segmentation/     # Segmentation domain logic
│   ├── framing/          # Video framing domain logic
│   ├── cleaning/         # Image cleaning domain logic
│   ├── splitting/        # Dataset splitting domain logic
│   ├── synthesis/        # Image synthesis domain logic
│   └── lama/             # LaMa inpainting implementation
├── tests/                # Test files (94 tests)
├── data/                 # Data directory (large files ignored)
├── models/               # Model weights (large files ignored)
├── docs/                 # Plans, specs, change summaries
├── requirements.txt      # Python dependencies
└── setup.py              # Package setup
```

## Usage

### 1. Segment Command

Segment insects from images using multiple methods (SAM3, Otsu, GrabCut). Optionally generates annotations in COCO, VOC, or YOLO format.

#### Basic Usage

```bash
# SAM3 with alpha channel (transparent background)
entomokit segment \
    --input-dir images/clean_insects/ \
    --out-dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --segmentation-method sam3 \
    --device auto

# With COCO annotations
entomokit segment \
    --input-dir images/clean_insects/ \
    --out-dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --segmentation-method sam3 \
    --annotation-output-format coco

# With YOLO annotations and xyxy bbox format
entomokit segment \
    --input-dir images/ --out-dir outputs/ \
    --segmentation-method otsu \
    --annotation-output-format yolo \
    --coco-bbox-format xyxy
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Input directory | Required |
| `--out-dir` | Output directory | Required |
| `--segmentation-method` | `sam3`, `sam3-bbox`, `otsu`, `grabcut` | `sam3` |
| `--sam3-checkpoint` | SAM3 checkpoint path | Required for sam3/sam3-bbox |
| `--device` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--annotation-output-format` | `coco`, `voc`, `yolo` | None |
| `--coco-bbox-format` | `xywh`, `xyxy` | `xywh` |
| `--repair-strategy` | `opencv`, `sam3-fill`, `lama` | None |
| `--threads` | Parallel workers | 8 |

**Output structure (COCO example):**
```
output_dir/
├── annotations.coco.json     # COCO annotations
├── images/                   # Segmented images
│   ├── image_01.png
│   └── ...
```

**YOLO/VOC layout:**
```
output_dir/
├── images/
├── labels/                   # YOLO: .txt per image + data.yaml
└── Annotations/              # VOC: .xml per image + ImageSets/Main/
```

### 2. Extract Frames Command

Extract frames from video files. Accepts a directory or a single video file path.

```bash
# Extract from directory every 1 second
entomokit extract-frames --input-dir videos/ --out-dir frames/

# Extract from single video, time range 5s–30s
entomokit extract-frames --input-dir video.mp4 --out-dir frames/ \
    --start-time 5.0 --end-time 30.0

# Custom interval and format
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --interval 500 --out-image-format png
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Video directory or single video file | Required |
| `--out-dir` | Output directory | Required |
| `--interval` | Interval in milliseconds | 1000 |
| `--start-time` | Start time in seconds | 0 |
| `--end-time` | End time in seconds | video end |
| `--out-image-format` | jpg/png/tif | jpg |
| `--threads` | Parallel threads | 8 |
| `--max-frames` | Max frames per video | All |

### 3. Clean Command

Clean and deduplicate images with consistent naming.

```bash
# Basic (MD5 dedup)
entomokit clean --input-dir images/raw/ --out-dir images/cleaned/

# Recursive scan + perceptual hash
entomokit clean --input-dir images/ --out-dir cleaned/ \
    --recursive --dedup-mode phash --phash-threshold 5

# Resize to shorter side 512px
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size 512 --out-image-format png
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Input directory | Required |
| `--out-dir` | Output directory | Required |
| `--recursive` | Scan subdirectories | No |
| `--out-short-size` | Shorter side size (-1 = original) | 512 |
| `--dedup-mode` | `none`, `md5`, `phash` | md5 |
| `--phash-threshold` | Phash similarity threshold | 5 |
| `--out-image-format` | jpg/png/tif | jpg |
| `--threads` | Parallel threads | 12 |

### 4. Split-CSV Command

Split a labelled CSV into train / val / test files.

```bash
# Ratio split (80/10/10)
entomokit split-csv --raw-image-csv data/images.csv \
    --known-test-classes-ratio 0.1 --val-ratio 0.1 --out-dir datasets/

# Count split with image copy
entomokit split-csv --raw-image-csv data/images.csv --mode count \
    --known-test-classes-count 100 --val-count 50 \
    --copy-images --images-dir images/ --out-dir datasets/
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--raw-image-csv` | Input CSV (image, label columns) | Required |
| `--out-dir` | Output directory | Required |
| `--mode` | `ratio` or `count` | ratio |
| `--val-ratio` / `--val-count` | Validation split | None |
| `--known-test-classes-ratio` | Known-class test ratio | 0.1 |
| `--unknown-test-classes-ratio` | Unknown-class test ratio | 0 |
| `--copy-images` | Copy images into split subdirs | No |
| `--images-dir` | Source images dir (for copy) | None |
| `--seed` | Random seed | 42 |

**Output:**
```
output_dir/
├── train.csv
├── val.csv          # if --val-ratio / --val-count specified
├── test.known.csv
└── test.unknown.csv # if unknown classes configured
```

### 5. Synthesize Command

Composite target objects onto background images with rotation, color matching, and intelligent positioning.

```bash
# Basic synthesis
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --num-syntheses 10

# With COCO annotations and rotation
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --num-syntheses 10 \
    --annotation-output-format coco \
    --rotate 30

# With YOLO annotations
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --annotation-output-format yolo \
    --coco-bbox-format xyxy
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target-dir` | Target images (with alpha channel) | Required |
| `--background-dir` | Background images | Required |
| `--out-dir` | Output directory | Required |
| `--num-syntheses` | Syntheses per target | 10 |
| `--annotation-output-format` | `coco`, `voc`, `yolo` | None |
| `--coco-bbox-format` | `xywh`, `xyxy` | `xywh` |
| `--rotate` | Max rotation degrees | 0 |
| `--avoid-black-regions` | Skip dark background areas | No |
| `--color-match-strength` | 0–1 color matching | 0.5 |
| `--threads` | Parallel workers | 4 |

**Output (COCO):**
```
output_dir/
├── images/
├── annotations.coco.json
```

**Output (YOLO):**
```
output_dir/
├── images/
├── labels/
└── data.yaml
```

---

### 6. Classify Commands

All classification commands require the `classify` extras:

```bash
pip install -e ".[classify]"
```

#### `classify train`

```bash
entomokit classify train \
    --train-csv data/train.csv \
    --images-dir data/images/ \
    --out-dir runs/exp1/ \
    --base-model convnextv2_femto \
    --augment medium \
    --max-epochs 50 \
    --device auto
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train-csv` | CSV with `image`, `label` columns | Required |
| `--images-dir` | Training images directory | Required |
| `--out-dir` | Output directory | Required |
| `--base-model` | timm backbone name | `convnextv2_femto` |
| `--augment` | `none/light/medium/heavy` or JSON array | `medium` |
| `--max-epochs` | Max training epochs | 50 |
| `--time-limit` | Time limit in hours | 1.0 |
| `--focal-loss` | Enable focal loss | No |
| `--device` | `auto/cpu/cuda/mps` | `auto` |
| `--batch-size` | Batch size | 32 |

#### `classify predict`

```bash
# AutoGluon model
entomokit classify predict \
    --images-dir data/test/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/predict/

# ONNX model
entomokit classify predict \
    --input-csv test.csv \
    --onnx-model model.onnx \
    --out-dir runs/predict/
```

`--input-csv` and `--images-dir` are mutually exclusive. `--model-dir` and `--onnx-model` are mutually exclusive.

#### `classify evaluate`

```bash
entomokit classify evaluate \
    --test-csv data/test.csv \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/eval/
```

Metrics saved to `logs/evaluations.txt`: accuracy, precision/recall (macro+micro), F1, MCC, ROC-AUC.

#### `classify embed`

```bash
# Pretrained timm backbone (no training required)
entomokit classify embed \
    --images-dir data/images/ \
    --base-model convnextv2_femto \
    --label-csv data/labels.csv \
    --visualize \
    --out-dir runs/embed/

# Fine-tuned AutoGluon backbone
entomokit classify embed \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --label-csv data/labels.csv \
    --out-dir runs/embed/
```

Outputs: `embeddings.csv`, `logs/metrics.csv` (NMI, ARI, Recall@1/5/10, kNN Acc, mAP@R, Purity, Silhouette), `umap.pdf`.

#### `classify cam`

```bash
entomokit classify cam \
    --label-csv data/test.csv \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --cam-method gradcam \
    --out-dir runs/cam/ \
    --save-npy
```

Supports: `gradcam`, `gradcampp`, `layercam`, `scorecam`, `eigencam`, `ablationcam`. Auto-detects CNN vs ViT architecture. Outputs `figures/`, `arrays/`, `cam_summary.csv`. **ONNX not supported** (requires PyTorch hooks).

#### `classify export-onnx`

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ \
    --opset 17 \
    --input-size 224
```

## Common Behaviours

**Logging:** All commands save `log.txt` to the output directory (command, timestamp, all parameter values). Use `--verbose` for debug-level output.

**Graceful shutdown:** Press `Ctrl+C` — the current image finishes before exiting; partial results are saved.

**Device selection:** `--device auto` chooses CUDA → MPS → CPU automatically.

## Model Requirements

### SAM3 Model

For SAM3-based methods (`sam3`, `sam3-bbox`), download the checkpoint from Hugging Face and pass it with `--sam3-checkpoint`.

### LaMa Model

For `--repair-strategy lama`, place the Big-LaMa model at:
```
models/big-lama/
├── config.yaml
└── models/best.ckpt
```

For `otsu` and `grabcut` no external model is required.

### AutoGluon / timm (classify commands)

Install the `classify` extras — AutoGluon will download backbone weights automatically on first use.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
