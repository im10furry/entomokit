# Unified Scripts Architecture Design

**Date:** 2026-02-16  
**Status:** Approved

## Overview

This document describes the new unified architecture for organizing the insect image processing pipeline scripts. The goal is to maintain script independence while sharing common utilities and dependencies.

## Architecture Principles

### 1. Script Independence
Each script remains fully independent and can be:
- Run standalone without installing other scripts' dependencies
- Imported as a module in other code
- Deployed individually

### 2. Shared Common Library
Common utilities are extracted to `src/common/`:
- `cli.py` - Command-line infrastructure (shutdown handler, logging, argument parsing)
- `validators.py` - Directory/file validation, image/video detection
- `logging.py` - Standardized logging configuration

### 3. Domain-Driven Organization
Each script category has its own domain module:
- `src/segmentation/` - Insect segmentation (SAM3, Otsu, GrabCut)
- `src/framing/` - Video frame extraction
- `src/cleaning/` - Image cleaning and deduplication
- `src/splitting/` - Dataset splitting

### 4. Flexible Dependencies
Dependencies are managed via extras in `setup.py`:
```bash
pip install .[segmentation]    # For segment.py, synthesize.py
pip install .[cleaning]        # For clean_figs.py
pip install .[video]           # For extract_frames.py
pip install .[data]            # For split_dataset.py
pip install .[dev]             # Development dependencies
```

## Directory Structure

```
imagekit/
├── scripts/              # CLI entry points (thin wrappers)
│   ├── __init__.py
│   ├── segment.py
│   ├── synthesize.py
│   ├── clean_figs.py
│   ├── extract_frames.py
│   └── split_dataset.py
├── src/
│   ├── __init__.py
│   ├── common/          # Shared utilities
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── validators.py
│   │   └── logging.py
│   ├── segmentation/    # Segmentation domain
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── framing/         # Video framing domain
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── cleaning/        # Image cleaning domain
│   │   ├── __init__.py
│   │   └── processor.py
│   └── splitting/       # Dataset splitting domain
│       ├── __init__.py
│       └── splitter.py
├── docs/
│   └── plans/
├── old_scripts/         # Original scripts (kept for reference)
├── tests/
├── data/
├── models/
├── outputs/
├── requirements.txt     # Base dependencies only
└── setup.py             # With extras_require
```

## Script Structure

Each script follows this pattern (thin wrapper ~30-50 lines):

```python
#!/usr/bin/env python3
"""Script description."""

import argparse
from src.common.cli import setup_shutdown_handler, get_shutdown_flag, setup_logging, save_log
from src.{domain}.{module} import {ProcessorClass}


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='...')
    # ... argument definitions
    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()
    
    processor = {ProcessorClass}(...)
    results = processor.process(...)
    
    save_log(output_dir, args)
    logger.info(f"Processing complete: {results}")


if __name__ == '__main__':
    main()
```

## Usage

### Direct Script Execution
```bash
python scripts/segment.py --input_dir images/ --out_dir outputs/
python scripts/clean_figs.py --input_dir images/ --out_dir cleaned/
python scripts/extract_frames.py --input_dir videos/ --out_dir frames/
python scripts/split_dataset.py --raw_image_csv data.csv --mode ratio
```

### Installed Package
```bash
pip install -e .
insect-segment --input_dir images/ --out_dir outputs/
insect-clean --input_dir images/ --out_dir cleaned/
insect-extract --input_dir videos/ --out_dir frames/
insect-split --raw_image_csv data.csv --mode ratio
```

### Selective Installation
```bash
# Only segmentation (no video, no data dependencies)
pip install .[segmentation]

# Only cleaning
pip install .[cleaning]

# Only video extraction
pip install .[video]

# Only data splitting
pip install .[data]

# Development (all)
pip install .[dev]
```

## Dependencies

### Core (always required)
- numpy>=1.24.0
- Pillow>=10.0.0
- tqdm>=4.65.0

### Extras
- **segmentation**: torch, torchvision, opencv-python
- **cleaning**: imagehash
- **video**: opencv-python
- **data**: pandas

## Benefits

1. **Independent Maintenance**: Each script can be modified without affecting others
2. **Flexible Installation**: Users only install what they need
3. **Code Reuse**: Common utilities centralized in `src/common/`
4. **Domain Organization**: Related functionality grouped logically
5. **Easy Testing**: Each domain can be tested independently

## Migration Path

1. ✅ Create new directory structure
2. ✅ Extract common utilities to `src/common/`
3. ✅ Create domain modules (`src/segmentation/`, `src/framing/`, etc.)
4. ✅ Update scripts to use new imports
5. ✅ Update `requirements.txt` and `setup.py`
6. ✅ Document architecture
7. 🔄 Test each script independently
8. 🔄 Update README with new structure

## Future Enhancements

Possible future improvements:
- Add integration tests for pipeline chaining
- Add configuration files for complex workflows
- Add CLI command chaining (e.g., `insect-pipeline segment then clean`)
- Add Docker support with optional dependencies
