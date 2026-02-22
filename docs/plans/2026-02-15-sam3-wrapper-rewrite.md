# SAM3 Wrapper Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite SAM3 wrapper to use ISAT's SAM3 implementation (which is Facebook's official SAM3) with CPU support and local checkpoint loading.

**Architecture:** 
- Replace old `sam3.build_sam3` import with ISAT's `ISAT.segment_any.sam3.model_builder`
- Use `build_sam3_image_model` to load checkpoint
- Use `Sam3Predictor` wrapper for inference
- Keep existing API compatible with current code

**Tech Stack:** Python, PyTorch, ISAT (contains Facebook's official SAM3)

**Key Files:**
- `/Users/zf/tools/miniconda3/envs/isat/lib/python3.10/site-packages/ISAT/segment_any/sam3/model_builder.py` - Model builder
- `/Users/zf/tools/miniconda3/envs/isat/lib/python3.10/site-packages/ISAT/segment_any/sam3/build_sam.py` - Predictor wrapper

**Steps:**

### Task 1: Update sam3_wrapper.py imports and initialization

**Files:**
- Modify: `src/sam3_wrapper.py`

**Step 1.1: Update imports**

Replace existing imports with ISAT SAM3:
```python
from ISAT.segment_any.sam3.model_builder import build_sam3_image_model
from ISAT.segment_any.sam3.build_sam import Sam3Predictor
```

**Step 1.2: Update _load_model() method**

```python
def _load_model(self) -> None:
    """Load SAM3 model from checkpoint."""
    if not self.checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {self.checkpoint_path}"
        )
    
    logger.info(f"Loading SAM3 model from {self.checkpoint_path}")
    logger.info(f"Using device: {self.device}")
    
    import os
    from ISAT.segment_any.sam3.model_builder import build_sam3_image_model
    from ISAT.segment_any.sam3.build_sam import Sam3Predictor
    
    # Find BPE tokenizer file from ISAT installation
    import ISAT.segment_any.sam3 as sam3_module
    isat_dir = os.path.dirname(sam3_module.__file__)
    bpe_path = os.path.join(isat_dir, "bpe_simple_vocab_16e6.txt.gz")
    
    # Build model from ISAT (which contains Facebook's official SAM3)
    # Enable inst_interactive_predictor for SAM3 to work properly
    self.model = build_sam3_image_model(
        checkpoint_path=str(self.checkpoint_path),
        bpe_path=bpe_path,
        load_from_HF=False,
        enable_inst_interactivity=True,  # Enable SAM1 interactive mode for SAM3
        device=self.device,
        eval_mode=True,
    )
    
    # Wrap with predictor
    self.predictor = Sam3Predictor(self.model)
    
    logger.info("SAM3 model loaded successfully")
```

**Step 1.3: Update predict() method for text prompt**

```python
def predict(
    self,
    image: np.ndarray,
    text_prompt: Optional[str] = None,
    # ... other params
) -> List[np.ndarray]:
    self.predictor.set_image(image)
    
    if text_prompt is not None:
        # Use text prompt prediction
        from PIL import Image
        pil_image = Image.fromarray(image)
        masks, scores = self.predictor.predict_with_text_prompt(pil_image, text_prompt)
    else:
        masks, scores, logits = self.predictor.predict(
            multimask_output=multimask_output,
            **prompt_kwargs
        )
    
    mask_list = [masks[i] for i in range(len(masks))]
    return mask_list
```

**Step 2: Run tests**

```bash
cd /Users/zf/data/coding/segmentation_synthesize
/Users/zf/tools/miniconda3/envs/isat/bin/python -m pytest tests/ -v
```

Expected: All 48 tests pass.

**Step 3: Test with real image**

```bash
cd /Users/zf/data/coding/segmentation_synthesize
/Users/zf/tools/miniconda3/envs/isat/bin/python -c "
from src.sam3_wrapper import SAM3Wrapper
import numpy as np

# Load model
wrapper = SAM3Wrapper('models/sam3.pt', device='cpu')
print(f'Model loaded on: {wrapper.device}')

# Create test image
img = np.ones((1008, 1008, 3), dtype=np.uint8) * 255

# Set image
wrapper.predictor.set_image(img)
print('Image encoded successfully')

# Test prediction with text prompt
from PIL import Image
masks, scores = wrapper.predictor.predict_with_text_prompt(Image.fromarray(img), 'insect')
print(f'Found {len(masks)} masks')
"
```

Expected: Should load model and segment image successfully.

---