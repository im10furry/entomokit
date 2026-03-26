"""Augmentation preset definitions."""

PRESETS: dict = {
    "light": [
        {"name": "HorizontalFlip", "p": 0.5},
        {
            "name": "RandomBrightnessContrast",
            "p": 0.3,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
        },
    ],
    "medium": [
        {"name": "HorizontalFlip", "p": 0.5},
        {"name": "VerticalFlip", "p": 0.3},
        {
            "name": "RandomBrightnessContrast",
            "p": 0.4,
            "brightness_limit": 0.3,
            "contrast_limit": 0.3,
        },
        {
            "name": "Affine",
            "p": 0.4,
            "translate_percent": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            "scale": (0.9, 1.1),
            "rotate": (-15, 15),
        },
        {"name": "GaussNoise", "p": 0.2, "std_range": (0.02, 0.08)},
    ],
    "heavy": [
        {"name": "HorizontalFlip", "p": 0.5},
        {"name": "VerticalFlip", "p": 0.3},
        {
            "name": "RandomBrightnessContrast",
            "p": 0.5,
            "brightness_limit": 0.4,
            "contrast_limit": 0.4,
        },
        {
            "name": "Affine",
            "p": 0.5,
            "translate_percent": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            "scale": (0.85, 1.15),
            "rotate": (-25, 25),
        },
        {"name": "GaussNoise", "p": 0.3, "std_range": (0.02, 0.12)},
        {"name": "ElasticTransform", "p": 0.2, "alpha": 50, "sigma": 5},
        {
            "name": "CoarseDropout",
            "p": 0.2,
            "num_holes_range": (1, 8),
            "hole_height_range": (0.02, 0.08),
            "hole_width_range": (0.02, 0.08),
        },
    ],
    "conservative": [
        {"name": "HorizontalFlip", "p": 0.5},
        {
            "name": "RandomBrightnessContrast",
            "p": 0.2,
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
        },
    ],
}

PRESETS["safe-for-small-dataset"] = PRESETS["conservative"]
