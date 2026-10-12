"""
NeuroGuide AI - Configuration & Constants
"""

import os
from pathlib import Path
from typing import Dict, List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
SAMPLES_DIR = PROJECT_ROOT / "backend" / "samples"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Classes & Mappings
CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)

# Human-readable labels and descriptions
CLASS_METADATA = {
    "glioma": {
        "display_name": "Glioma",
        "description": "Infiltrating primary parenchymal neuroepithelial brain tumor arising from glial cells.",
        "who_grade_typical": "Grade II-IV (e.g., Glioblastoma)",
        "clinical_urgency": "High",
    },
    "meningioma": {
        "display_name": "Meningioma",
        "description": "Typically slow-growing, extra-axial dural-based tumor arising from arachnoid cap cells.",
        "who_grade_typical": "Grade I (Benign) to Grade II/III (Atypical/Malignant)",
        "clinical_urgency": "Moderate to High",
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "description": "Sellar / parasellar adenoma/neuroendocrine lesion in the pituitary fossa.",
        "who_grade_typical": "Grade I (Usually benign adenoma)",
        "clinical_urgency": "Moderate",
    },
    "notumor": {
        "display_name": "No Tumor (Normal)",
        "description": "Healthy brain parenchyma with no observable intracranial mass lesions or neoplastic alterations.",
        "who_grade_typical": "N/A",
        "clinical_urgency": "Routine",
    },
}

# Image Preprocessing Constants
IMAGE_SIZE: int = 224
IMAGE_CHANNELS: int = 3
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# CLAHE Parameters
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

# Contour Skull-Stripping Parameters
BLUR_KERNEL_SIZE: tuple = (5, 5)
CONTOUR_MIN_AREA_RATIO: float = 0.05
CONTOUR_PADDING_PERCENT: float = 0.01

# Dataset Split Ratios
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
RANDOM_SEED: int = 42

# Training Hyperparameters defaults
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_WORKERS: int = 0  # 0 for Windows compatibility
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_EPOCHS: int = 25
