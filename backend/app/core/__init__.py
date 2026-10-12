"""NeuroGuide AI - Core ML & Preprocessing Components"""

from backend.app.core.dataset import (
    BrainMRIAugmenter,
    BrainMRIDataset,
    create_dataloaders,
)
from backend.app.core.gradcam import (
    COLORMAP_REGISTRY,
    GradCAM,
    determine_anatomical_location,
    explain_scan,
    extract_roi,
)
from backend.app.core.model import (
    NeuroBrainClassifier,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from backend.app.core.ood_detector import OODDetector
from backend.app.core.preprocessor import BrainMRIPreprocessor
from backend.app.core.uncertainty import (
    CLINICAL_LIMITATIONS_CATALOG,
    UncertaintyEvaluator,
    calculate_probability_margin,
    calculate_shannon_entropy,
)

__all__ = [
    "BrainMRIPreprocessor",
    "BrainMRIAugmenter",
    "BrainMRIDataset",
    "create_dataloaders",
    "NeuroBrainClassifier",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
    "GradCAM",
    "COLORMAP_REGISTRY",
    "extract_roi",
    "determine_anatomical_location",
    "explain_scan",
    "calculate_shannon_entropy",
    "calculate_probability_margin",
    "UncertaintyEvaluator",
    "CLINICAL_LIMITATIONS_CATALOG",
    "OODDetector",
]

