"""
NeuroGuide AI - Model Uncertainty & Clinical Limitations Framework

Implements Phase 4 of the NeuroGuide AI architecture:
1. Shannon Prediction Entropy (H in bits) calculation.
2. Top-1 vs Top-2 Probability Margin (Delta) calculation.
3. Confidence Evaluator categorizing scans into:
   - HIGH CONFIDENCE (H < 0.6 and Delta > 0.4)
   - MODERATE CONFIDENCE (0.6 <= H <= 1.2)
   - AMBIGUOUS / BORDERLINE WARNING (H > 1.2 or Delta < 0.15)
4. Automated clinical caveat generation for overlapping diagnoses.
5. Comprehensive clinical & technical limitations catalog.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.app.config import CLASS_METADATA, NUM_CLASSES


# Clinical Limitations Catalog as defined in Section 4.3
CLINICAL_LIMITATIONS_CATALOG = [
    {
        "id": "single_slice_bias",
        "title": "2D Single-Slice Bias",
        "severity": "Warning",
        "description": (
            "Clinical MRI diagnosis requires reviewing contiguous 3D volumetric slice stacks "
            "across multiple anatomical orientations (Axial, Coronal, and Sagittal). Analyzing an "
            "isolated 2D slice risks missing tumor volume margins, dural attachments, or subtle satellites."
        ),
    },
    {
        "id": "missing_pulse_sequences",
        "title": "Pulse Sequence Context Missing",
        "severity": "Warning",
        "description": (
            "Radiologists evaluate contrast behavior across multimodal sequences (T1-weighted, "
            "T2-weighted, FLAIR, and T1+Gadolinium contrast). This model processes standalone "
            "scans without sequence metadata, which can mimic non-neoplastic pathologies."
        ),
    },
    {
        "id": "absence_of_histopathology",
        "title": "Absence of Molecular & Histopathological Markers",
        "severity": "Critical",
        "description": (
            "Definitive neuro-oncology diagnosis and WHO CNS grading require tissue biopsy and "
            "molecular profiling (e.g., IDH1/IDH2 mutation status, 1p/19q co-deletion, MGMT promoter "
            "methylation, and ATRX loss), which cannot be deduced from macroscopic imaging alone."
        ),
    },
    {
        "id": "scanner_motion_artifacts",
        "title": "Susceptibility to Scanner & Motion Artifacts",
        "severity": "Notice",
        "description": (
            "Patient micro-movements, metallic dental hardware, or gradient field inhomogeneities "
            "can induce ringing or signal dropouts that trigger false-positive Grad-CAM activations."
        ),
    },
    {
        "id": "non_diagnostic_research_use",
        "title": "Non-Diagnostic Educational Prototype",
        "severity": "Mandatory Disclaimer",
        "description": (
            "NeuroGuide AI is strictly intended for educational demonstration and medical explainability "
            "research. It is NOT a medical device, is not CE/FDA approved, and must NEVER replace the "
            "formal interpretation of a board-certified neuroradiologist or neurosurgeon."
        ),
    },
]


def calculate_shannon_entropy(
    probabilities: Union[Dict[str, float], List[float], np.ndarray],
    eps: float = 1e-12,
) -> float:
    """
    Computes Shannon prediction entropy H(p) in bits:
        H(p) = - sum_{i=1}^N p_i * log2(p_i)

    For 4 classes:
    - Minimum entropy = 0.0 (absolute certainty in 1 class).
    - Maximum entropy = log2(4) = 2.0 (uniform distribution, 25% each).

    Args:
        probabilities: Vector or dictionary of class probabilities summing to 1.
        eps: Small epsilon to prevent log2(0).

    Returns:
        float entropy value in [0.0, 2.0].
    """
    if isinstance(probabilities, dict):
        probs = np.array(list(probabilities.values()), dtype=np.float64)
    else:
        probs = np.array(probabilities, dtype=np.float64)

    probs = np.clip(probs, eps, 1.0)
    # Re-normalize just in case
    probs = probs / np.sum(probs)

    entropy = -np.sum(probs * np.log2(probs))
    return float(np.clip(round(entropy, 4), 0.0, math.log2(NUM_CLASSES)))


def calculate_probability_margin(
    probabilities: Union[Dict[str, float], List[float], np.ndarray],
) -> Tuple[float, Tuple[str, float], Tuple[str, float]]:
    """
    Computes top-1 vs top-2 probability margin Delta:
        Delta = p_{rank 1} - p_{rank 2}

    Args:
        probabilities: Dict or array of probabilities.

    Returns:
        delta: Margin difference in [0.0, 1.0].
        top1_entry: (class_name, prob)
        top2_entry: (class_name, prob)
    """
    if isinstance(probabilities, dict):
        sorted_pairs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    else:
        probs = list(probabilities)
        sorted_pairs = sorted(
            [(f"class_{i}", float(p)) for i, p in enumerate(probs)],
            key=lambda x: x[1],
            reverse=True,
        )

    if len(sorted_pairs) < 2:
        return 1.0, sorted_pairs[0], ("none", 0.0)

    top1 = sorted_pairs[0]
    top2 = sorted_pairs[1]
    delta = float(top1[1] - top2[1])

    return round(delta, 4), top1, top2


class UncertaintyEvaluator:
    """
    Evaluator quantifying model predictive uncertainty, diagnostic stability,
    and generating targeted clinical caveats for physicians and students.
    """

    # Thresholds specified in Section 4.1
    ENTROPY_HIGH_THRESHOLD = 0.60
    ENTROPY_AMBIGUOUS_THRESHOLD = 1.20

    MARGIN_HIGH_THRESHOLD = 0.40
    MARGIN_AMBIGUOUS_THRESHOLD = 0.15

    def evaluate(
        self,
        probabilities: Union[Dict[str, float], List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluates uncertainty metrics for a predicted probability distribution.

        Returns structured report:
        - entropy: float (0.0 to 2.0 bits)
        - normalized_entropy: float (0.0 to 1.0)
        - margin_delta: float (0.0 to 1.0)
        - confidence_level: 'HIGH' | 'MODERATE' | 'AMBIGUOUS'
        - confidence_label: Human-readable badge text
        - is_ambiguous: bool
        - top_class: str
        - runner_up_class: str
        - clinical_caveat: Optional clinical advisory text
        - limitations: List of relevant clinical limitations
        """
        entropy = calculate_shannon_entropy(probabilities)
        max_possible_entropy = math.log2(NUM_CLASSES)
        normalized_entropy = round(entropy / max_possible_entropy, 4)

        margin_delta, top1, top2 = calculate_probability_margin(probabilities)

        # Confidence categorization logic
        if entropy < self.ENTROPY_HIGH_THRESHOLD and margin_delta > self.MARGIN_HIGH_THRESHOLD:
            confidence_level = "HIGH"
            confidence_label = "High Diagnostic Confidence"
            confidence_color = "#10B981"  # Emerald green
            is_ambiguous = False
        elif entropy > self.ENTROPY_AMBIGUOUS_THRESHOLD or margin_delta < self.MARGIN_AMBIGUOUS_THRESHOLD:
            confidence_level = "AMBIGUOUS"
            confidence_label = "Borderline / Ambiguous Scan"
            confidence_color = "#EF4444"  # Red
            is_ambiguous = True
        else:
            confidence_level = "MODERATE"
            confidence_label = "Moderate Confidence (Clinical Review Advised)"
            confidence_color = "#F59E0B"  # Amber
            is_ambiguous = False

        # Generate targeted clinical caveat if margin is low or entropy is moderate/high
        top1_name = CLASS_METADATA.get(top1[0], {}).get("display_name", top1[0].capitalize())
        top2_name = CLASS_METADATA.get(top2[0], {}).get("display_name", top2[0].capitalize())

        caveats: List[str] = []
        if is_ambiguous:
            caveats.append(
                f"The AI model detected overlapping radiological features between {top1_name} "
                f"({round(top1[1] * 100, 1)}%) and {top2_name} ({round(top2[1] * 100, 1)}%). "
                f"Prediction margin is narrow (Δ = {round(margin_delta * 100, 1)}%). "
                f"Comprehensive multi-sequence review is strongly indicated."
            )
        elif confidence_level == "MODERATE":
            caveats.append(
                f"Secondary differential feature observed: {top2_name} holds "
                f"{round(top2[1] * 100, 1)}% residual probability. Correlate with clinical history."
            )

        clinical_caveat_text = " ".join(caveats) if caveats else None

        return {
            "entropy": entropy,
            "max_entropy": round(max_possible_entropy, 4),
            "normalized_entropy": normalized_entropy,
            "margin_delta": margin_delta,
            "confidence_level": confidence_level,
            "confidence_label": confidence_label,
            "confidence_color": confidence_color,
            "is_ambiguous": is_ambiguous,
            "top_prediction": {
                "class": top1[0],
                "display_name": top1_name,
                "probability": round(top1[1], 4),
            },
            "runner_up_prediction": {
                "class": top2[0],
                "display_name": top2_name,
                "probability": round(top2[1], 4),
            },
            "clinical_caveat": clinical_caveat_text,
            "clinical_limitations": CLINICAL_LIMITATIONS_CATALOG,
        }
