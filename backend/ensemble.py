"""
ensemble.py
Inference wrapper for the Linear Regression meta-learner ensemble.

After running train_ensemble.py, this module loads model_ensemble.joblib
and provides a single fuse() function used by the FastAPI endpoint.
"""

import os
import numpy as np
import joblib
from typing import Tuple, Dict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_ensemble.joblib")

_meta_model = None
THRESHOLD: float = 0.5


def load_ensemble():
    """Load the ensemble meta-learner from disk."""
    global _meta_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Ensemble model not found at {MODEL_PATH}. "
            "Please run `python train_ensemble.py` first."
        )
    _meta_model = joblib.load(MODEL_PATH)
    print(f"✅ Ensemble meta-learner loaded from {MODEL_PATH}")


def fuse(p_svm: float, p_cnn: float, p_waveform: float
         ) -> Tuple[str, float, Dict[str, float]]:
    """
    Combine the three model probabilities using the learned meta-learner.

    Args:
        p_svm:      SVM probability of FAKE (0–1)
        p_cnn:      VGG16 CNN probability of FAKE (0–1)
        p_waveform: 1D CNN probability of FAKE (0–1)

    Returns:
        (label, ensemble_score, individual_scores)
        label:           'REAL' or 'FAKE'
        ensemble_score:  float in [0, 1] — higher means more likely FAKE
        individual_scores: dict with keys 'svm', 'cnn', 'waveform'
    """
    if _meta_model is None:
        raise RuntimeError("Ensemble model not loaded. Call load_ensemble() first.")

    X = np.array([[p_cnn, p_waveform, p_svm]], dtype=np.float32)

    # LinearRegression → raw float; clip to [0, 1] to get probability-like score
    raw = float(_meta_model.predict(X)[0])
    score = float(np.clip(raw, 0.0, 1.0))

    label = "FAKE" if score > THRESHOLD else "REAL"

    return label, score, {
        "svm": round(p_svm, 4),
        "cnn": round(p_cnn, 4),
        "waveform": round(p_waveform, 4),
    }
