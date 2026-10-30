"""
model.py  (Updated — Multi-View)
Loads all three trained models and the ensemble meta-learner at startup,
then provides a unified predict_all() function used by main.py.

Models:
  model_svm.joblib       — Acoustic SVM (trained by train.py)
  model_cnn.keras        — Mel spectrogram VGG16 (trained by train_cnn.py)
  model_waveform.keras   — Raw waveform 1D CNN (trained by train_waveform.py)
  model_ensemble.joblib  — Linear Regression meta-learner (trained by train_ensemble.py)
"""

import os
import numpy as np
import joblib
from typing import Tuple, Dict

BASE_DIR = os.path.dirname(__file__)

SVM_PATH      = os.path.join(BASE_DIR, "model_svm.joblib")
COMPAT_PATH   = os.path.join(BASE_DIR, "model.joblib")    # backward compat alias
CNN_PATH      = os.path.join(BASE_DIR, "model_cnn.keras")
WAVE_PATH     = os.path.join(BASE_DIR, "model_waveform.keras")
ENSEMBLE_PATH = os.path.join(BASE_DIR, "model_ensemble.joblib")

_svm_model      = None
_cnn_model      = None
_wave_model     = None
_ensemble_model = None


def load_model():
    """
    Load all four models from disk.
    Called once at FastAPI startup.
    Falls back gracefully if CNN/Waveform models are not yet trained.
    """
    global _svm_model, _cnn_model, _wave_model, _ensemble_model

    # ── SVM (required) ────────────────────────────────────────
    svm_path = SVM_PATH if os.path.exists(SVM_PATH) else COMPAT_PATH
    if not os.path.exists(svm_path):
        raise FileNotFoundError(
            f"SVM model not found at {SVM_PATH}. Run `python train.py` first."
        )
    _svm_model = joblib.load(svm_path)
    print(f"✅ SVM model loaded from {svm_path}")

    # ── VGG16 CNN (optional — warn if missing) ────────────────
    if os.path.exists(CNN_PATH):
        import tensorflow as tf
        _cnn_model = tf.keras.models.load_model(CNN_PATH)
        print(f"✅ CNN model loaded from {CNN_PATH}")
    else:
        print(f"⚠️  CNN model not found at {CNN_PATH} — run train_cnn.py to enable")

    # ── 1D Waveform CNN (optional) ────────────────────────────
    if os.path.exists(WAVE_PATH):
        import tensorflow as tf
        _wave_model = tf.keras.models.load_model(WAVE_PATH)
        print(f"✅ Waveform CNN model loaded from {WAVE_PATH}")
    else:
        print(f"⚠️  Waveform CNN not found at {WAVE_PATH} — run train_waveform.py to enable")

    # ── Ensemble meta-learner (optional) ─────────────────────
    if os.path.exists(ENSEMBLE_PATH):
        _ensemble_model = joblib.load(ENSEMBLE_PATH)
        print(f"✅ Ensemble meta-learner loaded from {ENSEMBLE_PATH}")
    else:
        print(f"⚠️  Ensemble model not found at {ENSEMBLE_PATH} — run train_ensemble.py to enable")


def predict(features: np.ndarray) -> Tuple[str, float]:
    """
    Backward-compatible single-model prediction (SVM only).
    Used by tests that pre-date the multi-view API.
    """
    if _svm_model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    features_2d = features.reshape(1, -1)
    proba   = _svm_model.predict_proba(features_2d)[0]
    pred_idx = int(np.argmax(proba))
    label    = "FAKE" if pred_idx == 1 else "REAL"
    return label, float(proba[pred_idx])


def predict_all(y: np.ndarray, sr: int) -> Dict:
    """
    Run all available models on a pre-loaded waveform, then fuse via ensemble.

    Args:
        y:  Float32 mono waveform (any length — extractors handle padding/trimming)
        sr: Sample rate

    Returns:
        dict with keys:
            prediction      — final label 'REAL' or 'FAKE'
            ensemble_score  — float [0,1]; higher = more likely FAKE
            svm_score       — SVM FAKE probability
            cnn_score       — VGG16 FAKE probability (or None)
            waveform_score  — 1D CNN FAKE probability (or None)
            mode            — 'full_ensemble' | 'partial' | 'svm_only'
    """
    from feature_extractor import extract_acoustic_features
    from spectrogram_extractor import extract_spectrogram
    from waveform_extractor import extract_waveform

    if _svm_model is None:
        raise RuntimeError("Models not loaded. Call load_model() first.")

    # ── SVM ───────────────────────────────────────────────────
    feat   = extract_acoustic_features(y, sr)
    p_svm  = float(_svm_model.predict_proba(feat.reshape(1, -1))[0][1])

    # ── VGG16 CNN ─────────────────────────────────────────────
    p_cnn = None
    if _cnn_model is not None:
        spec  = extract_spectrogram(y, sr)
        p_cnn = float(_cnn_model.predict(spec[np.newaxis], verbose=0)[0][0])

    # ── 1D Waveform CNN ───────────────────────────────────────
    p_wave = None
    if _wave_model is not None:
        wv     = extract_waveform(y, sr).reshape(1, -1, 1)
        p_wave = float(_wave_model.predict(wv, verbose=0)[0][0])

    # ── Ensemble fusion ───────────────────────────────────────
    if _ensemble_model is not None and p_cnn is not None and p_wave is not None:
        # Full meta-learner (Linear or Logistic Regression)
        X_meta = np.array([[p_cnn, p_wave, p_svm]], dtype=np.float32)
        raw_score = float(_ensemble_model.predict(X_meta)[0])
        score = float(np.clip(raw_score, 0.0, 1.0))
        mode  = "full_ensemble"
    elif p_cnn is not None and p_wave is not None:
        # Fallback: plain average of three (if ensemble model not yet trained)
        score = (0.4 * p_cnn + 0.3 * p_wave + 0.3 * p_svm)
        mode  = "partial"
    else:
        # SVM only
        score = p_svm
        mode  = "svm_only"

    # Use a slightly lower threshold because classifiers trained on
    # imbalanced data tend to under-estimate the fake probability.
    FAKE_THRESHOLD = 0.40
    label = "FAKE" if score > FAKE_THRESHOLD else "REAL"
    return {
        "prediction":     label,
        "ensemble_score": round(score, 4),
        "svm_score":      round(p_svm,  4),
        "cnn_score":      round(p_cnn,  4) if p_cnn  is not None else None,
        "waveform_score": round(p_wave, 4) if p_wave is not None else None,
        "mode":           mode,
    }
