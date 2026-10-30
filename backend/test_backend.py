"""
test_backend.py  (Updated — Multi-View Ensemble)
Unit and integration tests for the Deepfake Audio Detection backend.

Run with:
    cd backend
    python -m pytest test_backend.py -v
"""

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _make_wav_bytes(freq: float = 220.0, duration: float = 3.0, sr: int = 16000) -> bytes:
    """Generate a pure-tone WAV file as raw bytes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _make_wav_array(freq: float = 220.0, duration: float = 3.0, sr: int = 16000):
    """Return a raw numpy float32 waveform array."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ─────────────────────────────────────────────────────
# Preprocessor tests
# ─────────────────────────────────────────────────────

class TestPreprocessor:
    def test_preprocess_returns_waveform(self):
        from preprocessor import preprocess_audio_bytes
        wav = _make_wav_bytes()
        y, sr = preprocess_audio_bytes(wav)
        assert y.ndim == 1
        assert sr == 16000
        assert len(y) > 0

    def test_normalize_peaks_at_one(self):
        from preprocessor import normalize
        raw = (np.random.randn(16000) * 5).astype(np.float32)
        normalized = normalize(raw)
        assert np.max(np.abs(normalized)) <= 1.0 + 1e-5

    def test_segment_fixed_length(self):
        from preprocessor import segment
        y = np.random.randn(50000).astype(np.float32)
        segs = segment(y, seg_len=48000)
        for s in segs:
            assert len(s) == 48000


# ─────────────────────────────────────────────────────
# Feature extractor tests (acoustic / SVM view)
# ─────────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_extract_acoustic_features_shape(self):
        from feature_extractor import extract_acoustic_features
        y = _make_wav_array()
        feat = extract_acoustic_features(y, 16000)
        assert feat.ndim == 1
        assert feat.shape[0] == 186, f"Expected 186 features, got {feat.shape[0]}"

    def test_extract_acoustic_features_finite(self):
        from feature_extractor import extract_acoustic_features
        y = _make_wav_array()
        feat = extract_acoustic_features(y, 16000)
        assert np.all(np.isfinite(feat)), "Some acoustic features are NaN or Inf"

    def test_extract_features_bytes_backward_compat(self):
        """extract_features(bytes) must remain compatible with main.py."""
        from feature_extractor import extract_features
        wav_bytes = _make_wav_bytes()
        feat = extract_features(wav_bytes)
        assert feat.shape[0] == 186

    def test_different_audio_gives_different_features(self):
        from feature_extractor import extract_acoustic_features
        f1 = extract_acoustic_features(_make_wav_array(freq=220), 16000)
        f2 = extract_acoustic_features(_make_wav_array(freq=880), 16000)
        assert not np.allclose(f1, f2)

    def test_feature_names_count(self):
        from feature_extractor import feature_names
        assert len(feature_names()) == 186


# ─────────────────────────────────────────────────────
# Spectrogram extractor tests (VGG16 CNN view)
# ─────────────────────────────────────────────────────

class TestSpectrogramExtractor:
    def test_returns_correct_shape(self):
        from spectrogram_extractor import extract_spectrogram
        y = _make_wav_array()
        img = extract_spectrogram(y, sr=16000)
        assert img.shape == (224, 224, 3), f"Expected (224, 224, 3), got {img.shape}"

    def test_values_in_range(self):
        from spectrogram_extractor import extract_spectrogram
        y = _make_wav_array()
        img = extract_spectrogram(y, sr=16000)
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_different_audio_different_spectrograms(self):
        from spectrogram_extractor import extract_spectrogram
        img1 = extract_spectrogram(_make_wav_array(freq=220), sr=16000)
        img2 = extract_spectrogram(_make_wav_array(freq=3000), sr=16000)
        assert not np.allclose(img1, img2)


# ─────────────────────────────────────────────────────
# Waveform extractor tests (1D CNN view)
# ─────────────────────────────────────────────────────

class TestWaveformExtractor:
    def test_returns_correct_length(self):
        from waveform_extractor import extract_waveform, WAVEFORM_LENGTH
        y = _make_wav_array()
        wv = extract_waveform(y, sr=16000)
        assert wv.shape == (WAVEFORM_LENGTH,), \
            f"Expected ({WAVEFORM_LENGTH},), got {wv.shape}"

    def test_short_audio_padded(self):
        from waveform_extractor import extract_waveform, WAVEFORM_LENGTH
        y = _make_wav_array(duration=1.0)   # 1 second — shorter than 3 s target
        wv = extract_waveform(y, sr=16000)
        assert wv.shape[0] == WAVEFORM_LENGTH

    def test_long_audio_trimmed(self):
        from waveform_extractor import extract_waveform, WAVEFORM_LENGTH
        y = _make_wav_array(duration=5.0)   # 5 seconds — longer than 3 s target
        wv = extract_waveform(y, sr=16000)
        assert wv.shape[0] == WAVEFORM_LENGTH

    def test_peak_normalized(self):
        from waveform_extractor import extract_waveform
        y = (_make_wav_array() * 10.0)      # over-amplified
        wv = extract_waveform(y, sr=16000)
        assert np.max(np.abs(wv)) <= 1.0 + 1e-5


# ─────────────────────────────────────────────────────
# Ensemble module tests
# ─────────────────────────────────────────────────────

class TestEnsemble:
    def test_meta_learner_output_range(self):
        """fuse() must return a score in [0, 1]."""
        import os
        ensemble_path = os.path.join(os.path.dirname(__file__), "model_ensemble.joblib")
        if not os.path.exists(ensemble_path):
            pytest.skip("model_ensemble.joblib not found — run train_ensemble.py first")
        from ensemble import load_ensemble, fuse
        load_ensemble()
        _, score, _ = fuse(p_svm=0.7, p_cnn=0.8, p_waveform=0.75)
        assert 0.0 <= score <= 1.0

    def test_learned_weights_present(self):
        """Verify ensemble meta-learner has learned coefficients."""
        import os, joblib
        ensemble_path = os.path.join(os.path.dirname(__file__), "model_ensemble.joblib")
        if not os.path.exists(ensemble_path):
            pytest.skip("model_ensemble.joblib not found — run train_ensemble.py first")
        meta = joblib.load(ensemble_path)
        coefs = meta.coef_ if hasattr(meta, "coef_") else None
        assert coefs is not None and len(coefs) == 3


# ─────────────────────────────────────────────────────
# SVM model tests (backward-compatible)
# ─────────────────────────────────────────────────────

class TestModel:
    def test_model_loads(self):
        import os
        for path in ["model_svm.joblib", "model.joblib"]:
            full = os.path.join(os.path.dirname(__file__), path)
            if os.path.exists(full):
                from model import load_model
                load_model()
                return
        pytest.skip("No SVM model found — run `python train.py` first")

    def test_predict_returns_valid_label(self):
        import os
        if not any(os.path.exists(os.path.join(os.path.dirname(__file__), p))
                   for p in ["model_svm.joblib", "model.joblib"]):
            pytest.skip("No SVM model found")
        from model import load_model, predict
        load_model()
        features = np.random.rand(186).astype(np.float32)
        label, confidence = predict(features)
        assert label in ("REAL", "FAKE")
        assert 0.0 <= confidence <= 1.0


# ─────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        import os
        if not any(os.path.exists(os.path.join(os.path.dirname(__file__), p))
                   for p in ["model_svm.joblib", "model.joblib"]):
            pytest.skip("No SVM model found — run `python train.py` first")
        from main import app
        return TestClient(app)

    def test_health_check(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models" in data   # new: models-ready dict

    def test_predict_wav_file(self, client):
        wav_bytes = _make_wav_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("REAL", "FAKE")
        assert 0.0 <= data["confidence"] <= 1.0
        assert "%" in data["confidence_pct"]
        assert "mode" in data                   # new: mode field
        assert "svm_score" in data              # new: individual scores

    def test_predict_returns_ensemble_scores(self, client):
        """Response must contain individual model scores."""
        wav_bytes = _make_wav_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        # SVM score always present; CNN/waveform may be None if not trained yet
        assert "svm_score" in data
        assert "cnn_score" in data
        assert "waveform_score" in data
        assert "ensemble_score" in data

    def test_predict_empty_file_rejected(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400

    def test_predict_invalid_format_rejected(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"fake content", "application/pdf")},
        )
        assert resp.status_code == 415
