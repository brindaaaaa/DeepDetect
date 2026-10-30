"""
main.py  (Updated — Multi-View Ensemble API)
FastAPI application for deepfake audio detection using the multi-view ensemble pipeline.

Endpoints:
    GET  /           → health check + model status
    POST /predict    → accepts audio file, returns multi-view ensemble prediction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
import traceback
import os

from model import load_model, predict_all

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# ────────────────────────────────────────────
# App setup
# ────────────────────────────────────────────

app = FastAPI(
    title="Deepfake Audio Detection API — Multi-View Ensemble",
    description=(
        "Upload an audio file and get a REAL / FAKE prediction from three "
        "specialized models (SVM, VGG16-CNN, 1D-CNN) fused via Linear Regression."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load all models when the server starts."""
    load_model()


# ────────────────────────────────────────────
# Response schemas
# ────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction:     str            # "REAL" or "FAKE"
    confidence:     float          # ensemble score 0.0–1.0
    confidence_pct: str            # e.g. "87.4%"
    message:        str            # human-readable summary

    # Individual model scores (None if model not yet trained)
    svm_score:      Optional[float] = None
    cnn_score:      Optional[float] = None
    waveform_score: Optional[float] = None
    ensemble_score: Optional[float] = None

    # Metadata
    filename:       Optional[str]   = None
    feature_count:  int             = 186
    mode:           str             = "svm_only"   # full_ensemble | partial | svm_only


# ────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────

@app.get("/health")
async def health_check():
    base = os.path.dirname(__file__)
    models_ready = {
        "svm":      os.path.exists(os.path.join(base, "model_svm.joblib"))
                    or os.path.exists(os.path.join(base, "model.joblib")),
        "cnn":      os.path.exists(os.path.join(base, "model_cnn.keras")),
        "waveform": os.path.exists(os.path.join(base, "model_waveform.keras")),
        "ensemble": os.path.exists(os.path.join(base, "model_ensemble.joblib")),
    }
    return {
        "status":  "ok",
        "service": "Multi-View Deepfake Audio Detection API",
        "version": "2.0.0",
        "models":  models_ready,
    }


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html at the root URL."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


# Mount static files AFTER route definitions so /predict isn't shadowed
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Accept an audio file and return a multi-view ensemble deepfake prediction.

    Supported formats: .wav, .mp3, .flac, .ogg, .m4a
    The response includes individual scores from each model plus the fused ensemble score.
    """
    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/mpeg",
        "audio/mp3", "audio/flac", "audio/ogg",
        "audio/m4a", "audio/mp4", "application/octet-stream",
    }
    content_type = file.content_type or ""
    filename     = file.filename or "unknown"
    ext          = filename.lower().split(".")[-1]
    allowed_exts = {"wav", "mp3", "flac", "ogg", "m4a", "aac"}

    if ext not in allowed_exts and content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format: {ext}. Use wav, mp3, flac, ogg, or m4a.",
        )

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # ── Preprocess: load, mono, 16 kHz, normalize ────────
        from preprocessor import preprocess_audio_bytes
        y, sr = preprocess_audio_bytes(audio_bytes)

        # ── Run all models + ensemble fusion ─────────────────
        result = predict_all(y, sr)

        label      = result["prediction"]
        fake_score = result["ensemble_score"]   # 0→REAL, 1→FAKE
        mode       = result["mode"]

        # Confidence = certainty in the stated label (not raw fake probability)
        # FAKE: confidence = fake_score  (e.g. 0.95 → "95% sure it's FAKE")
        # REAL: confidence = 1 - fake_score (e.g. 0.05 → "95% sure it's REAL")
        confidence = fake_score if label == "FAKE" else (1.0 - fake_score)
        pct        = f"{confidence * 100:.1f}%"

        emoji   = "🟢" if label == "REAL" else "🔴"
        model_str = {
            "full_ensemble": "ensemble (SVM + VGG16 + WaveCNN)",
            "partial":       "partial ensemble (weights averaged)",
            "svm_only":      "SVM (acoustic features)",
        }.get(mode, mode)

        message = (
            f"{emoji} {label} speech detected via {model_str} "
            f"(confidence: {pct}, fake score: {fake_score * 100:.1f}%)"
        )

        return PredictionResponse(
            prediction     = label,
            confidence     = round(confidence, 4),
            confidence_pct = pct,
            message        = message,
            svm_score      = result.get("svm_score"),
            cnn_score      = result.get("cnn_score"),
            waveform_score = result.get("waveform_score"),
            ensemble_score = fake_score,
            filename       = filename,
            feature_count  = 186,
            mode           = mode,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}",
        )
