"""
waveform_extractor.py
Prepares raw waveform tensors for the 1D CNN model.

Target output: fixed-length float32 array of WAVEFORM_LENGTH samples.
  • Clips that are too long → trimmed to WAVEFORM_LENGTH
  • Clips that are too short → zero-padded on the right
"""

import numpy as np
import librosa
import io


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SR_TARGET: int         = 16_000   # Standard sample rate
SEGMENT_DURATION: float = 3.0     # Seconds
WAVEFORM_LENGTH: int   = int(SR_TARGET * SEGMENT_DURATION)   # 48 000


def extract_waveform(y: np.ndarray, sr: int = SR_TARGET) -> np.ndarray:
    """
    Resample (if needed), normalize, and pad/trim a waveform to WAVEFORM_LENGTH.

    Args:
        y:  Input waveform (mono float32, any sample rate)
        sr: Sample rate of y

    Returns:
        np.ndarray of shape (WAVEFORM_LENGTH,), dtype float32
    """
    # Resample if needed
    if sr != SR_TARGET:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)

    y = y.astype(np.float32)

    # Peak-normalize
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak

    # Pad or trim to exact length
    if len(y) >= WAVEFORM_LENGTH:
        y = y[:WAVEFORM_LENGTH]
    else:
        y = np.pad(y, (0, WAVEFORM_LENGTH - len(y)))

    return y.astype(np.float32)


def extract_waveform_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Convenience wrapper: load audio bytes then extract waveform tensor.

    Args:
        audio_bytes: Raw audio file bytes (wav/mp3/flac/etc.)

    Returns:
        np.ndarray of shape (WAVEFORM_LENGTH,), dtype float32
    """
    buf = io.BytesIO(audio_bytes)
    y, sr = librosa.load(buf, sr=SR_TARGET, mono=True)
    return extract_waveform(y, sr)
