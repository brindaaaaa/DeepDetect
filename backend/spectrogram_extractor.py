"""
spectrogram_extractor.py
Converts a raw waveform into a 224×224×3 Mel spectrogram image
suitable for VGG16 input.

Pipeline:
    waveform (float32, mono) → Mel spectrogram → dB scale → normalize [0,1]
    → resize to 224×224 → stack as 3-channel RGB image
"""

import numpy as np
import librosa
from PIL import Image


# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
N_MELS: int   = 128      # Number of Mel filter banks
N_FFT: int    = 2048     # FFT window size
HOP_LEN: int  = 512      # Hop length for STFT
F_MAX: int    = 8_000    # Maximum frequency (Hz) — covers speech range
IMG_SIZE: int = 224      # VGG16 input resolution


def extract_spectrogram(y: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """
    Convert a mono waveform to a 224×224×3 Mel spectrogram image.

    Args:
        y:  Float32 waveform array (mono, any length)
        sr: Sample rate

    Returns:
        np.ndarray of shape (224, 224, 3), dtype float32, values in [0, 1]
    """
    # 1. Compute Mel spectrogram power
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
        fmax=F_MAX,
    )

    # 2. Convert to decibels
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 3. Normalize to [0, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 1e-6:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_db)

    # 4. Resize to IMG_SIZE × IMG_SIZE using PIL
    #    PIL expects shape (H, W), so mel_norm is already (n_mels, time)
    img_gray = Image.fromarray((mel_norm * 255).astype(np.uint8), mode="L")
    img_resized = img_gray.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # 5. Stack as 3-channel image for VGG16
    img_arr = np.array(img_resized).astype(np.float32) / 255.0  # (224, 224)
    img_rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)    # (224, 224, 3)

    return img_rgb


def extract_spectrogram_from_bytes(audio_bytes: bytes, sr: int = 16_000) -> np.ndarray:
    """
    Convenience wrapper: load audio bytes then extract spectrogram.

    Args:
        audio_bytes: Raw audio file bytes (wav/mp3/flac/etc.)
        sr: Target sample rate

    Returns:
        np.ndarray of shape (224, 224, 3), dtype float32, values in [0, 1]
    """
    import io
    buf = io.BytesIO(audio_bytes)
    y, _ = librosa.load(buf, sr=sr, mono=True)
    y = y.astype(np.float32)
    return extract_spectrogram(y, sr)
