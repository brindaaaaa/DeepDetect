"""
feature_extractor.py
Extracts 186-dim acoustic feature vectors for deepfake detection.
Features: MFCCs (40+delta), Chroma, Spectral Contrast, Centroid/Bandwidth/Rolloff, ZCR, RMS, Mel stats

Public API:
  extract_acoustic_features(y, sr)       -- from waveform array (used by train*.py & ensemble)
  extract_features(audio_bytes)          -- from raw bytes (backward-compatible, used by main.py)
"""

import numpy as np
import librosa
import io


def extract_acoustic_features(y: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """
    Compute a 186-dim acoustic feature vector from a pre-loaded waveform array.
    Used by train.py (SVM training) and train_ensemble.py.

    Args:
        y:  Float32 mono waveform
        sr: Sample rate

    Returns:
        np.ndarray of shape (186,), dtype float32
    """
    y = y.astype(np.float32)

    # Trim silence — ensure at least 2048 samples so delta(width=9) always works
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    # --- 1. MFCCs (40 mean + 40 std + 40 delta mean) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std  = np.std(mfccs,  axis=1)
    # Adaptive delta width: largest odd number <= min(9, n_frames)
    _n_frames = mfccs.shape[1]
    _delta_w  = min(9, _n_frames) if min(9, _n_frames) % 2 == 1 else min(9, _n_frames) - 1
    _delta_w  = max(_delta_w, 1)
    mfcc_delta = librosa.feature.delta(mfccs, width=_delta_w)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    # --- 2. Chroma (12) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # --- 3. Spectral Contrast (7) ---
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # --- 4. Spectral Centroid / Bandwidth / Rolloff (3) ---
    centroid_mean  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth_mean = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff_mean   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # --- 5. Zero-Crossing Rate (2) ---
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean, zcr_std = float(np.mean(zcr)), float(np.std(zcr))

    # --- 6. RMS Energy (2) ---
    rms = librosa.feature.rms(y=y)
    rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))

    # --- 7. Mel spectrogram statistics (20+20) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)[:20]
    mel_std  = np.std(mel_db,  axis=1)[:20]

    features = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_delta_mean,      # 120
        chroma_mean, contrast_mean,                 # 19
        [centroid_mean, bandwidth_mean, rolloff_mean],  # 3
        [zcr_mean, zcr_std],                        # 2
        [rms_mean, rms_std],                        # 2
        mel_mean, mel_std,                          # 40
    ])  # Total: 186
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_features(audio_bytes: bytes, sr_target: int = 16_000) -> np.ndarray:
    """
    Backward-compatible entry-point used by main.py (FastAPI).
    Loads audio from raw bytes and returns a 186-dim feature vector.
    Works with .wav, .mp3, .flac, .ogg etc.
    """
    # Load from bytes
    audio_buf = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buf, sr=sr_target, mono=True)

    # Trim silence from edges (important for short TTS clips)
    y, _ = librosa.effects.trim(y, top_db=20)

    # --- 1. MFCCs (40 coefficients) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)       # shape (40,)
    mfcc_std  = np.std(mfccs,  axis=1)       # shape (40,)

    # --- 2. Delta MFCCs ---
    _n_frames2 = mfccs.shape[1]
    _delta_w2  = min(9, _n_frames2) if min(9, _n_frames2) % 2 == 1 else min(9, _n_frames2) - 1
    _delta_w2  = max(_delta_w2, 1)
    mfcc_delta = librosa.feature.delta(mfccs, width=_delta_w2)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)   # shape (40,)

    # --- 3. Chroma STFT ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)    # shape (12,)

    # --- 4. Spectral Contrast ---
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)  # shape (7,)

    # --- 5. Spectral Centroid ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)

    # --- 6. Spectral Bandwidth ---
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)

    # --- 7. Spectral Rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)

    # --- 8. Zero-Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std  = np.std(zcr)

    # --- 9. RMS Energy ---
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std  = np.std(rms)

    # --- 10. Mel Spectrogram statistics ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)[:20]   # first 20 mel bands mean
    mel_std  = np.std(mel_db,  axis=1)[:20]   # first 20 mel bands std

    # Concatenate all features
    features = np.concatenate([
        mfcc_mean,          # 40
        mfcc_std,           # 40
        mfcc_delta_mean,    # 40
        chroma_mean,        # 12
        contrast_mean,      # 7
        [centroid_mean, bandwidth_mean, rolloff_mean],  # 3
        [zcr_mean, zcr_std],   # 2
        [rms_mean, rms_std],   # 2
        mel_mean,           # 20
        mel_std,            # 20
    ])
    # Total: 40+40+40+12+7+3+2+2+20+20 = 186 features

    return features.astype(np.float32)


def feature_names() -> list:
    """Return human-readable names for each feature dimension."""
    names = []
    names += [f"mfcc_mean_{i}" for i in range(40)]
    names += [f"mfcc_std_{i}"  for i in range(40)]
    names += [f"mfcc_delta_mean_{i}" for i in range(40)]
    names += [f"chroma_{i}"    for i in range(12)]
    names += [f"contrast_{i}"  for i in range(7)]
    names += ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]
    names += ["zcr_mean", "zcr_std"]
    names += ["rms_mean", "rms_std"]
    names += [f"mel_mean_{i}"  for i in range(20)]
    names += [f"mel_std_{i}"   for i in range(20)]
    return names
