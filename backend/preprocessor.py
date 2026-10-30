"""
preprocessor.py
Shared audio preprocessing module for all three detection views.

Steps:
  1. Load audio from bytes or file path
  2. Convert to mono
  3. Resample to 16 kHz
  4. Normalize amplitude to [-1, 1]
  5. Segment into fixed-length clips (default 3 seconds)
"""

import numpy as np
import librosa
import io
from typing import List, Tuple, Union


SR_TARGET: int = 16_000          # Standard sample rate (Hz)
SEGMENT_DURATION: float = 3.0    # Clip length in seconds
SEGMENT_SAMPLES: int = int(SR_TARGET * SEGMENT_DURATION)  # 48 000 samples


# ─────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────

def load_audio(source: Union[bytes, str], sr: int = SR_TARGET) -> Tuple[np.ndarray, int]:
    """
    Load audio from raw bytes or a file path.
    Returns (waveform_float32, sample_rate).
    """
    if isinstance(source, (bytes, bytearray)):
        buf = io.BytesIO(source)
        y, sr_loaded = librosa.load(buf, sr=sr, mono=True)
    else:
        y, sr_loaded = librosa.load(source, sr=sr, mono=True)
    return y.astype(np.float32), sr_loaded


def normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalize waveform to [-1, 1]. Returns unchanged if silent."""
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak
    return y.astype(np.float32)


def segment(y: np.ndarray, seg_len: int = SEGMENT_SAMPLES,
            hop: int = None) -> List[np.ndarray]:
    """
    Split a waveform into non-overlapping (or overlapping) fixed-length clips.
    Clips shorter than seg_len are zero-padded.
    Returns a list of numpy arrays each of length seg_len.
    """
    if hop is None:
        hop = seg_len  # non-overlapping by default
    clips = []
    start = 0
    while start < len(y):
        clip = y[start: start + seg_len]
        if len(clip) < seg_len:
            clip = np.pad(clip, (0, seg_len - len(clip)))
        clips.append(clip.astype(np.float32))
        start += hop
    return clips


# ─────────────────────────────────────────────────────────
# Main pipeline entry-point (used by API)
# ─────────────────────────────────────────────────────────

def preprocess_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline for a single audio upload.
    Returns: (normalized_mono_waveform, sample_rate)
    """
    y, sr = load_audio(audio_bytes)
    y = normalize(y)
    return y, sr


def preprocess_to_segments(audio_bytes: bytes,
                            seg_duration: float = SEGMENT_DURATION
                            ) -> List[np.ndarray]:
    """
    Full pipeline that returns a list of fixed-length 3-second segments.
    Used by training scripts to build per-clip datasets.
    """
    y, sr = preprocess_audio_bytes(audio_bytes)
    seg_len = int(sr * seg_duration)
    return segment(y, seg_len)


# ─────────────────────────────────────────────────────────
# Dataset loader helpers (used by training scripts)
# ─────────────────────────────────────────────────────────

import os
import random


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def collect_audio_files(root_dir: str) -> List[str]:
    """
    Recursively collect all audio file paths under root_dir.
    Returns a sorted list of absolute paths.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTENSIONS:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def load_dataset(real_dirs: List[str],
                 fake_dirs: List[str],
                 max_per_class: int = 2000,
                 seg_duration: float = SEGMENT_DURATION,
                 sr: int = SR_TARGET,
                 seed: int = 42) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """
    Load and preprocess audio files from real and fake dataset directories.
    Segments each file into SEGMENT_DURATION-second clips.
    Returns: (real_segments, fake_segments) where each item is (waveform, label).

    Args:
        real_dirs: list of directories containing real/genuine audio files
        fake_dirs: list of directories containing fake/spoof audio files
        max_per_class: maximum number of segments to load per class
        seg_duration: segment length in seconds
        sr: target sample rate
        seed: for reproducibility when shuffling

    NOTE: If real_dirs or fake_dirs contain empty strings ('') they will be skipped.
    """
    rng = random.Random(seed)
    seg_len = int(sr * seg_duration)

    def _load_from_dirs(dirs: List[str], label: int) -> List[Tuple[np.ndarray, int]]:
        all_files = []
        for d in dirs:
            if not d or not os.path.isdir(d):
                continue
            all_files.extend(collect_audio_files(d))
        rng.shuffle(all_files)

        segs = []
        for fpath in all_files:
            if len(segs) >= max_per_class:
                break
            try:
                y, _ = librosa.load(fpath, sr=sr, mono=True)
                y = normalize(y)
                for clip in segment(y, seg_len):
                    if len(segs) < max_per_class:
                        segs.append((clip, label))
            except Exception as e:
                print(f"  [WARN] Could not load {fpath}: {e}")
        return segs

    real_segs = _load_from_dirs(real_dirs, label=0)
    fake_segs = _load_from_dirs(fake_dirs, label=1)
    return real_segs, fake_segs
