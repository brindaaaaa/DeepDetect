"""
train_waveform.py  (View 3: Raw Waveform / 1D CNN)
Trains a 1D CNN directly on raw 48 000-sample waveforms to capture
temporal speech patterns and rhythm anomalies in AI-generated audio.

Usage:
    python train_waveform.py

Output:
    model_waveform.keras
"""

import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ──────────────────────────────────────────────────────────────────────
# TODO: UPDATE PATHS — point these to your local dataset directories
# ──────────────────────────────────────────────────────────────────────
FOR_ROOT      = r"T:\d1_for\for\for-norm\for-norm"  # FoR dataset root
ASVSPOOF_ROOT = ""   # Not used — set if you have ASVspoof 2019
# ──────────────────────────────────────────────────────────────────────

MAX_PER_CLASS   = 1500
BATCH_SIZE      = 32
EPOCHS          = 40
SR              = 16_000
WAVEFORM_LENGTH = 48_000   # 3 s × 16 kHz

from preprocessor import load_dataset
from waveform_extractor import extract_waveform


def build_dataset_dirs():
    real_dirs, fake_dirs = [], []
    if FOR_ROOT and os.path.isdir(FOR_ROOT):
        real_dirs += [os.path.join(FOR_ROOT, "training", "real"),
                      os.path.join(FOR_ROOT, "testing",  "real")]
        fake_dirs += [os.path.join(FOR_ROOT, "training", "fake"),
                      os.path.join(FOR_ROOT, "testing",  "fake")]
        print(f"  ✅ FoR dataset: {FOR_ROOT}")
    else:
        print("  ⚠️  FOR_ROOT not set — skipping FoR")

    if ASVSPOOF_ROOT and os.path.isdir(ASVSPOOF_ROOT):
        for split in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev"]:
            d = os.path.join(ASVSPOOF_ROOT, split, "flac")
            if os.path.isdir(d):
                real_dirs.append(d)
        print(f"  ✅ ASVspoof dataset: {ASVSPOOF_ROOT}")
    else:
        print("  ⚠️  ASVSPOOF_ROOT not set — skipping ASVspoof")

    return real_dirs, fake_dirs


def segments_to_waveforms(segments, sr=SR):
    """Convert (waveform, label) list to padded/trimmed arrays."""
    X, y = [], []
    for i, (wav, label) in enumerate(segments):
        if i % 200 == 0:
            print(f"  [{i}/{len(segments)}] Preparing waveforms...", end="\r")
        x = extract_waveform(wav, sr)          # (48 000,)
        X.append(x)
        y.append(label)
    print()
    return (np.array(X, dtype=np.float32).reshape(-1, WAVEFORM_LENGTH, 1),
            np.array(y, dtype=np.float32))


def build_1d_cnn(input_length=WAVEFORM_LENGTH):
    """
    1D CNN architecture for raw waveform classification.

    Architecture:
        Input(48000, 1)
        → Conv1D(32, 9) → BN → ReLU → MaxPool(4)     [12000]
        → Conv1D(64, 9) → BN → ReLU → MaxPool(4)     [3000]
        → Conv1D(128, 9) → BN → ReLU → MaxPool(4)    [750]
        → Conv1D(256, 9) → BN → ReLU → MaxPool(4)    [187]
        → GlobalAvgPool1D                             [256]
        → Dense(128) → Dropout(0.4) → Dense(1, sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        # Block 1
        layers.Conv1D(32, kernel_size=9, padding="same", activation="relu",
                      input_shape=(input_length, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        # Block 2
        layers.Conv1D(64, kernel_size=9, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        # Block 3
        layers.Conv1D(128, kernel_size=9, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        # Block 4
        layers.Conv1D(256, kernel_size=9, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        # Aggregation
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid", name="output"),  # FAKE probability
    ], name="waveform_1d_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    print("🌊  Multi-View Audio Deepfake — View 3: Waveform 1D CNN Training")
    print("=" * 65)

    # ── Load datasets ─────────────────────────────────────────
    print("\n📥 Loading audio segments...")
    real_dirs, fake_dirs = build_dataset_dirs()

    if real_dirs or fake_dirs:
        real_segs, fake_segs = load_dataset(
            real_dirs=real_dirs,
            fake_dirs=fake_dirs,
            max_per_class=MAX_PER_CLASS,
            sr=SR,
        )
    else:
        raise RuntimeError(
            "No dataset paths configured.\n"
            "Please set FOR_ROOT and/or ASVSPOOF_ROOT at the top of train_waveform.py\n"
            "# TODO: UPDATE PATH"
        )

    all_segs = real_segs + fake_segs
    print(f"  Real: {len(real_segs)}  |  Fake: {len(fake_segs)}")

    # ── Prepare waveform tensors ──────────────────────────────
    print("\n🌊 Preparing raw waveform tensors (48 000 samples each)...")
    t0 = time.time()
    X, y = segments_to_waveforms(all_segs)
    print(f"  X shape: {X.shape}  ({time.time()-t0:.1f}s)")

    # ── Train / test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Build and train ───────────────────────────────────────
    import tensorflow as tf

    model = build_1d_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-7
        ),
    ]

    print("\n⚡ Training 1D CNN...")
    t1 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\n  Training done in {time.time()-t1:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n✅ 1D CNN ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # ── Save ─────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "model_waveform.keras")
    model.save(out_path)
    print(f"\n💾 Waveform CNN saved: {out_path}")
    print("\n➡️  Next: python train_ensemble.py")
