"""
train_cnn.py  (View 2: Spectrogram / VGG16 CNN)
Fine-tunes a VGG16 model on 224×224 Mel spectrogram images to detect
spectral artifacts introduced by AI speech synthesis.

Usage:
    python train_cnn.py

Output:
    model_cnn.keras
"""

import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# ──────────────────────────────────────────────────────────────────────
# TODO: UPDATE PATHS — point these to your local dataset directories
# ──────────────────────────────────────────────────────────────────────
FOR_ROOT      = r"T:\d1_for\for\for-norm\for-norm"  # FoR dataset root
ASVSPOOF_ROOT = ""   # Not used — set if you have ASVspoof 2019
# ──────────────────────────────────────────────────────────────────────

MAX_PER_CLASS = 1500   # Limit per class (CNN training is memory-intensive)
BATCH_SIZE    = 32
EPOCHS        = 30
IMG_SIZE      = 224
SR            = 16_000

from preprocessor import load_dataset
from spectrogram_extractor import extract_spectrogram


def build_dataset_dirs():
    real_dirs, fake_dirs = [], []
    if FOR_ROOT and os.path.isdir(FOR_ROOT):
        real_dirs += [os.path.join(FOR_ROOT, "training", "real"),
                      os.path.join(FOR_ROOT, "testing", "real")]
        fake_dirs += [os.path.join(FOR_ROOT, "training", "fake"),
                      os.path.join(FOR_ROOT, "testing", "fake")]
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


def segments_to_spectrograms(segments, sr=SR):
    """Convert a list of (waveform, label) tuples to (image, label) pairs."""
    images, labels = [], []
    for i, (y, label) in enumerate(segments):
        if i % 100 == 0:
            print(f"  [{i}/{len(segments)}] Converting to spectrograms...", end="\r")
        img = extract_spectrogram(y, sr)   # (224, 224, 3)
        images.append(img)
        labels.append(label)
    print()
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)


def build_vgg16_model():
    """Build VGG16 with frozen conv base + new dense classification head."""
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False   # Freeze conv base initially

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid", name="output"),  # Binary: REAL=0, FAKE=1
    ], name="spectrogram_vgg16")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


if __name__ == "__main__":
    print("🖼️   Multi-View Audio Deepfake — View 2: Spectrogram VGG16 Training")
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
            "Please set FOR_ROOT and/or ASVSPOOF_ROOT at the top of train_cnn.py\n"
            "# TODO: UPDATE PATH"
        )

    all_segs = real_segs + fake_segs
    print(f"  Real: {len(real_segs)}  |  Fake: {len(fake_segs)}")

    # ── Convert to spectrograms ───────────────────────────────
    print("\n🎨 Converting waveforms to Mel spectrogram images (224×224×3)...")
    t0 = time.time()
    X, y = segments_to_spectrograms(all_segs)
    print(f"  X shape: {X.shape}  |  Labels: {y.shape}  ({time.time()-t0:.1f}s)")

    # ── Train / test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Build model ───────────────────────────────────────────
    print("\n🤖 Building VGG16 model...")
    model, base = build_vgg16_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7
        ),
    ]

    # ── Phase 1: Train dense head with frozen base ────────────
    print("\n⚡ Phase 1: Training classifier head (conv base frozen)...")
    t1 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"  Phase 1 done in {time.time()-t1:.1f}s")

    # ── Phase 2: Fine-tune last 4 conv layers ─────────────────
    print("\n⚡ Phase 2: Fine-tuning last 4 layers of VGG16...")
    base.trainable = True
    for layer in base.layers[:-4]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n✅ VGG16 ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # ── Save ─────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "model_cnn.keras")
    model.save(out_path)
    print(f"\n💾 CNN model saved: {out_path}")
    print("\n➡️  Next: python train_waveform.py")
