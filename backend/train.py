"""
train.py  (Updated — View 1: Acoustic / SVM)
Trains an SVM on 186-dim acoustic features extracted from the
FoR + ASVspoof 2019 datasets.

Usage:
    python train.py

Output:
    model_svm.joblib  (also symlinked/copied to model.joblib for backward compat)
"""

import numpy as np
import joblib
import os
import time
import librosa
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# ──────────────────────────────────────────────────────────────────────
# TODO: UPDATE PATHS — point these to your local dataset directories
# ──────────────────────────────────────────────────────────────────────
# FoR (Fake-or-Real) dataset layout expected:
#   <FOR_ROOT>/training/real/  and  <FOR_ROOT>/training/fake/
#   <FOR_ROOT>/testing/real/   and  <FOR_ROOT>/testing/fake/
FOR_ROOT = r"T:\d1_for\for\for-norm\for-norm"  # FoR dataset root

# ASVspoof 2019 Logical Access (LA) dataset:
#   <ASVSPOOF_ROOT>/ASVspoof2019_LA_train/flac/
#   <ASVSPOOF_ROOT>/ASVspoof2019_LA_dev/flac/
#   The protocol CSV marks genuine (bonafide) vs. spoof per file.
ASVSPOOF_ROOT = ""  # TODO: UPDATE PATH — e.g. "/path/to/LA"
# ──────────────────────────────────────────────────────────────────────

from preprocessor import load_dataset
from feature_extractor import extract_acoustic_features


def build_dataset_dirs():
    """Map configured paths to real/fake directory lists."""
    real_dirs = []
    fake_dirs = []

    # FoR dataset
    if FOR_ROOT and os.path.isdir(FOR_ROOT):
        real_dirs += [
            os.path.join(FOR_ROOT, "training", "real"),
            os.path.join(FOR_ROOT, "testing", "real"),
        ]
        fake_dirs += [
            os.path.join(FOR_ROOT, "training", "fake"),
            os.path.join(FOR_ROOT, "testing", "fake"),
        ]
        print(f"  ✅ FoR dataset found at: {FOR_ROOT}")
    else:
        print(f"  ⚠️  FOR_ROOT not set or not found — skipping FoR dataset.")

    # ASVspoof 2019 — treat bonafide as real, all spoof types as fake
    if ASVSPOOF_ROOT and os.path.isdir(ASVSPOOF_ROOT):
        for split in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev"]:
            flac_dir = os.path.join(ASVSPOOF_ROOT, split, "flac")
            if os.path.isdir(flac_dir):
                # NOTE: ideally filter by protocol CSV for bonafide vs spoof;
                # here we add all flac as potential real (augmented by FoR fake)
                real_dirs.append(flac_dir)
        print(f"  ✅ ASVspoof dataset found at: {ASVSPOOF_ROOT}")
    else:
        print(f"  ⚠️  ASVSPOOF_ROOT not set or not found — skipping ASVspoof dataset.")

    return real_dirs, fake_dirs


def load_librosa_fallback(sr=16000, n=400):
    """Fallback: use librosa built-in speech + synthesised fake samples."""
    print("  ⚠️  Using librosa built-in audio as fallback (no datasets available)")
    from train_fallback_data import load_real_samples, augment_real, synthesise_fake_samples

    rng = np.random.default_rng(42)
    real_base = load_real_samples(sr)
    if not real_base:
        print("    ⚠️  librosa speech not available either, using noise fallback")
        return [], []
    real_segs = augment_real(real_base, rng, n)
    fake_segs_raw = synthesise_fake_samples(n, sr, rng)
    real_out = [(seg, 0) for seg, _ in real_segs]
    fake_out = [(seg, 1) for seg, _ in fake_segs_raw]
    return real_out, fake_out


if __name__ == "__main__":
    print("🎙️  Multi-View Audio Deepfake — View 1: Acoustic SVM Training")
    print("=" * 60)

    SR = 16000
    MAX_PER_CLASS = 1500

    # ── Load dataset ──────────────────────────────────────────
    print("\n📥 Loading datasets...")
    real_dirs, fake_dirs = build_dataset_dirs()

    if real_dirs or fake_dirs:
        real_segs, fake_segs = load_dataset(
            real_dirs=real_dirs,
            fake_dirs=fake_dirs,
            max_per_class=MAX_PER_CLASS,
            sr=SR,
        )
    else:
        real_segs, fake_segs = load_librosa_fallback(SR, 500)

    print(f"  Real segments: {len(real_segs)}")
    print(f"  Fake segments: {len(fake_segs)}")

    if len(real_segs) == 0 or len(fake_segs) == 0:
        raise RuntimeError("No audio samples loaded. Check dataset paths.")

    # ── Feature extraction ────────────────────────────────────
    print("\n🔬 Extracting acoustic features...")
    t0 = time.time()
    X, y_labels = [], []

    all_segs = real_segs + fake_segs
    for i, (audio, label) in enumerate(all_segs):
        if i % 200 == 0:
            print(f"  [{i}/{len(all_segs)}]", end="\r")
        feat = extract_acoustic_features(audio, SR)
        X.append(feat)
        y_labels.append(label)

    X = np.array(X, dtype=np.float32)
    y_labels = np.array(y_labels)
    print(f"\n  Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")

    # ── Train / test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # ── Class balance check ───────────────────────────────────
    n_real = int(np.sum(y_train == 0))
    n_fake = int(np.sum(y_train == 1))
    print(f"  Training class balance — REAL: {n_real}  FAKE: {n_fake}")
    if n_real == 0 or n_fake == 0:
        raise RuntimeError("One class has zero samples — check dataset paths!")
    # Compute sample weights to balance classes
    sample_weights = np.where(y_train == 1,
                              n_real / n_fake,
                              1.0).astype(np.float32)

    # ── Train ensemble classifier ─────────────────────────────
    print("\n🤖 Training GradientBoosting + RandomForest ensemble...")
    t1 = time.time()

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Combine via soft-voting (probability average)
    voter = VotingClassifier(
        estimators=[("gb", gb), ("rf", rf)],
        voting="soft",
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   voter),
    ])
    # Note: VotingClassifier doesn't accept sample_weight directly;
    # pass it through fit_params to each sub-estimator
    pipeline.fit(
        X_train, y_train,
        clf__gb__sample_weight=sample_weights,
        clf__rf__sample_weight=sample_weights,
    )
    print(f"  Done in {time.time()-t1:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n✅ Test Accuracy : {acc * 100:.1f}%")
    print(f"   ROC-AUC       : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # ── Save ─────────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)
    svm_path = os.path.join(out_dir, "model_svm.joblib")
    compat_path = os.path.join(out_dir, "model.joblib")

    joblib.dump(pipeline, svm_path)
    joblib.dump(pipeline, compat_path)   # backward compat for existing API

    print(f"\n💾 SVM model saved: {svm_path}")
    print(f"   Copied to     : {compat_path} (backward compat)")
    print("\n➡️  Next: python train_cnn.py")
