"""
train_ensemble.py  (Step 4 — run AFTER all three base models are trained)
Trains a Linear Regression meta-learner (stacking) on a shared validation set.

How it works:
  1. Load all three trained models (SVM, VGG16-CNN, 1D-CNN)
  2. Run inference on a held-out validation set from both datasets
  3. Stack predictions into X_meta (N×3) → fit LinearRegression on y (0/1)
  4. Normalize learned coefficients so they sum to 1 → interpretable weights
  5. Also fit LogisticRegression for comparison (uses probabilities directly)
  6. Save the best meta-learner as model_ensemble.joblib
  7. Print full evaluation table: Accuracy, Precision, Recall, F1, ROC-AUC, EER

Usage:
    python train_ensemble.py

Output:
    model_ensemble.joblib
"""

import os
import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)

# ──────────────────────────────────────────────────────────────────────
# TODO: UPDATE PATHS — point these to your local dataset directories
# ──────────────────────────────────────────────────────────────────────
FOR_ROOT      = r"T:\d1_for\for\for-norm\for-norm"  # FoR dataset root
ASVSPOOF_ROOT = ""   # Not used — set if you have ASVspoof 2019
# ──────────────────────────────────────────────────────────────────────

MAX_VAL_SAMPLES = 800   # validation samples per class for meta-learner training
SR = 16_000


# ─────────────────────────────────────────────────────────
# Imports from other modules (must be in same directory)
# ─────────────────────────────────────────────────────────
from preprocessor import load_dataset
from feature_extractor import extract_acoustic_features
from spectrogram_extractor import extract_spectrogram
from waveform_extractor import extract_waveform


# ─────────────────────────────────────────────────────────
# EER calculation
# ─────────────────────────────────────────────────────────
def compute_eer(y_true, y_score):
    """Compute Equal Error Rate (EER)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    # EER is where FPR ≈ FNR
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


# ─────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────
def evaluate_model(name, y_true, y_prob, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    eer  = compute_eer(y_true, y_prob)
    return {"Model": name, "Acc": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC": auc, "EER": eer}


def print_metrics_table(rows):
    header = f"{'Model':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'EER':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['Model']:<18} "
              f"{r['Acc']*100:>8.1f}% "
              f"{r['Precision']*100:>9.1f}% "
              f"{r['Recall']*100:>7.1f}% "
              f"{r['F1']*100:>7.1f}% "
              f"{r['AUC']:>8.4f} "
              f"{r['EER']:>8.4f}")
    print("=" * len(header))


def build_dataset_dirs():
    real_dirs, fake_dirs = [], []
    if FOR_ROOT and os.path.isdir(FOR_ROOT):
        real_dirs += [os.path.join(FOR_ROOT, "testing", "real")]
        fake_dirs += [os.path.join(FOR_ROOT, "testing", "fake")]
    if ASVSPOOF_ROOT and os.path.isdir(ASVSPOOF_ROOT):
        d = os.path.join(ASVSPOOF_ROOT, "ASVspoof2019_LA_eval", "flac")
        if os.path.isdir(d):
            real_dirs.append(d)
    return real_dirs, fake_dirs


if __name__ == "__main__":
    print("🔗  Multi-View Ensemble — Linear Regression Meta-Learner Training")
    print("=" * 65)

    # ── Load model files ──────────────────────────────────────
    base_dir = os.path.dirname(__file__)

    svm_path  = os.path.join(base_dir, "model_svm.joblib")
    cnn_path  = os.path.join(base_dir, "model_cnn.keras")
    wave_path = os.path.join(base_dir, "model_waveform.keras")

    for label, path in [("SVM", svm_path), ("CNN", cnn_path), ("WaveCNN", wave_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} model not found: {path}\nRun train*.py first.")

    print("\n📦 Loading base models...")
    svm_model = joblib.load(svm_path)
    import tensorflow as tf
    cnn_model  = tf.keras.models.load_model(cnn_path)
    wave_model = tf.keras.models.load_model(wave_path)
    print("  ✅ All three base models loaded.")

    # ── Load validation data ──────────────────────────────────
    print("\n📥 Loading validation dataset for meta-learner training...")
    real_dirs, fake_dirs = build_dataset_dirs()

    if real_dirs or fake_dirs:
        real_segs, fake_segs = load_dataset(
            real_dirs=real_dirs,
            fake_dirs=fake_dirs,
            max_per_class=MAX_VAL_SAMPLES,
            sr=SR,
        )
    else:
        # Fallback: use librosa built-ins (for development without datasets)
        print("  ⚠️  No dataset paths set — using librosa fallback for meta-learner")
        print("  ⚠️  This will produce a rough meta-learner; update paths for production")
        import librosa
        real_segs, fake_segs = [], []
        for name in ["libri1", "libri2", "libri3"]:
            try:
                path = librosa.ex(name)
                y, _ = librosa.load(path, sr=SR, mono=True)
                seg_len = int(SR * 3.0)
                for start in range(0, len(y) - seg_len, seg_len):
                    real_segs.append((y[start:start+seg_len].astype(np.float32), 0))
            except Exception:
                pass
        # Minimal synthetic fake for fallback
        rng = np.random.default_rng(99)
        t = np.linspace(0, 3.0, int(SR*3.0))
        for _ in range(min(len(real_segs), 100)):
            f0 = rng.uniform(100, 280)
            y = sum((1.0/k)*np.sin(2*np.pi*k*f0*t) for k in range(1, 8)).astype(np.float32)
            y /= max(np.max(np.abs(y)), 1e-6)
            fake_segs.append((y, 1))

    all_segs = real_segs + fake_segs
    print(f"  Real: {len(real_segs)}  |  Fake: {len(fake_segs)}")

    # ── Generate model probability scores ─────────────────────
    print("\n⚙️  Running all three base models on validation set...")
    t0 = time.time()

    P_svm, P_cnn, P_wave, Y = [], [], [], []

    for i, (wav, label) in enumerate(all_segs):
        if i % 100 == 0:
            print(f"  [{i}/{len(all_segs)}]", end="\r")

        # SVM — acoustic features
        feat = extract_acoustic_features(wav, SR)
        p_svm = float(svm_model.predict_proba(feat.reshape(1, -1))[0][1])

        # VGG16 CNN — spectrogram
        spec = extract_spectrogram(wav, SR)
        p_cnn = float(cnn_model.predict(spec[np.newaxis], verbose=0)[0][0])

        # 1D CNN — waveform
        wv = extract_waveform(wav, SR).reshape(1, -1, 1)
        p_wave = float(wave_model.predict(wv, verbose=0)[0][0])

        P_svm.append(p_svm)
        P_cnn.append(p_cnn)
        P_wave.append(p_wave)
        Y.append(label)

    print(f"\n  Inference done in {time.time()-t0:.1f}s")

    P_svm  = np.array(P_svm,  dtype=np.float32)
    P_cnn  = np.array(P_cnn,  dtype=np.float32)
    P_wave = np.array(P_wave, dtype=np.float32)
    Y      = np.array(Y,       dtype=np.float32)

    # Meta-learner features: [p_cnn, p_waveform, p_svm]  (order matches ensemble.py)
    X_meta = np.column_stack([P_cnn, P_wave, P_svm])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_meta, Y, test_size=0.25, random_state=42, stratify=Y.astype(int)
    )

    # ── Fit Linear Regression meta-learner ────────────────────
    print("\n📐 Fitting Linear Regression meta-learner...")
    linreg = LinearRegression()
    linreg.fit(X_tr, y_tr)

    coefs  = linreg.coef_                         # [w_cnn, w_waveform, w_svm]
    coefs_pos = np.maximum(coefs, 0)              # clip negatives to 0
    total = coefs_pos.sum()
    if total > 1e-6:
        norm_weights = coefs_pos / total
    else:
        norm_weights = np.array([1/3, 1/3, 1/3])

    print(f"\n  Linear Regression coefficients (raw): {coefs}")
    print(f"  Normalized weights:")
    print(f"    CNN        (VGG16) : {norm_weights[0]:.4f}  ({norm_weights[0]*100:.1f}%)")
    print(f"    Waveform   (1DCNN) : {norm_weights[1]:.4f}  ({norm_weights[1]*100:.1f}%)")
    print(f"    Acoustic   (SVM)   : {norm_weights[2]:.4f}  ({norm_weights[2]*100:.1f}%)")

    # ── Also fit Logistic Regression (comparison) ─────────────
    print("\n📊 Fitting Logistic Regression meta-learner (comparison)...")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_tr, y_tr.astype(int))

    # ── Evaluate all models on validation split ───────────────
    print("\n📈 Evaluating all models on validation split...")
    metrics_rows = []

    # Individual models
    for mname, probs in [("SVM", X_val[:, 2]), ("VGG16-CNN", X_val[:, 0]),
                          ("WaveCNN-1D", X_val[:, 1])]:
        preds = (probs > 0.5).astype(int)
        metrics_rows.append(evaluate_model(mname, y_val.astype(int), probs, preds))

    # LinReg ensemble
    lr_raw  = linreg.predict(X_val)
    lr_prob = np.clip(lr_raw, 0.0, 1.0)
    lr_pred = (lr_prob > 0.5).astype(int)
    metrics_rows.append(evaluate_model("LinReg-Ensemble", y_val.astype(int), lr_prob, lr_pred))

    # LogReg ensemble
    log_prob = logreg.predict_proba(X_val)[:, 1]
    log_pred = (log_prob > 0.5).astype(int)
    metrics_rows.append(evaluate_model("LogReg-Ensemble", y_val.astype(int), log_prob, log_pred))

    print_metrics_table(metrics_rows)

    # ── Choose best meta-learner ──────────────────────────────
    best_lr  = metrics_rows[3]["AUC"]   # LinReg AUC
    best_log = metrics_rows[4]["AUC"]   # LogReg AUC

    if best_lr >= best_log:
        best_model = linreg
        best_name  = "Linear Regression"
    else:
        best_model = logreg
        best_name  = "Logistic Regression"

    print(f"\n✅ Selected meta-learner: {best_name}  (AUC: {max(best_lr, best_log):.4f})")

    # ── Save ensemble model ───────────────────────────────────
    out_path = os.path.join(base_dir, "model_ensemble.joblib")
    joblib.dump(best_model, out_path)
    print(f"\n💾 Ensemble model saved: {out_path}")
    print("\n🚀 All models ready! Start the API:")
    print("   uvicorn main:app --reload --port 8000")
