# 🛡️ DeepDetect — Deepfake Audio Detection System

A full-stack AI system that analyzes audio files and determines whether they are **authentic human speech** or **AI-generated deepfake audio**.

---

## 🗂️ Project Structure

```
DeepDetect/
├── backend/
│   ├── main.py                 # FastAPI REST API
│   ├── preprocessor.py         # Audio downsampling & normalization
│   ├── feature_extractor.py    # Acoustic feature extractor (SVM input)
│   ├── spectrogram_extractor.py # Mel-spectrogram extractor (CNN input)
│   ├── waveform_extractor.py   # Raw waveform preprocessor (1D-CNN input)
│   ├── ensemble.py             # Fusion logic for multi-view predictions
│   ├── model.py                # Pipeline loader and prediction interface
│   ├── train.py                # Script to train SVM model
│   ├── train_cnn.py            # Script to train VGG16 CNN model
│   ├── train_waveform.py       # Script to train 1D-CNN model
│   ├── train_ensemble.py       # Script to train Meta-Learner (Ensemble)
│   ├── model_svm.joblib        # Trained SVM classifier
│   ├── model_cnn.keras         # Trained VGG16 CNN model
│   ├── model_waveform.keras    # Trained 1D-CNN model
│   ├── model_ensemble.joblib   # Trained ensemble fusion weights
│   ├── test_backend.py         # Backend pytest unit/integration tests
│   └── requirements.txt        # Python package dependencies
├── frontend/
│   ├── index.html              # Frontend user interface
│   ├── style.css               # Dark glassmorphism styles
│   └── app.js                  # Audio upload, waveform rendering, and API calls
├── vnv/                        # Virtual environment folder with dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the detection model
```bash
python train.py
```
This generates `model.joblib` — required before starting the API.

### 3. Start the API server
```bash
uvicorn main:app --reload --port 8000
```
API will be live at: http://localhost:8000

### 4. Open the Frontend
Open `frontend/index.html` in your browser (Chrome / Safari / Edge).

---

## 🔬 How It Works

| Step | Detail |
|---|---|
| **Audio Upload** | WAV, MP3, FLAC, OGG, M4A supported |
| **Feature Extraction** | 186 features — MFCCs (×3), Chroma, Spectral Contrast, Centroid, ZCR, RMS, Mel Spectrogram |
| **ML Model** | Gradient Boosting Classifier (sklearn Pipeline with StandardScaler) |
| **Output** | `REAL` or `FAKE` + confidence score (0–100%) |

---

## 🧪 Running Tests
```bash
cd backend
python train.py          # Train first
python -m pytest test_backend.py -v
```

---

## 📡 API Reference

### `POST /predict`
Upload an audio file and get a prediction.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 0.923,
  "confidence_pct": "92.3%",
  "message": "🟢 This audio appears to be REAL (confidence: 92.3%)",
  "filename": "sample.wav",
  "feature_count": 186
}
```

### `GET /`
Health check — returns `{"status": "ok"}`.

---

## 🎯 Accuracy Notes

- **Demo model**: Trained on synthetic audio — works as a prototype
- **For production**: Replace synthetic data in `train.py` with [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) dataset
- **Upgrade path**: Swap GradientBoosting for a CNN on mel-spectrograms for higher accuracy

---

*DeepDetect Audio Detection System*
