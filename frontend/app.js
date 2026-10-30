/**
 * app.js — DeepGuard Frontend Logic
 * Handles: drag-drop upload, WaveSurfer waveform, API call, result rendering
 */

const API_URL = "http://localhost:8000/predict";

// ── DOM refs ──────────────────────────────────────────────────
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const clearBtn = document.getElementById("clearBtn");

const waveformSection = document.getElementById("waveformSection");
const analyzeWrap = document.getElementById("analyzeWrap");
const analyzeBtn = document.getElementById("analyzeBtn");
const playBtn = document.getElementById("playBtn");
const waveTime = document.getElementById("waveTime");

const processingSection = document.getElementById("processingSection");
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");

const resultPanel = document.getElementById("resultPanel");
const resultBadge = document.getElementById("resultBadge");
const resultEmoji = document.getElementById("resultEmoji");
const resultLabel = document.getElementById("resultLabel");
// confidenceValue / confidenceFill removed — those IDs are not in the HTML
const resultMessage = document.getElementById("resultMessage");
const metaFile = document.getElementById("metaFile");
const metaFeatures = document.getElementById("metaFeatures");
const analyzeAgainBtn = document.getElementById("analyzeAgainBtn");

const errorPanel = document.getElementById("errorPanel");
const errorMessage = document.getElementById("errorMessage");
const retryBtn = document.getElementById("retryBtn");

// ── State ─────────────────────────────────────────────────────
let selectedFile = null;
let wavesurfer = null;

// ── WaveSurfer init ───────────────────────────────────────────
function initWaveSurfer() {
    if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
    wavesurfer = WaveSurfer.create({
        container: "#waveform",
        waveColor: "rgba(99,179,237,0.6)",
        progressColor: "#63b3ed",
        cursorColor: "#f687b3",
        barWidth: 2,
        barRadius: 3,
        barGap: 1,
        height: 72,
        responsive: true,
        backend: "WebAudio",
        normalize: true,
    });

    wavesurfer.on("ready", () => {
        const dur = wavesurfer.getDuration();
        waveTime.textContent = formatTime(dur);
    });
    wavesurfer.on("audioprocess", () => {
        const cur = wavesurfer.getCurrentTime();
        const dur = wavesurfer.getDuration();
        waveTime.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
    });
    wavesurfer.on("play", () => { playBtn.textContent = "⏸ Pause"; });
    wavesurfer.on("pause", () => { playBtn.textContent = "▶ Play"; });
    wavesurfer.on("finish", () => { playBtn.textContent = "▶ Play"; });
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
}

// ── File handling ─────────────────────────────────────────────
function handleFile(file) {
    if (!file) return;

    const allowedExts = ["wav", "mp3", "flac", "ogg", "m4a", "aac"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowedExts.includes(ext)) {
        showError(`Unsupported format: .${ext}. Please upload WAV, MP3, FLAC, OGG, or M4A.`);
        return;
    }

    selectedFile = file;

    // Show file info bar
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    show(fileInfo);

    // Load waveform
    show(waveformSection);
    initWaveSurfer();
    const objectUrl = URL.createObjectURL(file);
    wavesurfer.load(objectUrl);

    // Show analyze button; hide any previous result
    show(analyzeWrap);
    hide(resultPanel);
    hide(errorPanel);
    hide(processingSection);
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function resetUI() {
    selectedFile = null;
    fileInput.value = "";
    if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
    hide(fileInfo);
    hide(waveformSection);
    hide(analyzeWrap);
    hide(resultPanel);
    hide(errorPanel);
    hide(processingSection);
}

// ── Drag & Drop ───────────────────────────────────────────────
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
});
dropZone.addEventListener("click", (e) => {
    // Avoid double-triggering when the label/button is clicked
    if (e.target.tagName !== "LABEL" && e.target.tagName !== "INPUT") {
        fileInput.click();
    }
});
fileInput.addEventListener("change", () => {
    if (fileInput.files?.[0]) handleFile(fileInput.files[0]);
});
clearBtn.addEventListener("click", resetUI);
analyzeAgainBtn.addEventListener("click", resetUI);
retryBtn.addEventListener("click", () => { hide(errorPanel); });

// ── Play button ───────────────────────────────────────────────
playBtn.addEventListener("click", () => {
    if (wavesurfer) wavesurfer.playPause();
});

// ── Analyse ───────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Show processing state
    hide(analyzeWrap);
    hide(resultPanel);
    hide(errorPanel);
    show(processingSection);
    resetSteps();

    try {
        // Animate processing steps
        await animateStep(step1, 800);
        await animateStep(step2, 900);

        // Build form data and send to API
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        await animateStep(step3, 400);

        hide(processingSection);
        showResult(data);

    } catch (err) {
        hide(processingSection);
        let msg = err.message || "Unknown error";
        if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
            msg = "Cannot reach the DeepDetect API server. Make sure the backend is running on port 8000.";
        }
        showError(msg);
    }
});

// ── Processing step animation ─────────────────────────────────
function resetSteps() {
    [step1, step2, step3].forEach(s => {
        s.classList.remove("active", "done");
    });
}

function animateStep(el, delay) {
    return new Promise((resolve) => {
        el.classList.add("active");
        setTimeout(() => {
            el.classList.remove("active");
            el.classList.add("done");
            resolve();
        }, delay);
    });
}

// ── Result rendering ──────────────────────────────────────────
function showResult(data) {
    const isReal = data.prediction === "REAL";

    resultBadge.className = `result-badge ${isReal ? "real" : "fake"}`;
    resultEmoji.textContent = isReal ? "🟢" : "🔴";
    resultLabel.textContent = data.prediction;

    // Plain verdict sentence — no confusing %
    const resultVerdict = document.getElementById("resultVerdict");
    if (resultVerdict) {
        resultVerdict.textContent = isReal
            ? "This audio is REAL — genuine human speech."
            : "This audio is FAKE — AI-generated deepfake.";
        resultVerdict.className = `result-verdict ${isReal ? "verdict-real" : "verdict-fake"}`;
    }

    resultMessage.textContent = data.message;
    metaFile.textContent = `📁 ${data.filename || selectedFile?.name || "—"}`;
    metaFeatures.textContent = `📊 ${data.feature_count} features`;

    show(resultPanel);
}

// ── Error rendering ───────────────────────────────────────────
function showError(msg) {
    errorMessage.textContent = msg;
    show(errorPanel);
    show(analyzeWrap);
}

// ── Helpers ───────────────────────────────────────────────────
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }
