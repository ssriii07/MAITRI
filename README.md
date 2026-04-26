<h1 align="center">🚀 MAITRI</h1>

<p align="center">
<b>Multimodal AI System for Autonomous Astronaut Well-Being</b><br>
<i>Designed for the moment when Earth cannot respond</i>
</p>

<p align="center">
AI • Offline • Explainable • Secure • Real-Time
</p>

---

## 🌑 When Space Goes Silent

At Mars conjunction, communication delays exceed **44 minutes**.

No guidance.
No reassurance.
No intervention.

Just silence.

**MAITRI is built for that exact condition.**

An on-device AI system that independently understands, monitors, and supports
human mental and physiological state—without relying on Earth.

---

## 🧠 What This Actually Is

MAITRI is not a chatbot.
Not a model.
Not a dashboard.

It is a **multi-layered autonomous decision system** designed to interpret human state
and respond with context-aware, explainable, and grounded intelligence.

---

## ⚙️ System Flow

Human State → Signal → Understanding → Reasoning → Decision → Support

---

## 🧩 Core Capabilities

* Detects physiological stress (Random Forest — 81.61% accuracy)
* Understands emotional context from text
* Generates grounded responses using RAG (FAISS + Llama 3)
* Explains decisions using SHAP
* Tracks mood over time and detects anomalies
* Fully offline with AES-256 encryption

---

## 📊 Performance

| Metric    | Value       |
| --------- | ----------- |
| Accuracy  | 81.61%      |
| F1 Score  | 81.05%      |
| Recall    | 74%         |
| Latency   | 1.5s        |
| Stability | 100% uptime |

---

## 🏗️ Project Structure

MAITRI/
├── backend/
├── frontend/
├── ml/
└── database/

---

## 🛠️ Tech Stack

Python • FastAPI • React • Tailwind
scikit-learn • SHAP • neurokit2
MentalBERT • SpaCy • VADER
FAISS • Llama 3 • Whisper

---

## 🚀 Run Locally

git clone https://github.com/ssriii07/MAITRI.git
cd MAITRI

pip install -r requirements.txt
ollama run llama3

python -m uvicorn backend.main:app --reload --port 8001

cd frontend
npm install
npm run dev

---

## 🌍 Why This Matters

> Humans must function without real-time human support.

MAITRI enables autonomous mental and physiological care
in environments where Earth cannot help.

---

<p align="center">
🛰️ Built for Deep Space • 🧠 Focused on Human Resilience
</p>
