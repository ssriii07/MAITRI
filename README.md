```markdown
# MAITRI — Multimodal AI Companion for Astronaut Well-Being

> Mental and Agentic Intelligence for Total Resilience Integration

MAITRI is a fully on-device multimodal AI companion system designed 
to monitor and support the psychological and physiological well-being 
of astronauts during long-duration deep-space missions — with zero 
internet dependency.

---

## Problem Statement

Astronauts on deep-space missions face communication delays exceeding 
44 minutes at Mars conjunction, making real-time Earth-based medical 
and psychological consultation impossible. No existing system provides 
autonomous multimodal health monitoring and psychological support 
in a single on-device architecture.

---

## What MAITRI Does

- Detects physiological stress from biosensor data using Random Forest
  trained on WESAD dataset — 81.61% LOSO accuracy
- Analyses emotional state from text using MentalBERT, SpaCy and VADER
- Generates evidence-grounded responses using FAISS + Llama 3 RAG
  pipeline grounded in NASA health guidelines
- Explains every prediction using SHAP top-5 feature importance
- Monitors mood trends longitudinally and detects personal anomalies
- Stores all data privately with AES-256 encryption on-device

---


## System Architecture

```
Layer 1 — Input              : Text, Voice, Physiological CSV
Layer 2 — Feature Extraction : MentalBERT, Whisper, neurokit2
Layer 3 — AI Core            : Fusion Classifier, RAG, SHAP
Layer 4 — Decision Engine    : Risk Tier LOW / MODERATE / HIGH
Layer 5 — Output             : Chat, Dashboard, Journal, Alerts
```

---

## Results

| Metric | Value |
|---|---|
| LOSO Accuracy | 81.61% |
| Weighted F1 | 81.05% |
| Stress Recall (threshold 0.35) | 74% |
| API Response Latency | 1.5s avg |
| System Uptime (72hr) | 100% |
| User Satisfaction | 4.6 / 5.0 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI 0.128 |
| Frontend | React.js 18, Tailwind CSS |
| Database | SQLite + AES-256 encryption |
| ML | scikit-learn, neurokit2, SHAP |
| NLP | MentalBERT, SpaCy, VADER |
| RAG | FAISS, Llama 3 7B via Ollama |
| Speech | Whisper, Librosa |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ssriii07/MAITRI.git
cd MAITRI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ollama with Llama 3
```bash
ollama run llama3
```

### 4. Start the backend
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 5. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

### 6. Open in browser
```
http://localhost:5175
```

---

## Hardware Requirements

- RAM: 16GB minimum
- Storage: 10GB minimum
- CPU: Multi-core (no GPU required)
- OS: macOS / Linux / Windows
```
