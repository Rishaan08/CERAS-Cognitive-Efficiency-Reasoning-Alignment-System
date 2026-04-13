# CERAS: Cognitive Efficiency and Reasoning Alignment System

![Status](https://img.shields.io/badge/Status-Active%20Prototype-success)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB)
![Reasoning](https://img.shields.io/badge/Reasoning-Tree%20of%20Thoughts-blueviolet)
![LLM Providers](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini%20%7C%20OpenAI-informational)

CERAS is a solver-grounded, multi-verifier AI tutoring framework that combines:
- explicit reasoning (Tree-of-Thoughts + verifier passes),
- cognitive efficiency estimation (CEPM + CNN + fusion),
- and adaptive pedagogical feedback in real time.

The project is designed for research and demonstration of how structured reasoning plus behavioral-cognitive signals can improve learning-oriented AI responses compared to a plain single-pass chatbot pipeline.

> [!CAUTION]
> **PROPRIETARY SOURCE CODE**  
> This repository is protected by a proprietary license. Unauthorized copying, modification, or distribution is strictly prohibited. See [LICENSE](LICENSE) for legal terms.

## Research Motivation

Most educational assistants optimize for correctness only. CERAS addresses a broader target:
- answer quality,
- reasoning transparency,
- and learner readiness alignment.

Instead of producing one-shot answers, CERAS decomposes tasks into strategies, verifies intermediate reasoning, estimates user cognitive readiness, and adjusts final feedback style and complexity.

## Core Contributions

- **Solver-grounded reasoning pipeline** using Tree-of-Thoughts with explicit search and scoring.
- **Dual-model cognitive scoring** via CEPM (structural) and CNN (semantic/intention) predictors.
- **Fusion-based readiness estimation** that outputs fused score, confidence, diagnostics, and readiness label.
- **Adaptive response generation** conditioned on cognitive score and diagnostics.
- **Operational full stack** (FastAPI + React/Vite) with live telemetry, typing analytics, and provider/model switching.

## System Architecture

```mermaid
graph TD
    U[User / Student] --> FE[React Dashboard]
    FE --> API[FastAPI Service]

    API --> TOT[Tree-of-Thoughts Engine]
    TOT --> VER[Multi-Verifier Reasoning]

    API --> FX[Feature Extraction]
    FX --> CEPM[CEPM Model (LightGBM)]
    FX --> CNN[CNN Model (Keras)]
    CEPM --> FUSION[CERAS Fusion Layer]
    CNN --> FUSION

    VER --> ADAPT[Adaptive Response Generator]
    FUSION --> ADAPT
    ADAPT --> FE
```

For a deeper architecture breakdown, see [final_architecture.md](final_architecture.md).

## Feature Highlights

- **Reasoning engine**: strategy decomposition, candidate generation, verifier checks, and best-path selection.
- **Multi-provider LLM support**: Groq, Gemini, OpenAI (separate main and verifier providers/models).
- **Cognitive analytics**: CEPM score, CNN score, fused score, confidence, readiness, strengths, suggestions.
- **Live prompt instrumentation**: prompt quality metrics, typing dynamics, formulation-time capture.
- **Adaptive tutoring**: personalized follow-up responses and generated learning plans.
- **Research-friendly observability**: in-memory API logs and session diagnostics.

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React 19, Vite, Vanilla CSS |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Reasoning/Orchestration | Python pipeline modules, LangChain integrations |
| ML Models | LightGBM, TensorFlow/Keras, scikit-learn |
| Data/Utilities | NumPy, Pandas |
| Optional Persistence | Supabase |

## Repository Structure

```text
.
├── server.py                   # FastAPI entrypoint and API orchestration
├── API_DOCUMENTATION.md        # Full endpoint-level API details
├── final_architecture.md       # Detailed architectural discussion
├── frontend/                   # React + Vite application
│   ├── src/
│   │   ├── api.js              # Frontend API client
│   │   ├── components/         # UI components (dashboard, analytics, etc.)
│   │   ├── hooks/              # Typing analytics/auth hooks
│   │   └── lib/                # Supabase/session helper utilities
├── src/ceras/                  # Core reasoning and ML inference modules
│   ├── pipeline_1.py
│   ├── inference.py
│   ├── fusion.py
│   ├── llm_utils.py
│   └── tree_of_thoughts.py
├── artifacts/                  # Trained model/scaler artifacts used at runtime
├── models/                     # Training-side model scripts
├── data/                       # Datasets and preprocessing assets
├── notebooks/                  # Experiment notebooks
└── tests/                      # Test suite
```

## Quick Start

### 1) Prerequisites

- Python 3.10+
- Node.js 18+
- At least one provider API key: Groq, Gemini, or OpenAI

### 2) Backend setup

```bash
pip install -r requirements.txt
python server.py
```

The backend runs on `http://localhost:8000`.  
Model artifacts are loaded in a background thread on startup.

### 3) Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173`.

### 4) Configure providers

Enter API keys in the sidebar and validate connectivity from the UI.  
You can choose separate main and verifier providers/models per session.

## Environment Variables

### Frontend (optional)

Create `frontend/.env` if needed:

```env
VITE_API_BASE=/api
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

Notes:
- `VITE_API_BASE` defaults to `/api`.
- Supabase variables are only required for authentication/session persistence flows.

## API Overview

Base URL: `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service and model loading status |
| `GET` | `/api/logo` | Project logo asset |
| `GET` | `/api/logs` | Recent in-memory backend logs |
| `POST` | `/api/check-connection` | Validate provider API keys |
| `POST` | `/api/run-session` | Full reasoning + CE scoring pipeline |
| `POST` | `/api/adaptive-response` | Personalized tutoring summary |
| `POST` | `/api/parse-file` | Parse PDF/DOCX/CSV/TXT/MD into text |
| `POST` | `/api/followup` | Socratic follow-up conversation endpoint |
| `POST` | `/api/generate-plan` | Generate adaptive learning plan |

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for request/response schemas and examples.

## Reproducibility and Evaluation Workflow

- Use `tests/` for automated checks (`pytest`).
- Use `notebooks/` for exploratory analysis and model experiments.
- Keep inference artifacts under `artifacts/` to ensure consistent runtime behavior.
- Capture session outputs (scores, diagnostics, logs, generated steps) for qualitative and quantitative reporting.

## Known Limitations

- Current release is a research prototype, not a production clinical/educational decision system.
- Runtime quality and cost vary by selected LLM provider and model.
- Cognitive efficiency metrics are model-based estimates and should be interpreted as assistive signals, not absolute learner labels.

## Publication and Citation

If you reference CERAS in an academic submission, cite your paper and link this repository as the implementation artifact.

Suggested BibTeX template:

```bibtex
@misc{ceras2026,
  title        = {CERAS: Cognitive Efficiency and Reasoning Alignment System},
  author       = {Wolfie8935 and Rishaan08},
  year         = {2026},
  note         = {Research prototype and implementation repository},
  howpublished = {\url{https://github.com/Wolfie8935/CERAS-Cognitive-Efficiency-Reasoning-Alignment-System}}
}
```

## License

Copyright (c) 2026 Wolfie8935 and Rishaan08. All rights reserved.

This software is provided for reference only. You may view the code, but you may not use, copy, modify, merge, publish, or distribute it without explicit written permission. See [LICENSE](LICENSE).
