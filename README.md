# CERAS — Cognitive Efficiency & Reasoning Alignment System

> [!CAUTION]
> **PROPRIETARY SOURCE CODE**
> This repository is protected by a proprietary license. Unauthorized copying, modification, or distribution is strictly prohibited. See [LICENSE](LICENSE) for details.

![Status](https://img.shields.io/badge/Status-Active_Prototype-success)
![LLM Backend](https://img.shields.io/badge/LLM-Groq_Llama_3-blueviolet)
![Framework](https://img.shields.io/badge/Framework-LangChain_0.3-green)

**CERAS** is a **Solver-Grounded, Multi-Verifier AI Tutor** that measures *cognitive learning readiness* in real-time. It goes beyond simple chatbots by "thinking" before it answers—using a **Tree-of-Thoughts (ToT)** architecture to decompose problems, verify logic, and adapt its teaching style based on the student's cognitive load.

---

## 🏗️ System Architecture

The core of CERAS is a **System 2 Reasoning Engine** that separates *planning* (decomposition) from *execution* (solving) and *verification*.

```mermaid
graph TD
    User[Student] -->|Query| Streamlit[Streamlit Dashboard]

    subgraph Reasoning_Engine_Online ["🧠 Solver-Grounded Engine (Groq)"]
        Streamlit -->|Input| Pipeline
        
        Pipeline -->|1. Decompose| Decomposer[Llama 3.3-70b\nStrategy Proposer]
        Decomposer -->|Strategy| ToT[Tree of Thoughts\nData Structure]
        
        ToT -->|2. Solve| Solver[Llama 3.3-70b\nStep Generator]
        Solver -->|Draft Steps| Verifier1

        subgraph Verification_Loop ["⚡ Multi-Stage Verification"]
            Verifier1[Gatekeeper\nLlama 3.1-8b] -- Rejected --> Solver
            Verifier1 -- Approved --> Verifier2[Quality Audit\nLlama 3.1-8b]
        end

        Verifier2 -->|Verified Steps| ToT
    end

    subgraph Cognitive_Layer ["👁️ Cognitive Efficiency (CE)"]
        Streamlit -->|Behavioral Signals| CEPM[CEPM Model\n(Time/Focus/Clicks)]
        CEPM -->|CE Score (0-1)| Adaptive[Adaptive Response\nTheory of Mind]
        
        Verification_Loop -.->|Logic Quality| Adaptive
        Adaptive -->|Personalized Output| Streamlit
    end
```

---

## 🚀 Key Features

### 1. **Solver-Grounded Reasoning**
Unlike standard LLMs that hallucinate, CERAS uses a rigid **Reasoning Pipeline**:
1.  **Decomposition**: Breaks complex queries (e.g., "Explain Quantum Entanglement") into atomic, pedagogical sub-problems.
2.  **Tree Search**: Explores multiple reasoning paths using Depth-First Search (DFS) on a custom Tree Data Structure.
3.  **Strict Verification**: Every step is double-checked by a specialized "Verifier" model before being shown to the user.

### 2. **Real-Time Cognitive Diagnostics**
The system doesn't just grade *correctness*; it grades *efficiency*.
-   **CE Score (Cognitive Efficiency)**: A 0-1 metrics combining behavioral speed, focus, and logical consistency.
-   **Intention Clustering**: Uses 1D-CNNs to detect browsing patterns (e.g., "Rushing", "Struggling", "Flow State").

### 3. **Adaptive "Theory of Mind"**
The AI adjusts its personality based on the student's state:
-   **Low CE (< 0.5)**: *Supportive & Detailed*. Breaks things down further.
-   **High CE (> 0.8)**: *Challenging & Concise*. Pushes for mastery.

### 4. **Interactive Learning Dashboard**
A modern Streamlit UI providing:
-   **Example Prompts**: "Good" vs "Bad" examples to train students on effective questioning.
-   **Live Telemetry**: Real-time visualization of Formulation Time, System Latency, and Cognitive Load.
-   **Session Reports**: Downloadable JSON summaries of the learning session.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **LLM Backend** | **Groq API** | Ultra-low latency inference (Llama 3.3 70b, Llama 3.1 8b) |
| **Orchestration** | **LangChain** | Chain management and prompt engineering |
| **Interface** | **Streamlit** | Interactive web dashboard for students |
| **ML Models** | **LightGBM / PyTorch** | Cognitive Efficiency Prediction (CEPM) & CNN Feature Extraction |
| **Data Logic** | **Python (NetworkX)** | Tree-of-Thoughts graph management |
| **Fuzzy Logic** | **Custom ANFIS** | Neuro-Fuzzy alignment of reasoning scores |

---

## 💻 Setup & Installation

### Prerequisites
-   Python 3.12+
-   Conda
-   **Groq API Key** (Required for reasoning engine)

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Wolfie8935/CERAS-Cognitive-Efficiency-Reasoning-Alignment-System.git
    cd ceras
    ```

2.  **Create Environment**:
    ```bash
    conda env create -f environment.yml
    conda activate ceras
    ```

3.  **Configure Credentials**:
    Create a `.env` file in the root directory:
    ```bash
    GROQ_API_KEY=gsk_your_key_here
    ```

4.  **Run the Application**:
    Navigate to the source directory and launch Streamlit:
    ```bash
    cd src/ceras
    streamlit run streamlit_app.py
    ```

---

## 📂 Project Structure

```text
ceras/
├── artifacts/              # Model weights (CEPM, CNN, ANFIS)
├── src/ceras/
│   ├── interface/          # UI Logic
│   │   └── streamlit_app.py  # Main Dashboard
│   ├── reasoning/          # AI Reasoning Layer
│   │   ├── llm_utils.py      # Groq Interface & Prompts
│   │   ├── decomposer.py     # Strategy Generator
│   │   └── verifier.py       # Logic Gatekeeper
│   ├── models/             # ML Models
│   │   └── cepm.py           # Cognitive Efficiency Predictor
│   └── pipeline_1.py       # Main Orchestrator
├── environment.yml         # Dependencies
└── LICENSE                 # Proprietary License
```

---

## 📜 License

**Copyright (c) 2026 Wolfie8935. All Rights Reserved.**

This software is provided for reference only. You may view the code, but you may not use, copy, modify, merge, publish, or distribute it without explicit written permission. See [LICENSE](LICENSE) for full text.
