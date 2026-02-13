# 🧠 CAMRE EDU: Project Overview

**CAMRE EDU** (Cognitive Adaptive Multi-Modal Reasoning Engine for Education) is an intelligent learning lab that combines **Generative AI (GenAI)** with **Machine Learning (ML)** to provide a personalized, cognitively aware learning experience.

## 🏗️ Technical Architecture

### 1. Generative AI (The Reasoning Engine)
The core reasoning capability is powered by **Groq** (using Llama 3 models), orchestrated via **LangChain**.
-   **Model**: `llama-3.3-70b-versatile` (Primary), falling back to `llama-3.1-8b-instant`.
-   **Methodology**:
    -   **Tree of Thoughts (ToT)** / **Decomposition**: The system does not just "guess" the answer. It breaks complex problems into pedagogical sub-problems using a Dynamic Programming approach.
    -   **Verification**: Each sub-step is verified by a secondary model call to ensure accuracy before proceeding.
    -   **Adaptive Synthesis**: The final response is not static. A "Teacher LM" synthesizes the solution based on the user's computed Cognitive Efficiency score (see below).

### 2. Machine Learning (The Cognitive Engine)
We use a **Multi-Modal Feature Fusion** approach to assess the user's cognitive state in real-time.
*Note: Currently simulated in `streamlit_app.py`, but designed for real sensor integration.*

#### **Live Data Tracking (Inputs)**
We track the following raw signals from the user interaction:
1.  **Interaction Latency**: Time taken to formulate and submit the query.
2.  **Input Volume/Complexity**: Character and token counts of the prompt.
3.  **Gaze/Visual Attention** (Simulated): "Active/Distracted" status.
4.  **Behavioral Patterns**: Typing speed and correction rate (implied).

#### **ML Models & Signals**
These raw inputs are fed into three specialized modules:
1.  **CEPM (Cognitive Efficiency Prediction Model)**:
    -   *Type*: Behavioral ML Model.
    -   *Function*: Analyzes interaction patterns (latency vs. complexity) to estimate cognitive load.
2.  **CNN (Convolutional Neural Network)**:
    -   *Type*: Computer Vision Model.
    -   *Function*: (Simulated) Analyzes facial landmarks to detect confusion, focus, or fatigue.
3.  **ANFIS (Adaptive Network-Based Fuzzy Inference System)**:
    -   *Type*: Neuro-Fuzzy System.
    -   *Function*: Maps non-linear inputs (like variable latency) to precise cognitive states.

### 3. Fusion & Scoring (Dempster-Shafer Theory)
The outputs from the three ML models are **fused** into a single metric:
-   **Fused CE Score (0.0 - 1.0)**: Represents the user's overall "Cognitive Efficiency".
    -   **< 0.4 (Low)**: High Load/Confusion. -> Triggers *Supportive/Detailed* AI response.
    -   **0.4 - 0.7 (Moderate)**: Active Learning. -> Triggers *Balanced* AI response.
    -   **> 0.8 (High)**: Flow/Mastery. -> Triggers *Advanced/Concise* AI response.

## 🚀 Workflow Summary
1.  **User Input** -> Captured & telemetrized (Latency, Complexity).
2.  **GenAI Pipeline** -> Decomposes problem -> Verifies steps -> Solves.
3.  **ML Pipeline** -> Calculates CEPM, CNN, ANFIS scores -> Fuses into CE Score.
4.  **Adaptive Response** -> LLM generates a final teaching summary matched to the CE Score.
