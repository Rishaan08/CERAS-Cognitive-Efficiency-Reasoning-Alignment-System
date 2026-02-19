import streamlit as st
import time
import json
from datetime import datetime
import numpy as np
import os
import sys
import joblib
from pathlib import Path
import tensorflow as tf
import re

#Page Configuration
base_dir = Path(__file__).resolve().parents[2]
logo_path = base_dir / "assets" / "ceras_logo.png"

st.set_page_config(
    page_title="CERAS",
    page_icon=str(logo_path),
    layout="wide",
)

artifact_dir = "./artifacts"

#Load trained model
@st.cache_resource
def load_models():
    cepm_model = joblib.load(os.path.join(artifact_dir, "cepm_lightgbm.pkl"))
    cepm_scaler = joblib.load(os.path.join(artifact_dir, "cepm_scaler.pkl"))

    cnn_model = tf.keras.models.load_model(os.path.join(artifact_dir, "cnn_ce_model.keras"))
    cnn_scaler = joblib.load(os.path.join(artifact_dir, "cnn_scaler.pkl"))
    # cnn_model = tf.keras.models.load_model(os.path.join(artifact_dir, "cnn_ce_model.keras"))
    # cnn_scaler = joblib.load(os.path.join(artifact_dir, "cnn_scaler.pkl"))

    # Load selected feature names
    cepm_features = np.load(os.path.join(artifact_dir, "cepm_features.npy"), allow_pickle=True).tolist()
    cnn_features = np.load(os.path.join(artifact_dir, "cnn_features.npy"), allow_pickle=True).tolist()

    return (
        cepm_model,
        cepm_scaler,
        cnn_model,
        cnn_scaler,
        cepm_features,
        cnn_features
    )

(
    cepm_model,
    cepm_scaler,
    cnn_model,
    cnn_scaler,
    cepm_features,
    cnn_features
) = load_models()

#Reasoning pipeline
from pipeline_1 import main as run_infer

#CERAS fusion engine
from fusion import CERASFusion

#Connection Check
from llm_utils import check_connection

#CSS Styles
st.markdown(
    """
    <style>
    .example-card {
        background: #1e293b; 
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .example-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.3);
    }
    .example-header {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .example-body {
        font-size: 0.95rem;
        color: #e2e8f0;
        line-height: 1.6;
        text-align: justify;
        flex-grow: 1;
        background: rgba(0, 0, 0, 0.2);
        padding: 12px;
        border-radius: 8px;
    }
    .prompt-box-bad {
        background-color: #450a0a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #7f1d1d;
        margin-bottom: 10px;
        color: #fca5a5;
        font-size: 0.9em;
        height: 100%;
    }
    .sidebar-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Session State Initialization
if "groq_status" not in st.session_state:
    st.session_state.groq_status = "Waiting"
if "gemini_status" not in st.session_state:
    st.session_state.gemini_status = "Waiting"

def perform_check(provider):
    key_key = f"input_{provider.lower()}_key"
    api_key = st.session_state.get(key_key)
    
    status_key = f"{provider.lower()}_status"
    st.session_state[status_key] = "Checking..."
    
    if check_connection(provider, api_key):
        st.session_state[status_key] = "Connected"
    else:
        st.session_state[status_key] = "Not Connected"

def check_groq():
    perform_check("Groq")

def check_gemini():
    perform_check("Gemini")

#Sidebar
with st.sidebar:

    if logo_path.exists():
        st.markdown(
            """
            <div style="display:flex;
                        flex-direction:column;
                        align-items:center;
                        text-align:center;
                        padding-top:30px;
                        padding-bottom:25px;">
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <img src="assets/ceras_logo.png"
                 width="260"
                 class="logo-glow" />
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
                <h2 style="margin-top:15px;">CERAS</h2>
                <p style="color:#94a3b8; font-size:14px;">
                    Cognitive Efficiency & Reasoning Alignment System
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🔑 API Configuration")
    
    with st.expander("API Keys", expanded=True):
        st.session_state.groq_api_key = st.text_input("Groq API Key", type="password", key="input_groq_key", on_change=check_groq)
        st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password", key="input_gemini_key", on_change=check_gemini)

    st.markdown("### 🤖 Model Selection")
    
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "groq/compound",
        "groq/compound-mini",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]
    
    gemini_models = [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-robotics-er-1.5-preview"
    ]
    
    # Main Reasoner
    c1, c2 = st.columns([0.4, 0.6])
    with c1:
        st.session_state.main_provider = st.radio("Main Provider", ["Groq", "Gemini"], index=0, key="radio_main_provider", horizontal=False, label_visibility="collapsed")
    with c2:
        if st.session_state.main_provider == "Groq":
            st.session_state.main_model = st.selectbox("Model", groq_models, index=0, key="select_main_model", label_visibility="collapsed")
        else:
            st.session_state.main_model = st.selectbox("Model", gemini_models, index=0, key="select_main_model", label_visibility="collapsed")

    # Verifier
    st.caption("Verifier Model")
    v1, v2 = st.columns([0.4, 0.6])
    with v1:
        st.session_state.verifier_provider = st.radio("Verifier Provider", ["Groq", "Gemini"], index=0, key="radio_verifier_provider", horizontal=False, label_visibility="collapsed")
    with v2:
        if st.session_state.verifier_provider == "Groq":
             # Default verifier to faster model
            st.session_state.verifier_model = st.selectbox("Verifier", groq_models, index=1, key="select_verifier_model", label_visibility="collapsed")
        else:
            st.session_state.verifier_model = st.selectbox("Verifier", gemini_models, index=0, key="select_verifier_model", label_visibility="collapsed")


    st.markdown("### ⚙️ System Status")

    g_status = st.session_state.groq_status
    g_color = "#4ade80" if g_status == "Connected" else ("#ef4444" if g_status == "Not Connected" else "#94a3b8")
    
    gem_status = st.session_state.gemini_status
    gem_color = "#4ade80" if gem_status == "Connected" else ("#ef4444" if gem_status == "Not Connected" else "#94a3b8")

    st.markdown(
        f"""
        <div class="sidebar-box">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>🔌 <b>Groq API</b></span>
                <span style="color:{g_color};">● {g_status}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>✨ <b>Gemini API</b></span>
                <span style="color:{gem_color};">● {gem_status}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>📡 <b>Telemetry</b></span>
                <span style="color:#60a5fa;">● Tracking</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🏗️ Architecture")

    st.markdown(
        """
        <div class="sidebar-box" style="font-size: 13px;">
            <p style="margin:8px 0;"> 🧩 <b>ToT-LLM</b><br><span style="color:#94a3b8;">Tree-of-Thoughts Reasoning Engine</span></p>
            <p style="margin:8px 0;"> 🧠 <b>CEPM</b><br><span style="color:#94a3b8;">Cognitive Engagement Modeling</span></p>
            <p style="margin:8px 0;"> 👁️ <b>CNN-Vis</b><br><span style="color:#94a3b8;">Behavioral Signal Analysis</span></p>
            <p style="margin:8px 0;"> 🔗 <b>Fusion Layer</b><br><span style="color:#94a3b8;">Multi-Modal Signal Integration</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("v1.2.0 • Neural Learning Stack")

#Session State
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if "formulation_time" not in st.session_state:
    st.session_state.formulation_time = 0.0

if "auto_run" not in st.session_state:
    st.session_state.auto_run = False

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""

def set_prompt(text):
    st.session_state.user_prompt = text
    # Disable auto-run, user wants to manually click run
    st.session_state.auto_run = False

#Header
if logo_path.exists():

    st.markdown(
        f"""
        <div style="display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    margin-top:80px;
                    margin-bottom:40px;">

            <img src="assets/ceras_logo.png"
                 width="300"
                 class="logo-glow"
                 style="margin-bottom:30px;" />

            <h1 style="font-size:52px;
                       letter-spacing:3px;
                       margin-bottom:10px;">
                CERAS
            </h1>

            <h3 style="color:#94a3b8;
                       font-weight:400;
                       font-size:22px;">
                Cognitive Efficiency Reasoning Alignment System
            </h3>

        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<div style="text-align: justify;">
<b>CERAS</b> is an advanced adaptive learning environment designed to optimize how you learn and solve complex problems. 
By fusing <b>Large Language Model (LLM)</b> reasoning capabilities with real-time <b>Cognitive Efficiency</b> metrics, current behavioral diagnostics, and 
neuro-fuzzy alignment, CERAS provides a personalized learning experience. It analyzes your input complexity, structure, and intent to guide you 
through deep concepts with tailored roadmaps, ensuring you don't just get answers, but truly master the material.
</div>
""", unsafe_allow_html=True)

with st.expander("🎓 Guide: How to Write the Perfect Prompt", expanded=False):
    st.markdown(
        """
        ### 🔑 Key Principles of Cognitive Efficiency
        To get the best results from CERAS (and any LLM), focus on these core elements:

        1.  **Context & Role**: Define *who* the model is (e.g., "Act as a senior physicist") and *what* the situation is.
        2.  **Explicit Constraints**: Set boundaries. Mention word counts, specific formats (JSON, Markdown), or stylistic requirements.
        3.  **Chain of Thought**: Ask the model to "explain its reasoning" or "break down the problem step-by-step" before giving the final answer.
        4.  **Few-Shot Examples**: Providing 1-2 examples of the desired output format is the single most effective way to guide behavior.
        5.  **Iterative Refinement**: Use the **Diagnostics** below to see where your prompt lacks density or clarity, then refine it.

        *Tip: Use the "Good Examples" below to see these principles in action!*
        """
    )

#Good Examples
st.markdown("### 🌟 GOOD EXAMPLES TO PROMPT (High CE Score)")
st.caption("These prompts are detailed and structured, leading to higher cognitive efficiency scores.")

gp1 = "Analyze the epistemological foundations of quantum entanglement by integrating formal mathematical structure, experimental validation, and philosophical interpretation into a coherent explanatory framework. Begin by describing how tensor product Hilbert spaces allow composite quantum systems to exhibit non-factorizable state vectors, and clarify why separability fails under entangled configurations. Then examine Bell’s inequalities, including the CHSH formulation, and explain how empirical violations observed in Aspect-type experiments undermine classical locality and deterministic realism. Extend the discussion toward decoherence theory, entropic correlations, and the role of measurement operators in collapsing superposed amplitudes. Contrast Copenhagen, Many-Worlds, and relational interpretations, focusing specifically on their ontological commitments and metaphysical implications. Additionally, evaluate how quantum information theory reframes entanglement as a computational resource enabling teleportation, superdense coding, and cryptographic security. Finally, synthesize these perspectives into a structured argument addressing whether entanglement necessitates nonlocal causation or instead demands a revision of classical intuitions regarding separability, causality, and physical realism."

gp2 = "Construct a systems-level biochemical and thermodynamic analysis of photosynthesis that integrates molecular structure, energetic transfer mechanisms, and ecological macro-dynamics. Begin by formally describing chloroplast ultrastructure and pigment absorption spectra in terms of quantum excitation states. Then analyze the light-dependent reactions as an electron transport optimization problem, including photolysis, proton gradients, chemiosmotic coupling, and ATP synthase rotation mechanics. Extend the discussion into the Calvin-Benson cycle using carbon fixation kinetics, RuBisCO efficiency constraints, and NADPH reduction pathways. Evaluate photosynthesis as an entropy-management system that converts low-entropy solar radiation into high-order biochemical organization. Finally, synthesize its planetary-scale implications for atmospheric regulation, carbon sequestration feedback loops, and biospheric energy flow stability."

gp3 = "Develop a multi-layered historical and epistemological examination of the Gutenberg printing press by integrating technological innovation theory, sociopolitical restructuring, and cognitive-cultural transformation. Begin by describing the mechanical engineering principles underlying movable type standardization and ink transfer reproducibility. Then analyze how mass replication altered information diffusion velocity and network topology across Renaissance Europe. Evaluate its causal role in accelerating scientific method formalization, destabilizing ecclesiastical epistemic monopolies, and enabling vernacular linguistic codification. Extend the analysis toward media ecology theory and distributed cognition, examining how print culture reshaped memory externalization and authority structures. Conclude by synthesizing how the printing press functioned as an epistemic amplifier that reconfigured knowledge production, institutional legitimacy, and political sovereignty."

gp4 = "Produce a mathematically grounded and architecturally comparative analysis of supervised and unsupervised machine learning paradigms, emphasizing objective functions, representational geometry, and statistical inference principles. Begin by defining supervised learning as an empirical risk minimization framework over labeled distributions and contrast it with unsupervised latent-variable modeling and manifold estimation. Analyze bias-variance trade-offs, generalization bounds, and overfitting dynamics under distributional shift. Compare algorithmic mechanisms such as Support Vector Machines, ensemble-based decision forests, K-Means clustering, and Principal Component Analysis through the lens of optimization landscapes and feature-space transformations. Extend the discussion toward interpretability constraints, scalability limits, and robustness under adversarial perturbations. Finally, synthesize these paradigms into a structured framework evaluating when hybrid semi-supervised or self-supervised approaches become epistemically advantageous."

#Layout: 2 Columns x 2 Rows with Custom Cards
c1, c2 = st.columns(2)

def render_card(title, text, prompt_var, btn_key, gradient):
    st.markdown(
        f"""
        <div class="example-card" style="background: {gradient};">
            <div class="example-header">
                <span>✨</span> {title}
            </div>
            <div class="example-body">
                {text[:280]}...
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Spacer
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button(f"Select: {title}", key=btn_key, use_container_width=True):
        set_prompt(prompt_var)
        st.rerun()

with c1:
    render_card(
        "Quantum Foundations", 
        gp1, gp1, "btn_gp1", 
        "linear-gradient(135deg, #4c1d95 0%, #1e1b4b 100%)" # Deep Purple
    )
    render_card(
        "Printing Press Analysis", 
        gp3, gp3, "btn_gp3", 
        "linear-gradient(135deg, #9f1239 0%, #4c0519 100%)" # Rose/Red
    )

with c2:
    render_card(
        "Photosynthesis Systems", 
        gp2, gp2, "btn_gp2", 
        "linear-gradient(135deg, #065f46 0%, #064e3b 100%)" # Emerald/Green
    )
    render_card(
        "ML Paradigm Theory", 
        gp4, gp4, "btn_gp4", 
        "linear-gradient(135deg, #1e40af 0%, #172554 100%)" # Blue
    )

if st.session_state.auto_run:
    pass

#Bad Examples
st.markdown("### ⚠️ BAD EXAMPLES TO PROMPT (Low CE Score)")
st.caption("These prompts look normal but lack depth, structure, or analytical clarity.")

bp1 = "Explain artificial intelligence in simple terms."
bp2 = "Describe how computers work."
bp3 = "Give a summary of World War II."
bp4 = "Explain why the sky is blue in a short answer."

bc1, bc2, bc3, bc4 = st.columns(4)

with bc1:
    st.markdown(f'<div class="prompt-box-bad">{bp1}</div>', unsafe_allow_html=True)
    if st.button("Use: AI Basic", key="btn_bp1"):
        set_prompt(bp1)
        st.rerun()

with bc2:
    st.markdown(f'<div class="prompt-box-bad">{bp2}</div>', unsafe_allow_html=True)
    if st.button("Use: Computers", key="btn_bp2"):
        set_prompt(bp2)
        st.rerun()

with bc3:
    st.markdown(f'<div class="prompt-box-bad">{bp3}</div>', unsafe_allow_html=True)
    if st.button("Use: WWII Summary", key="btn_bp3"):
        set_prompt(bp3)
        st.rerun()

with bc4:
    st.markdown(f'<div class="prompt-box-bad">{bp4}</div>', unsafe_allow_html=True)
    if st.button("Use: Sky Simple", key="btn_bp4"):
        set_prompt(bp4)
        st.rerun()

st.markdown("---")

#Input
c_in1, c_in2 = st.columns([0.8, 0.2])
with c_in2:
    if st.button("New Problem"):
        st.session_state.start_time = time.time()
        st.session_state.formulation_time = 0.0
        st.session_state.user_prompt = ""
        st.session_state.auto_run = False
        st.rerun()

prompt = st.text_area(
    "Enter your learning question or problem",
    height=150,
    help="Time starts tracking when you click 'New Problem' or reload.",
    key="user_prompt"
)

run_btn = st.button("▶ Run Learning Session")

#Run Pipeline
if (run_btn or st.session_state.auto_run) and prompt.strip():
    
    #Reset auto_run so it doesn't loop
    if st.session_state.auto_run:
        st.session_state.auto_run = False
    
    #Calculate User Latency (Formulation Time)
    st.session_state.formulation_time = time.time() - st.session_state.start_time
    
    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        # Collect API Config
        api_config = {
            "main_provider": st.session_state.get("main_provider", "Groq"),
            "verifier_provider": st.session_state.get("verifier_provider", "Groq"),
            "groq_api_key": st.session_state.get("groq_api_key", ""),
            "gemini_api_key": st.session_state.get("gemini_api_key", ""),
            "main_model": st.session_state.get("main_model"),
            "verifier_model": st.session_state.get("verifier_model"),
        }
        result = run_infer(prompt, api_config=api_config)
        runtime = time.time() - t0  # System Latency

def extract_ceras_features(prompt_text):

    words = prompt_text.split()
    prompt_length = np.clip(len(words), 1, 400)
    character_count = len(prompt_text)

    sentence_count = max(len(re.findall(r"[.!?]", prompt_text)), 1)

    unique_word_ratio = len(set(words)) / (prompt_length + 1e-6)
    unique_word_ratio = np.clip(unique_word_ratio, 0, 1)

    concept_density = sum(1 for w in words if len(w) > 6) / (prompt_length + 1e-6)
    concept_density = np.clip(concept_density, 0, 1)

    keystrokes = np.clip(character_count, 1, 2000)

    prompt_quality = np.clip(prompt_length / 150, 0, 1)

    #Simple live prompt_type mapping
    if prompt_length < 20:
        prompt_type = 0
    elif prompt_length < 60:
        prompt_type = 1
    elif prompt_length < 120:
        prompt_type = 2
    else:
        prompt_type = 3

    return {
        "prompt_length": float(prompt_length),
        "sentence_count": float(sentence_count),
        "unique_word_ratio": float(unique_word_ratio),
        "concept_density": float(concept_density),
        "prompt_quality": float(prompt_quality),
        "character_count": float(character_count),
        "keystrokes": float(keystrokes),
        "prompt_type": float(prompt_type),
    }

#Run Pipeline
should_run = False
if run_btn:
    should_run = True
if st.session_state.auto_run:
    should_run = True
    st.session_state.auto_run = False

#Collect API Config (Always available)
api_config = {
    "main_provider": st.session_state.get("main_provider", "Groq"),
    "verifier_provider": st.session_state.get("verifier_provider", "Groq"),
    "groq_api_key": st.session_state.get("groq_api_key", ""),
    "gemini_api_key": st.session_state.get("gemini_api_key", ""),
    "main_model": st.session_state.get("main_model"),
    "verifier_model": st.session_state.get("verifier_model"),
}

if should_run and prompt.strip():
    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        # Calculate User Latency (Formulation Time)
        st.session_state.formulation_time = time.time() - st.session_state.start_time
        
        result = run_infer(prompt, api_config=api_config)
        runtime = time.time() - t0
        
        # Store results in session state for persistence
        st.session_state.current_result = result
        st.session_state.current_runtime = runtime
        st.session_state.current_prompt = prompt

    #Render Results IMMEDIATELY
    result = st.session_state.current_result
    result_prompt = st.session_state.current_prompt
    st.markdown("## Learning Response")

    final_steps = result.get("final_answer", [])
    if isinstance(final_steps, list):
        for i, step in enumerate(final_steps, 1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.write(final_steps)
    
    #Generate and store adaptive response
    from llm_utils import generate_adaptive_response

    #Extract Real Features
    final_steps = result.get("final_answer", [])

    features = extract_ceras_features(result_prompt)

    #CEPM Inference
    cepm_input = np.array([features[f] for f in cepm_features]).reshape(1, -1)
    cepm_input_scaled = cepm_scaler.transform(cepm_input)
    cepm_score = float(np.clip(cepm_model.predict(cepm_input_scaled)[0], 0, 1))

    #CNN Inference
    cnn_input = np.array([features[f] for f in cnn_features]).reshape(1, -1)
    cnn_input = cnn_scaler.transform(cnn_input)

    if len(cnn_model.input_shape) == 3:
        cnn_input = cnn_input.reshape(cnn_input.shape[0], cnn_input.shape[1], 1)

    cnn_score = float(np.clip(np.squeeze(cnn_model.predict(cnn_input, verbose=0)), 0, 1))

    #Fusion
    fusion_engine = CERASFusion()

    fusion_df = fusion_engine.fuse(
        session_ids=["session_1"],
        cepm_scores=[cepm_score],
        cnn_scores=[cnn_score]
    )

    fused_score = fusion_df["fused_ce_score"].iloc[0]
    confidence = fusion_df["confidence"].iloc[0]
    diagnostics = fusion_df["diagnostics"].iloc[0]
    readiness = fusion_df["readiness_label"].iloc[0]

    feature_count = len(features)
    est_prompt_tokens = int(len(prompt) / 4)
    est_response_tokens = int(len(str(final_steps)) / 4)
    total_tokens = est_prompt_tokens + est_response_tokens
    
    # --- DIAGNOSTICS LOGIC ---
    strengths = []
    suggestions = []

    if cepm_score > 0.75:
        strengths.append("Strong structural complexity and adequate length.")
    else:
        suggestions.append("Try adding more specific constraints or context to increase structural density.")

    if cnn_score > 0.75:
        strengths.append("High semantic clarity; intent matches known high-performing patterns.")
    else:
        suggestions.append("Clarify the core intent. Use precise domain terminology to improve semantic alignment.")

    if not strengths:
        strengths.append("Prompt is functional but has room for optimization across all dimensions.")
    if not suggestions:
        suggestions.append("Excellent prompt! Maintains high cognitive efficiency.")


    
    #Session Metrics
    st.markdown("### ⏱️ Session Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Formulation Time", f"{st.session_state.formulation_time:.2f}s")
    with m2:
        st.metric("Processing Time", f"{runtime:.2f}s")
    with m3:
        st.metric("Est. Tokens", f"{total_tokens}")
    with m4:
        st.metric("Features Extracted", f"{feature_count}")

    #Diagnostic Report
    with st.expander("Cognitive Diagnostic Report", expanded=True):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Strengths**")
            for s in strengths:
                st.markdown(f"- {s}")
        with d2:
            st.markdown("**Suggestions for Improvement**")
            for s in suggestions:
                st.markdown(f"- {s}")

    # --- CE DISPLAY ---
    st.markdown("###Cognitive Efficiency Analysis")
    
    # Main Score Dashboard
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Fused CE Score", f"{fused_score:.2f}", delta="Target: > 0.7")
    with k2:
        st.metric("Structural (CEPM)", f"{cepm_score:.2f}")
    with k3:
        st.metric("Semantic (CNN)", f"{cnn_score:.2f}")
    with k4:
        # Color code the readiness
        r_color = "green" if "High" in readiness else ("orange" if "Medium" in readiness else "red")
        st.markdown(f"**Readiness:** :{r_color}[{readiness}]")
        st.caption(f"Confidence: {confidence:.2f}")

    # Explanation Fused CE Score
    with st.expander("What is the Fused CE Score?"):
        st.markdown("""
            ### Fused Cognitive Efficiency (CE) Score

            The **Fused CE Score** reflects how efficiently you are learning in this session.

            It combines two independent signals:

            • **Conceptual Strength (CEPM)** – Depth of understanding  
            • **Behavioral & Reasoning Alignment (CNN)** – Interaction patterns, engagement consistency, and structural reasoning signals 

            These are fused into a single score between **0 and 1**.

            ### What Your Level Means

            **0.00 – 0.44 → Foundation Building**  
            You may need to revisit core concepts and slow down. Strengthen fundamentals before moving forward.

            **0.45 – 0.59 → Developing Momentum**  
            You're engaging and learning, but some inconsistencies exist. Refining strategy will help.

            **0.60 – 0.74 → Progressing Confidently**  
            You demonstrate stable understanding and good engagement. Keep challenging yourself.

            **0.75 – 1.00 → Peak Learning State**  
            You are operating with strong clarity, alignment, and efficiency. Ready for advanced challenges.

            This score reflects learning efficiency — not intelligence — and adapts to your behavior in real time.
            """)

    # Live Telemetry
    with st.expander("📡 Live Telemetry & Diagnostics"):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Extracted Features**")
            st.json(features)
        with t2:
            st.markdown("**System Diagnostics**")
            st.write(diagnostics)

    #Generate Adaptive Response
    if "adaptive_res" not in st.session_state or st.session_state.current_prompt != prompt:
        try:
            with st.spinner("Generating personalized learning summary..."):
                st.session_state.adaptive_res = generate_adaptive_response(
                    prompt,
                    final_steps,
                    fused_score,
                    diagnostics,
                    api_config=api_config
                )
        except Exception as e:
            st.warning(f"Adaptive response unavailable: {e}")
            st.session_state.adaptive_res = None
    
    #Adaptive Learning Response
    st.markdown("## Adaptive Learning Response")
    
    if st.session_state.adaptive_res:
        st.markdown(st.session_state.adaptive_res)
    else:
        st.info("Adaptive response unavailable.")

    #Trace
    with st.expander("Reasoning Trace"):
        st.caption("Detailed logs of the decomposition and verification process.")
        logs = result.get("logs", "")
        st.code(logs)

    #Download Report
    report_text = f"""CERAS Session Report
    
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Session ID: session_1

    Prompt: {prompt}

    Metrics:
    - Formulation Time: {st.session_state.formulation_time:.2f}s
    - Processing Time: {runtime:.2f}s
    - Est. Tokens: {total_tokens}

    Scores:
    - Fused CE Score: {fused_score:.2f}
    - Structural (CEPM): {cepm_score:.2f}
    - Semantic (CNN): {cnn_score:.2f}
    - Readiness: {readiness} ({confidence:.2f} confidence)

    Diagnostics:
    Strengths: {chr(10).join(['- ' + s for s in strengths])}

    Suggestions: {chr(10).join(['- ' + s for s in suggestions])}

    Learning Response: {chr(10).join([f"{i}. {s}" for i, s in enumerate(final_steps, 1)]) if isinstance(final_steps, list) else str(final_steps)}
"""
    st.download_button(
        label="📥 Download Session Report",
        data=report_text,
        file_name=f"ceras_report_{int(time.time())}.txt",
        mime="text/plain"
    )

#Empty State
if not run_btn:
    st.info(
        "Enter a learning question above and run the session.\n\n"
        "CERAS will provide:\n"
        "• Step-by-step reasoning\n"
        "• Cognitive Efficiency score\n"
        "• Diagnostic feedback\n"
        "• Improvement suggestions"
    )

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:#9ca3af; margin-top:8px;">
        Developed by <b>© Aman Goel</b> & <b>Rishaan Yadav</b><br>
        <span style="color:#6b7280;">CERAS · CERAS Framework</span>
    </div>
    """,
    unsafe_allow_html=True
)