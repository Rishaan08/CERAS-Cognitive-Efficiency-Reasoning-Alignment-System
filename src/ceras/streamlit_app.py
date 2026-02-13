import streamlit as st
import time
import json
from datetime import datetime
import numpy as np

# --- Reasoning pipeline ---
from pipeline_1 import main as run_infer

# --- CERAS fusion engine ---
from fusion import CERASFusion

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="CAMRE EDU",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("## 🧠 CAMRE EDU - Intelligent Learning Lab")
st.caption("LLM Reasoning + Cognitive Efficiency + Behavioral Diagnostics")


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ⚙️ Run Configuration")
    show_trace = st.checkbox("Show Reasoning Trace", value=True)
    show_tree = st.checkbox("Show Tree JSON", value=False)
    
    st.markdown("---")
    st.markdown("### 🖥️ System Status")
    st.success("🟢 Groq API: Connected")
    st.success("🟢 Fusion Engine: Online")
    st.info("🔵 Telemetry: Active")


# ===================== SESSION STATE =====================
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if "formulation_time" not in st.session_state:
    st.session_state.formulation_time = 0.0

# ===================== INPUT =====================
c_in1, c_in2 = st.columns([0.8, 0.2])
with c_in2:
    if st.button("🔄 New Problem"):
        st.session_state.start_time = time.time()
        st.session_state.formulation_time = 0.0
        # We can't clear text_area directly without session_state binding, 
        # so we rely on user manually clearing or overwrite if we bound it.
        # But for now, just resetting the time is the key action.
        st.rerun()

prompt = st.text_area(
    "Enter your learning question or problem",
    height=150,
    help="Time starts tracking when you click 'New Problem' or reload."
)

run_btn = st.button("▶ Run Learning Session")


# ===================== RUN PIPELINE =====================
if run_btn and prompt.strip():
    
    # Calculate User Latency (Formulation Time)
    st.session_state.formulation_time = time.time() - st.session_state.start_time
    
    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        result = run_infer(prompt)
        runtime = time.time() - t0  # System Latency
def extract_ceras_features(prompt_text, llm_result):
    """
    Temporary simulation of CEPM / CNN / ANFIS signals.
    Replace this with real feature extraction + model inference later.
    """

    length = len(prompt_text)
    complexity = min(len(prompt_text.split()) / 50, 1.0)

    cepm_score = np.clip(0.4 + complexity * 0.4, 0, 1)
    cnn_score = np.clip(0.5 + complexity * 0.3, 0, 1)
    anfis_score = np.clip(0.45 + complexity * 0.35, 0, 1)

    return cepm_score, cnn_score, anfis_score


# ===================== RUN PIPELINE =====================
if run_btn and prompt.strip():

    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        result = run_infer(prompt)
        runtime = time.time() - t0

    # ===================== FINAL ANSWER =====================
    st.markdown("## ✅ Learning Response")

    final_steps = result.get("final_answer", [])

    if isinstance(final_steps, list):
        for i, step in enumerate(final_steps, 1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.write(final_steps)

    # ===================== RAW SENSOR DATA =====================
    st.markdown("---")
    st.markdown("### 📡 Live User Telemetry (Raw Inputs)")
    st.caption("Data captured from user interaction *before* feature extraction.")

    r1, r2, r3, r4 = st.columns(4)
    
    with r1:
        st.metric("⏱️ Formulation Time", f"{st.session_state.formulation_time:.2f}s", help="Time taken to type/submit")
    with r2:
        st.metric("⚡ System Latency", f"{runtime:.3f}s", help="AI Processing Time")
    with r3:
        st.metric("📝 Input Volume", f"{len(prompt)} chars")
    with r4:
        # Simulated "Live" status for external sensors
        st.metric("👁️ Gaze Tracker", "Active", delta="Tracking", delta_color="normal")

    # ===================== CERAS ANALYSIS =====================
    st.markdown("---")
    st.markdown("Cognitive Efficiency Analysis")
    
    # Extract simulated signals
    cepm_score, cnn_score, anfis_score = extract_ceras_features(prompt, result)

    fusion_engine = CERASFusion()

    fusion_df = fusion_engine.fuse(
        student_ids=[1],
        cepm_scores=[cepm_score],
        cnn_scores=[cnn_score],
        anfis_scores=[anfis_score]
    )

    fused_score = fusion_df["fused_ce_score"].iloc[0]
    confidence = fusion_df["confidence"].iloc[0]
    diagnostics = fusion_df["diagnostics"].iloc[0]
    readiness = fusion_df["readiness_label"].iloc[0]

    # (Metrics moved to visualization sections)

    # ===================== ADAPTIVE LEARNING RESPONSE =====================
    st.markdown("## 🎓 Adaptive Learning Response")
    from llm_utils import generate_adaptive_response  # lazy import
    
    with st.spinner("Generating personalized learning summary..."):
        adaptive_res = generate_adaptive_response(prompt, final_steps, fused_score, diagnostics)
        st.markdown(adaptive_res)

    # ===================== LIVE DATA VISUALIZATION =====================
    st.markdown("### 📊 Live Cognitive Signals")
    
    # Create a visual dashboard for the signals
    sig_col1, sig_col2, sig_col3 = st.columns(3)
    
    with sig_col1:
        st.markdown("**CEPM (Behavioral)**")
        st.progress(float(cepm_score), text=f"Load: {cepm_score:.2f}")
        st.caption("Derived from interaction cadence")
        
    with sig_col2:
        st.markdown("**CNN (Visual)**")
        st.progress(float(cnn_score), text=f"Focus: {cnn_score:.2f}")
        st.caption("Facial attention estimation")
        
    with sig_col3:
        st.markdown("**ANFIS (Neuro-Fuzzy)**")
        st.progress(float(anfis_score), text=f"State: {anfis_score:.2f}")
        st.caption("Non-linear state mapping")
    
    st.markdown("---")
    
    # Main Score Visualization
    m1, m2 = st.columns([1, 2])
    
    with m1:
        st.metric("🧠 Fused CE Score", f"{fused_score:.2f}", delta="Real-time")
        if fused_score > 0.7:
            st.success("State: High Efficiency (Flow)")
        elif fused_score > 0.4:
            st.warning("State: Moderate Load")
        else:
            st.error("State: High Cognitive Load")
            
    with m2:
        st.markdown("**Fusion Engine Confidence**")
        st.progress(float(confidence), text=f"Confidence: {confidence:.2f}")
        st.info(f"Readiness State: {readiness}")



    #RISHAAN CHANGEEEEE THISSSSSSSSS

    with st.expander("ℹ️ What is Fused CE Score?"):
        st.markdown("""
        **Fused Cognitive Efficiency (CE) Score** is a real-time metric derived from multiple sensors:
        - **0.0 - 0.4 (Low)**: Indicates high cognitive load, confusion, or lack of focus. Requires foundational support.
        - **0.4 - 0.7 (Moderate)**: Indicates active processing but some struggle or inconsistency.
        - **0.7 - 1.0 (High)**: Indicates flow state, mastery, and high efficiency.
        
        *Note: Validated by the CERAS Fusion Engine using Dempster-Shafer Theory.*
        """)





    # ===================== DIAGNOSTIC REPORT =====================
    st.markdown("##Cognitive Diagnostic Report")

    insights = []

    if diagnostics["concept_gap"]:
        insights.append(
            "Conceptual Gap Detected:  \n"
            "Your response suggests weak mastery of core principles required for this topic.  \n"
            "You may be attempting solution steps without fully understanding underlying theory."
        )

    if diagnostics["effort_gap"]:
        insights.append(
            "Engagement Gap Detected: \n"
            "Behavioral patterns indicate limited structured problem-solving effort.  \n"
            "You may be giving short or surface-level responses instead of structured reasoning."
        )

    if diagnostics["strategy_gap"]:
        insights.append(
            "Reasoning Strategy Misalignment:  \n"
            "Your approach lacks step-by-step logical structure.  \n"
            "Answers may be partially correct but reasoning flow is inconsistent."
        )

    if diagnostics["high_disagreement"]:
        insights.append(
            "Performance Instability Detected: \n"
            "Your cognitive strength and behavioral engagement signals disagree.  \n"
            "This may indicate inconsistent performance across similar tasks."
        )

    if not insights:
        insights.append(
            "Strong Cognitive Alignment: \n"
            "Your conceptual understanding, reasoning structure, and engagement are well aligned.  \n"
            "You are operating at a stable and confident learning state."
        )

    for msg in insights:
        st.markdown(msg)

    # ===================== IMPROVEMENT PLAN =====================
    st.markdown("Improvement Suggestion")

    if fused_score < 0.4:
        st.markdown(
            """
            Priority: Strengthen Foundations
            - Revisit core definitions and theoretical explanations.
            - Solve 10–15 basic structured practice problems.
            - Focus on understanding *why* each step works.
            - Avoid jumping directly to final answers.
            """
        )

    elif fused_score < 0.7:
        st.markdown(
            """
            Priority: Improve Structured Reasoning
            - Break problems into explicit step-by-step logic.
            - Practice writing intermediate reasoning before concluding.
            - Validate each step before moving forward.
            - Attempt mixed-difficulty practice sets.
            """
        )

    else:
        st.markdown(
            """
            Priority: Advance Mastery
            - Attempt multi-stage and cross-topic problems.
            - Practice timed reasoning tasks.
            - Teach the concept back in your own words.
            - Introduce complexity variations.
            """
        )

    # ===================== CONFIDENCE INTERPRETATION =====================
    st.markdown("Model Confidence Interpretation")

    if confidence < 0.5:
        st.write(
            "The system has low confidence due to inconsistent signals. "
            "Assessment should be interpreted cautiously."
        )
    elif confidence < 0.8:
        st.write(
            "The system has moderate confidence. "
            "Your learning signals are mostly consistent."
        )
    else:
        st.write(
            "The system has high confidence in this cognitive assessment."
        )

    # ===================== TRACE =====================
    with st.expander("🔍 Reasoning Trace", expanded=show_trace):
        st.caption("Detailed logs of the decomposition and verification process.")
        logs = result.get("logs", "")
        st.code(logs)

    # ===================== EXPORT =====================
    export = {
        "prompt": prompt,
        "fused_ce_score": float(fused_score),
        "confidence": float(confidence),
        "diagnostics": diagnostics,
        "timestamp": datetime.utcnow().isoformat()
    }

    st.download_button(
        "Download Session Report",
        data=json.dumps(export, indent=2),
        file_name="camre_session.json",
        mime="application/json",
    )


# ===================== EMPTY STATE =====================
if not run_btn:
    st.info(
        "Enter a learning question above and run the session.\n\n"
        "CAMRE EDU will provide:\n"
        "• Step-by-step reasoning\n"
        "• Cognitive Efficiency score\n"
        "• Diagnostic feedback\n"
        "• Improvement suggestions"
    )