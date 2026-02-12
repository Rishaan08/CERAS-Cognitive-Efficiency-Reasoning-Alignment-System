import streamlit as st
import time
import json
from datetime import datetime
import numpy as np

# --- Reasoning pipeline ---
from pipeline_1 import main as run_infer

# --- CERAS fusion engine ---
from ceras.fusion import CERASFusion

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="CAMRE EDU",
    page_icon="🧠",
    layout="wide",
)

# ===================== HEADER =====================
st.markdown("## 🧠 CAMRE EDU - Intelligent Learning Lab")
st.caption("LLM Reasoning + Cognitive Efficiency + Behavioral Diagnostics")


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ⚙️ Run Configuration")
    show_trace = st.checkbox("Show Reasoning Trace", value=True)
    show_tree = st.checkbox("Show Tree JSON", value=False)


# ===================== INPUT =====================
prompt = st.text_area(
    "Enter your learning question or problem",
    height=150
)

run_btn = st.button("▶ Run Learning Session")


# ===================== TEMP FEATURE ADAPTER =====================
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

    # ===================== METRICS =====================
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Fused CE Score", f"{fused_score:.2f}")

    with c2:
        st.metric("Confidence", f"{confidence:.2f}")

    with c3:
        st.metric("Execution Time", f"{runtime:.2f}s")

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
    if show_trace:
        st.markdown("Reasoning Trace")
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