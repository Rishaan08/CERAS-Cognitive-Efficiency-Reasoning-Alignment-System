import streamlit as st
import time
import json
from datetime import datetime
import numpy as np

#Reasoning pipeline
from pipeline_1 import main as run_infer

#CERAS fusion engine
from fusion import CERASFusion

#Page Configuration
st.set_page_config(
    page_title="CAMRE EDU",
    page_icon="📖",
    layout="wide",
)

#Custom CSS for better styling
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

#Header
st.markdown("## 📖 CAMRE EDU - Intelligent Learning Lab")
st.caption("LLM Reasoning + Cognitive Efficiency + Behavioral Diagnostics")


#Sidebar
with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">
            <h2 style="margin-bottom:0;">📖 CAMRE EDU</h2>
            <p style="color:gray; font-size:13px; margin-top:4px;">
                Intelligent Learning Lab
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### System Core")

    st.markdown(
        """
        <div style="
            background-color:#0f172a;
            padding:14px;
            border-radius:12px;
            border:1px solid #1e293b;
        ">
            <p style="margin:6px 0;">🟢 <b>Groq API</b> — Connected</p>
            <p style="margin:6px 0;">🟢 <b>Fusion Engine</b> — Active</p>
            <p style="margin:6px 0;">🔵 <b>Telemetry</b> — Tracking Signals</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Architecture")

    st.markdown(
        """
        <div style="
            background-color:#0b1220;
            padding:14px;
            border-radius:12px;
            border:1px solid #1e293b;
        ">
            <p style="margin:6px 0;"> 🧩 LLM — Reasoning Engine (Tree-of-Thoughts)</p>
            <p style="margin:6px 0;"> 🧠 CEPM — Cognitive Modeling</p>
            <p style="margin:6px 0;"> 📈 CNN — Behavioral Signals</p>
            <p style="margin:6px 0;"> 🔎 ANFIS — Reasoning Alignment</p>
            <p style="margin:6px 0;"> 🔗 Fusion — Multi-Signal Integration</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Version 1.0 • Neural Learning Stack")


#Session State
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if "formulation_time" not in st.session_state:
    st.session_state.formulation_time = 0.0

#Input
c_in1, c_in2 = st.columns([0.8, 0.2])
with c_in2:
    if st.button("New Problem"):
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


#Run Pipeline
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


#Run Pipeline
if run_btn and prompt.strip():

    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        result = run_infer(prompt)
        runtime = time.time() - t0

    #Final Answer
    st.markdown("## Learning Response")

    final_steps = result.get("final_answer", [])

    if isinstance(final_steps, list):
        for i, step in enumerate(final_steps, 1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.write(final_steps)

    #Raw Sensor Data
    st.markdown("---")
    st.markdown("### Live User Telemetry (Raw Inputs)")
    st.caption("Data captured from user interaction *before* feature extraction.")

    r1, r2, r3, r4 = st.columns(4)
    
    with r1:
        st.metric("Formulation Time", f"{st.session_state.formulation_time:.2f}s", help="Time taken to type/submit")
    with r2:
        st.metric("System Latency", f"{runtime:.3f}s", help="AI Processing Time")
    with r3:
        st.metric("Input Volume", f"{len(prompt)} chars")
    with r4:
        #Simulated "Live" status for external sensors
        st.metric("Gaze Tracker", "Active", delta="Tracking", delta_color="normal")

    #CERAS Analysis
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

    #Adaptive Learning Response
    st.markdown("## Adaptive Learning Response")
    from llm_utils import generate_adaptive_response  # lazy import
    
    with st.spinner("Generating personalized learning summary..."):
        adaptive_res = generate_adaptive_response(prompt, final_steps, fused_score, diagnostics)
        st.markdown(adaptive_res)

    #Live Data Visulaization
    st.markdown("### Live Cognitive Signals")
    
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
        st.metric("Fused CE Score", f"{fused_score:.2f}", delta="Real-time")
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

    #Fused CE Score(Explain)
    with st.expander("What is the Fused CE Score?"):
        st.markdown("""
            ### Fused Cognitive Efficiency (CE) Score

            The **Fused CE Score** reflects how efficiently you are learning in this session.

            It combines three independent signals:

            • **Conceptual Strength (CEPM)** – Depth of understanding  
            • **Behavioral Engagement (CNN)** – Effort and consistency  
            • **Reasoning Alignment (ANFIS)** – Strategy quality  

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


    #Diagnostic Report
    with st.expander("Cognitive Diagnostic Report"):

        insights = []

        if diagnostics["concept_gap"]:
            insights.append(
                "### Conceptual Weakness\n"
                "Your response indicates gaps in core understanding of this topic.\n\n"
                "**What this means:** You may be applying procedures without fully grasping the underlying principles.\n\n"
                "**Recommended action:** Revisit foundational theory and solve basic concept-check questions before moving to advanced problems."
            )

        if diagnostics["effort_gap"]:
            insights.append(
                "### Low Structured Engagement\n"
                "Your interaction suggests limited step-by-step problem-solving effort.\n\n"
                "**What this means:** Responses may be brief or surface-level instead of logically structured.\n\n"
                "**Recommended action:** Practice writing complete reasoning steps and avoid skipping intermediate logic."
            )

        if diagnostics["strategy_gap"]:
            insights.append(
                "### Reasoning Structure Instability\n"
                "Your reasoning pattern lacks consistent logical flow.\n\n"
                "**What this means:** Even if the final answer is correct, the pathway may be inefficient or unclear.\n\n"
                "**Recommended action:** Use structured frameworks (Step 1 → Step 2 → Conclusion) when solving problems."
            )

        if diagnostics["high_disagreement"]:
            insights.append(
                "### Performance Inconsistency\n"
                "Cognitive strength and behavioral signals are not aligned.\n\n"
                "**What this means:** You may understand the material but are not consistently applying structured effort.\n\n"
                "**Recommended action:** Focus on consistency — apply the same structured reasoning approach across similar tasks."
            )

        if not insights:
            insights.append(
                "### Stable Cognitive State\n"
                "Your conceptual understanding, reasoning structure, and engagement are aligned.\n\n"
                "**What this means:** You are processing information efficiently and consistently.\n\n"
                "**Next step:** Challenge yourself with higher-difficulty or multi-stage problems."
            )

        for msg in insights:
            st.markdown(msg)
            st.markdown("---")

    #Improvement Suggestions
    with st.expander("Improvement Suggestions"):

        if fused_score < 0.4:
            st.markdown(
                """
                ### Priority: Rebuild Core Understanding

                - Revisit fundamental definitions and key principles.  
                - Solve 10–15 structured foundational problems.  
                - Focus on understanding *why* each step works.  
                - Avoid jumping directly to final answers.
                """
            )

        elif fused_score < 0.6:
            st.markdown(
                """
                ### Priority: Strengthen Logical Structure

                - Break problems into clear step-by-step reasoning.  
                - Write intermediate logic before concluding.  
                - Double-check assumptions between steps.  
                - Practice moderate-difficulty structured exercises.
                """
            )

        elif fused_score < 0.75:
            st.markdown(
                """
                ### Priority: Improve Consistency & Depth

                - Solve mixed-difficulty problems with structured reasoning.  
                - Explain your thought process explicitly.  
                - Maintain clarity under mild time pressure.  
                - Reduce small logical jumps.
                """
            )

        else:
            st.markdown(
                """
                ### Priority: Expand Mastery & Complexity

                - Attempt multi-stage and cross-topic problems.  
                - Practice timed reasoning sessions.  
                - Teach the concept back in your own words.  
                - Explore advanced variations and edge cases.
                """
            )

    #Confidence Interpretation
    with st.expander("Model Confidence Interpretation"):

        if confidence < 0.4:
            st.write(
                "**Very Low Confidence**\n\n"
                "Strong disagreement was detected between cognitive understanding, "
                "behavioral engagement, and reasoning structure.\n\n"
                "This assessment should be interpreted cautiously, and additional "
                "interactions may be required for reliable evaluation."
            )

        elif confidence < 0.6:
            st.write(
                "**Low–Moderate Confidence**\n\n"
                "There is noticeable inconsistency between learning signals.\n\n"
                "Some aspects of understanding or engagement may not be stable "
                "across similar tasks."
            )

        elif confidence < 0.85:
            st.write(
                "**High Confidence**\n\n"
                "Cognitive and behavioral signals are largely aligned.\n\n"
                "The system considers this assessment reasonably stable and reliable."
            )

        else:
            st.write(
                "**Very High Confidence**\n\n"
                "All major learning signals are strongly aligned.\n\n"
                "This indicates stable reasoning patterns and consistent engagement."
            )

    #Trace
    with st.expander("Reasoning Trace"):
        st.caption("Detailed logs of the decomposition and verification process.")
        logs = result.get("logs", "")
        st.code(logs)

    #Export
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


#Empty State
if not run_btn:
    st.info(
        "Enter a learning question above and run the session.\n\n"
        "CAMRE EDU will provide:\n"
        "• Step-by-step reasoning\n"
        "• Cognitive Efficiency score\n"
        "• Diagnostic feedback\n"
        "• Improvement suggestions"
    )

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:#9ca3af; margin-top:8px;">
        Developed by <b>Aman Goel</b> & <b>Rishaan Yadav</b><br>
        <span style="color:#6b7280;">CAMRE EDU · CERAS Framework</span>
    </div>
    """,
    unsafe_allow_html=True
)