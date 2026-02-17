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
    page_title="CERAS",
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
    .prompt-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 10px;
        height: 200px;
        overflow-y: auto;
        font-size: 14px;
        color: #e2e8f0;
        text-align: justify;
    }
    .prompt-box-bad {
        background-color: #2a1515;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #451a1a;
        margin-bottom: 10px;
        font-size: 14px;
        color: #fca5a5;
        text-align: justify;
    }
    /* Justify text for better readability */
    p, li {
        text-align: justify;
    }
    /* Sidebar specific refinements */
    .sidebar-box {
        background-color: #0f172a;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

#Header
st.markdown("## 🧠 CERAS - Cognitive Efficiency & Reasoning Alignment System")
st.markdown("""
**CERAS** is an advanced adaptive learning environment designed to optimize how you learn and solve complex problems. 
By fusing **Large Language Model (LLM)** reasoning capabilities with real-time **Cognitive Efficiency** metrics, current behavioral diagnostics, and 
neuro-fuzzy alignment, CERAS provides a personalized learning experience. It analyzes your input complexity, structure, and intent to guide you 
through deep concepts with tailored roadmaps, ensuring you don't just get answers, but truly master the material.
""")


#Sidebar
with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding-bottom: 20px;">
            <h1 style="margin-bottom:0; font-size: 2.5rem;">🧠</h1>
            <h2 style="margin-top:0;">CERAS</h2>
            <p style="color:#94a3b8; font-size:14px; margin-top:5px;">
                Cognitive Efficiency & Reasoning Alignment System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ System Status")

    st.markdown(
        """
        <div class="sidebar-box">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>🔌 <b>Groq API</b></span>
                <span style="color:#4ade80;">● Connected</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>⚡ <b>Fusion Engine</b></span>
                <span style="color:#4ade80;">● Active</span>
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
            <p style="margin:8px 0;"> ⚖️ <b>ANFIS</b><br><span style="color:#94a3b8;">Neuro-Fuzzy Logic Alignment</span></p>
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


# ===================== GOOD EXAMPLES =====================
st.markdown("### 🌟 GOOD EXAMPLES TO PROMPT (High CE Score)")
st.caption("These prompts are detailed and structured, leading to higher cognitive efficiency scores.")

gp1 = "Explain the concept of quantum entanglement in detail, using a comprehensive analogy of two magic dice that are separated by a vast distance but still show the same numbers when rolled. Discuss the implications for information transfer and how this phenomenon challenges classical physics intuitions about locality and realism. Also touch upon the concept of 'spooky action at a distance' as described by Einstein."
gp2 = "Describe the biological process of photosynthesis in plants, detailing the light-dependent reactions and the Calvin cycle. Explain how chlorophyll captures light energy to convert carbon dioxide and water into glucose and oxygen, and discuss the importance of this process for life on Earth and the global carbon cycle. Mention the role of ATP and NADPH as energy carriers during this transformation."
gp3 = "Analyze the profound historical impact of the Gutenberg printing press on European society during the Renaissance. Discuss how it facilitated the spread of literacy, the standardization of languages, the dissemination of scientific knowledge, and the religious shifts associated with the Reformation, ultimately reshaping the cultural and intellectual landscape. Consider the long-term effects on democratization of information."
gp4 = "Compare and contrast supervised and unsupervised machine learning paradigms. Explain the key differences in their training data requirements, with supervised learning using labeled datasets and unsupervised learning finding patterns in unlabeled data. Provide specific examples of algorithms and real-world applications for each approach, such as classification versus clustering. Discuss the trade-offs in terms of data preparation and model interpretability."

# Layout: 2 Columns x 2 Rows for better readability
c1, c2 = st.columns(2)

with c1:
    st.markdown(f'<div class="prompt-box">{gp1}</div>', unsafe_allow_html=True)
    if st.button("� Use: Quantum Physics", key="btn_gp1"):
        set_prompt(gp1)
        st.rerun()

    st.markdown(f'<div class="prompt-box">{gp3}</div>', unsafe_allow_html=True)
    if st.button("� Use: Printing Press", key="btn_gp3"):
        set_prompt(gp3)
        st.rerun()

with c2:
    st.markdown(f'<div class="prompt-box">{gp2}</div>', unsafe_allow_html=True)
    if st.button("📝 Use: Photosynthesis", key="btn_gp2"):
        set_prompt(gp2)
        st.rerun()

    st.markdown(f'<div class="prompt-box">{gp4}</div>', unsafe_allow_html=True)
    if st.button("📝 Use: ML Paradigms", key="btn_gp4"):
        set_prompt(gp4)
        st.rerun()

if st.session_state.auto_run:
     # Force a rerun if auto_run was set by the button click callback 
     # (Though we are calling rerun inside the if, the callback alternative is safer if we weren't doing direct logic checks? 
     # actually simply calling set_prompt inside the if block works because we rerun immediately).
     pass 

# ===================== BAD EXAMPLES =====================
st.markdown("### ⚠️ BAD EXAMPLES TO PROMPT (Low CE Score)")
st.caption("These prompts are too short or vague, leading to lower cognitive efficiency scores.")

bp1 = "What is AI?"
bp2 = "Fix my code."
bp3 = "Tell me a joke."
bp4 = "Why is sky blue?"

bc1, bc2, bc3, bc4 = st.columns(4)

with bc1:
    st.markdown(f'<div class="prompt-box-bad">{bp1}</div>', unsafe_allow_html=True)
    if st.button("Use: AI?", key="btn_bp1"):
        set_prompt(bp1)
        st.rerun()

with bc2:
    st.markdown(f'<div class="prompt-box-bad">{bp2}</div>', unsafe_allow_html=True)
    if st.button("Use: Fix Code", key="btn_bp2"):
        set_prompt(bp2)
        st.rerun()

with bc3:
    st.markdown(f'<div class="prompt-box-bad">{bp3}</div>', unsafe_allow_html=True)
    if st.button("Use: Joke", key="btn_bp3"):
        set_prompt(bp3)
        st.rerun()

with bc4:
    st.markdown(f'<div class="prompt-box-bad">{bp4}</div>', unsafe_allow_html=True)
    if st.button("Use: Sky", key="btn_bp4"):
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
    
    # Reset auto_run so it doesn't loop
    if st.session_state.auto_run:
        st.session_state.auto_run = False
    
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
should_run = False
if run_btn:
    should_run = True
if st.session_state.auto_run:
    should_run = True
    st.session_state.auto_run = False

if should_run and prompt.strip():
    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        # Calculate User Latency (Formulation Time)
        st.session_state.formulation_time = time.time() - st.session_state.start_time
        
        result = run_infer(prompt)
        runtime = time.time() - t0
        
        # Store results in session state for persistence
        st.session_state.current_result = result
        st.session_state.current_runtime = runtime
        st.session_state.current_prompt = prompt

        # Generate and store adaptive response
        from llm_utils import generate_adaptive_response
        cepm_score, cnn_score, anfis_score = extract_ceras_features(prompt, result)
        
        # We need the diagnostics for the adaptive response
        # Re-calculating fusion here to get diagnostics for the adaptive response generation
        # This duplicates the logic below but ensures we have the data for the LLM call
        fusion_engine = CERASFusion()
        fusion_df = fusion_engine.fuse(
            student_ids=[1],
            cepm_scores=[cepm_score],
            cnn_scores=[cnn_score],
            anfis_scores=[anfis_score]
        )
        diagnostics = fusion_df["diagnostics"].iloc[0]
        fused_score = fusion_df["fused_ce_score"].iloc[0] # Needed for tone selection
        
        final_steps = result.get("final_answer", [])
        
        with st.spinner("Generating personalized learning summary..."):
             st.session_state.adaptive_res = generate_adaptive_response(prompt, final_steps, fused_score, diagnostics)

# Render Results if available
if "current_result" in st.session_state and st.session_state.current_result is not None:
    result = st.session_state.current_result
    runtime = st.session_state.current_runtime
    # Use the prompt that generated the result for scoring to ensure consistency
    result_prompt = st.session_state.current_prompt

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
        st.metric("Input Volume", f"{len(result_prompt)} chars")
    with r4:
        #Simulated "Live" status for external sensors
        st.metric("Gaze Tracker", "Active", delta="Tracking", delta_color="normal")

    #CERAS Analysis
    st.markdown("---")
    st.markdown("Cognitive Efficiency Analysis")
    
    # Extract simulated signals (Using result_prompt)
    cepm_score, cnn_score, anfis_score = extract_ceras_features(result_prompt, result)

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
    
    if "adaptive_res" in st.session_state:
        st.markdown(st.session_state.adaptive_res)
    else:
        st.info("Adaptive response not available.")

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
        "prompt": result_prompt,
        "fused_ce_score": float(fused_score),
        "confidence": float(confidence),
        "diagnostics": diagnostics,
        "timestamp": datetime.utcnow().isoformat()
    }

    st.download_button(
        "Download Session Report",
        data=json.dumps(export, indent=2),
        file_name="ceras_session.json",
        mime="application/json",
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