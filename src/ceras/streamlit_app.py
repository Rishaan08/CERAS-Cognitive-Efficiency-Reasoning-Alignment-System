import streamlit as st
import time
import json
from datetime import datetime

# --- Import your pipeline ---
from pipeline_1 import main as run_infer


# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="CERASSS Reasoning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== GLOBAL STYLES =====================
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .final-box {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 16px;
            margin-top: 12px;
        }
        .trace-box {
            background-color: #020617;
            border-radius: 10px;
            padding: 14px;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
        }
        .metric-box {
            background-color: #020617;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== HEADER =====================
st.markdown("## 🧠 **CERASSS Reasoning Lab**")
st.caption("GenAI · Tree-of-Thoughts · Multi-Verifier · ML-Ready Telemetry")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ⚙️ Run Configuration")

    show_trace = st.checkbox("Show Reasoning Trace", value=True)
    show_tree = st.checkbox("Show Tree JSON", value=False)

    st.markdown("---")
    st.markdown("### 🧪 Execution Mode")
    mode = st.radio(
        "Reasoning Mode",
        ["Auto (Intent-Aware)", "Learning-Focused", "Solver-Focused"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📦 Session Info")
    st.caption(f"Session start: {datetime.now().strftime('%H:%M:%S')}")

# ===================== MAIN INPUT =====================
prompt = st.text_area(
    "Enter your prompt",
    placeholder="e.g. can you help me learn <topic>\nor\nsolve <equation>",
    height=160,
)

run_btn = st.button("▶ Run Inference", use_container_width=False)

# ===================== RUN PIPELINE =====================
if run_btn and prompt.strip():
    with st.spinner("Running CERAS reasoning pipeline..."):
        t0 = time.time()
        result = run_infer(prompt)
        runtime = time.time() - t0

    # ===================== METRICS =====================
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-box'><b>Status</b><br>Completed</div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='metric-box'><b>Execution Time</b><br>{runtime:.2f}s</div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='metric-box'><b>LLM Calls</b><br>{result.get('llm_calls_used', '—')}</div>",
            unsafe_allow_html=True
        )

    # ===================== FINAL ANSWER =====================
    st.markdown("## ✅ Final Answer")

    final_steps = result.get("final_answer", [])

    st.markdown("<div class='final-box'>", unsafe_allow_html=True)
    if isinstance(final_steps, list):
        for i, step in enumerate(final_steps, 1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.write(final_steps)
    st.markdown("</div>", unsafe_allow_html=True)

    # ===================== REASONING TRACE =====================
    if show_trace:
        st.markdown("## 🧵 Reasoning Trace")
        logs = result.get("logs", "")
        st.markdown(
            f"<div class='trace-box'>{logs}</div>",
            unsafe_allow_html=True
        )

    # ===================== TREE JSON =====================
    if show_tree:
        st.markdown("## 🌳 Tree-of-Thoughts (JSON)")
        try:
            tree_json = result["tree"].to_dict()
            st.json(tree_json, expanded=False)
        except Exception:
            st.warning("Tree object could not be serialized.")

    # ===================== EXPORT =====================
    st.markdown("## 📤 Export")
    export = {
        "prompt": prompt,
        "final_answer": final_steps,
        "llm_calls": result.get("llm_calls_used"),
        "runtime_sec": runtime,
        "timestamp": datetime.utcnow().isoformat()
    }

    st.download_button(
        "Download run as JSON",
        data=json.dumps(export, indent=2),
        file_name="ceras_run.json",
        mime="application/json",
    )

# ===================== EMPTY STATE =====================
if not run_btn:
    st.info(
        "Enter a prompt above and click **Run Inference**.\n\n"
        "CERAS will automatically decide whether this is a **learning** or **solving** task, "
        "apply the correct reasoning pipeline, and return clean, high-quality steps."
    )
