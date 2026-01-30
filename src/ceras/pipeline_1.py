# pipeline_1.py
# Solver-grounded, multi-verifier Tree-of-Thoughts
# Final Answer = reviewed, corrected, high-quality steps

from tree_of_thoughts import TreeOfThoughts
from inference import run_inference_pipeline
from llm_utils import run_decomposer

import time
import io
import contextlib
import json
from typing import List


# ===================== CONFIG =====================
MAX_STRATEGIES = 2
MAX_FINAL_STEPS = 5
MAX_TOTAL_LLM_CALLS = 8   # quality mode


# ===================== HELPERS =====================
def normalize(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def clean(items: List[str], limit: int):
    out, seen = [], set()
    for s in items:
        t = normalize(s)
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break
    return out


# ===================== STRATEGY =====================
def propose_strategies(prompt: str) -> List[str]:
    meta_prompt = f"""
Propose at most {MAX_STRATEGIES} HIGH-LEVEL strategies
to solve the following problem.

Rules:
- Do NOT compute the solution
- Do NOT give steps
- One sentence per strategy

Problem:
{prompt}
"""
    out = run_inference_pipeline(meta_prompt, auto_extend=False)

    strategies = []
    if isinstance(out, dict):
        for k in ("strategies", "steps", "thoughts", "outputs"):
            if k in out:
                v = out[k]
                strategies.extend(v if isinstance(v, list) else [v])

    return clean(strategies, MAX_STRATEGIES)


# ===================== SOLVER STEP GENERATOR =====================
def generate_solver_steps(prompt: str, strategy: str) -> List[str]:
    solver_prompt = f"""
Solve the following problem step by step:

{prompt}

Chosen strategy:
{strategy}

Generate an ordered list of EXECUTABLE solution steps.

Rules:
- Each step must directly progress toward solving the problem
- NO meta reasoning (no planning, no identifying problem type)
- NO explanations
- Do NOT compute the final numeric answer
- Output as a numbered list
"""

    raw = run_decomposer(solver_prompt)
    if not isinstance(raw, list):
        raw = [raw]

    return clean(raw, MAX_FINAL_STEPS)


# ===================== QUALITY VERIFIER (NEW) =====================
def review_and_fix_steps(prompt: str, steps: List[str]) -> List[str]:
    """
    Second-pass verifier.
    Reviews steps for correctness and completeness.
    Fixes them if needed.
    """

    review_prompt = f"""
You are a strict solution reviewer.

Original problem:
{prompt}

Proposed solution steps:
{json.dumps(steps, indent=2)}

Your task:
- Check whether these steps correctly and completely solve the problem.
- If they are correct, respond with JSON:
  {{ "status": "correct", "final_steps": [...] }}

- If they are incorrect or incomplete, respond with JSON:
  {{
    "status": "fix",
    "issues": [
      {{
        "step_index": <index>,
        "problem": "<what is wrong or missing>",
        "fix": "<exact corrected step>"
      }}
    ],
    "additional_steps": ["<step to add>", ...]
  }}

Rules:
- Be precise and minimal
- Do NOT add explanations
- Output ONLY valid JSON
"""

    out = run_inference_pipeline(review_prompt, auto_extend=False)

    # Try to parse JSON from verifier output
    try:
        raw = out.get("raw_verifier") if isinstance(out, dict) else out
        if isinstance(raw, str):
            parsed = json.loads(raw)
        elif isinstance(raw, dict):
            parsed = raw
        else:
            return steps
    except Exception:
        return steps

    if parsed.get("status") == "correct":
        return clean(parsed.get("final_steps", steps), MAX_FINAL_STEPS)

    if parsed.get("status") == "fix":
        fixed_steps = steps[:]

        for issue in parsed.get("issues", []):
            idx = issue.get("step_index")
            fix = issue.get("fix")
            if isinstance(idx, int) and 0 <= idx < len(fixed_steps):
                fixed_steps[idx] = fix

        for s in parsed.get("additional_steps", []):
            fixed_steps.append(s)

        return clean(fixed_steps, MAX_FINAL_STEPS)

    return steps


# ===================== MAIN =====================
def main(prompt: str):
    stdout_buffer = io.StringIO()
    llm_calls_used = 0

    with contextlib.redirect_stdout(stdout_buffer):
        tree = TreeOfThoughts()

        root = tree.add_node(
            text=prompt,
            parent_id=None,
            role="root",
            metadata={"ts": time.time()}
        )
        tree.set_root(root.id)

        # ---- STRATEGY ----
        strategies = propose_strategies(prompt)
        llm_calls_used += 1
        strategy = strategies[0] if strategies else "Solve the problem directly."

        strategy_node = tree.add_node(
            text=strategy,
            parent_id=root.id,
            role="strategy"
        )

        # ---- SOLVER STEPS ----
        solver_steps = generate_solver_steps(prompt, strategy)
        llm_calls_used += 1

        # ---- VERIFIER #1 (GATEKEEPER) ----
        verified = run_inference_pipeline(
            prompt + "\nSteps:\n" + "\n".join(solver_steps),
            auto_extend=False
        )
        llm_calls_used += 1

        if verified.get("status") != "accepted":
            solver_steps = [
                "Apply the chosen strategy step by step to transform the given expression.",
                "Derive the result logically from the transformed expression."
            ]

        # ---- VERIFIER #2 (QUALITY REVIEW) ----
        final_steps = review_and_fix_steps(prompt, solver_steps)
        llm_calls_used += 1

        # ---- TREE BUILD ----
        for step in final_steps:
            tree.add_node(
                text=step,
                parent_id=strategy_node.id,
                role="subtask"
            )

        tree.save_json("tree_of_thoughts_example.json")

    return {
        "final_answer": final_steps,
        "strategy_used": strategy,
        "llm_calls_used": llm_calls_used,
        "tree": tree,
        "logs": stdout_buffer.getvalue()
    }
