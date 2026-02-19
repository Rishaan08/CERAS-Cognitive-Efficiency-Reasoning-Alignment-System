# pipeline_1.py
# Solver-grounded, multi-verifier Tree-of-Thoughts
# Final Answer = reviewed, corrected, high-quality steps

from tree_of_thoughts import TreeOfThoughts
from inference import run_inference_pipeline
from llm_utils import run_decomposer, call_llm
from inference import run_inference_pipeline, extract_json_from_text, normalize_subtasks
import re

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
def propose_strategies(prompt: str, api_config: dict = None) -> List[str]:
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
    out = call_llm(meta_prompt, api_config=api_config)
    
    # Clean output: simple line splitting since rules say "One sentence per strategy"
    lines = []
    if out:
        for line in out.splitlines():
            line = line.strip()
            # Remove numbering "1. ", "- "
            line = re.sub(r'^[\d+\.\)\-\*]+\s+', '', line)
            if len(line) > 10: 
                lines.append(line)

    return clean(lines, MAX_STRATEGIES)


# ===================== SOLVER STEP GENERATOR =====================
def generate_solver_steps(prompt: str, strategy: str, api_config: dict = None) -> List[str]:
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

    raw = run_decomposer(solver_prompt, api_config=api_config)
    if not isinstance(raw, list):
        raw = [raw]

    return clean(raw, MAX_FINAL_STEPS)


# ===================== QUALITY VERIFIER (NEW) =====================
def review_and_fix_steps(prompt: str, steps: List[str], api_config: dict = None) -> List[str]:
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

    raw_out = call_llm(review_prompt, api_config=api_config)

    # Try to parse JSON from verifier output
    try:
        parsed = extract_json_from_text(raw_out)
        if not parsed:
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
def main(prompt: str, api_config: dict = None):
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

        # 1. GENERATE CANDIDATES (Thoughts)
        print("--- Step 1: Generating Candidates ---")
        strategies = propose_strategies(prompt, api_config=api_config)
        llm_calls_used += 1
        
        candidates = []
        for strategy in strategies:
            print(f"Generating steps for strategy: {strategy}")
            steps = generate_solver_steps(prompt, strategy, api_config=api_config)
            llm_calls_used += 1
            candidates.append({"strategy": strategy, "steps": steps})

        # 2. VERIFY CANDIDATES
        print("\n--- Step 2: Verifying Candidates ---")
        verified_candidates = []
        for cand in candidates:
            # Format steps for verification
            steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(cand["steps"])])
            full_text = prompt + "\nStrategy: " + cand["strategy"] + "\nSteps:\n" + steps_text
            
            verification = run_inference_pipeline(
                full_text,
                auto_extend=False,
                api_config=api_config
            )
            llm_calls_used += 1
            
            cand["verification"] = verification
            if verification.get("status") == "accepted":
                verified_candidates.append(cand)
            else:
                 print(f"Candidate rejected: {verification.get('message')}")

        # 3. SELECT BEST PATH
        print("\n--- Step 3: Selecting Best Path ---")
        best_candidate = None
        if verified_candidates:
            # For now, pick the first accepted one (or could use confidence if available)
            best_candidate = verified_candidates[0]
            print("Selected a verified path.")
        elif candidates:
            # Fallback to the first generated path if none verified (graceful degradation)
            print("No path fully verified. Falling back to first generated path.")
            best_candidate = candidates[0]
        else:
             best_candidate = {"strategy": "Direct", "steps": ["Analysis failed to generate steps."]}
        
        
        final_strategy = best_candidate["strategy"]
        raw_steps = best_candidate["steps"]

        # 4. FINAL STEPS (Quality Review)
        print("\n--- Step 4: Final Polish ---")
        final_steps = review_and_fix_steps(prompt, raw_steps, api_config=api_config)
        llm_calls_used += 1

        # ---- TREE BUILD (For Visualization) ----
        strategy_node = tree.add_node(
            text=final_strategy,
            parent_id=root.id,
            role="strategy"
        )
        for step in final_steps:
            tree.add_node(
                text=step,
                parent_id=strategy_node.id,
                role="subtask"
            )

    return {
        "final_answer": final_steps,
        "strategy_used": final_strategy,
        "llm_calls_used": llm_calls_used,
        "tree": tree,
        "logs": stdout_buffer.getvalue()
    }
