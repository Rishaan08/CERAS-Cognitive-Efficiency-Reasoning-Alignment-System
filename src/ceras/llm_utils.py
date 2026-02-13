import json
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# ===================== MODEL =====================
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, api_key=os.environ.get("GROQ_API_KEY"))

# ===================== INTENT DETECTION =====================
def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in (
        "learn", "understand", "explain", "introduction",
        "guide", "tutorial", "walk me through"
    )):
        return "learning"
    return "solving"

# ===================== SOLVER-GROUNDED PROMPTS =====================
SOLVER_PREFERENCE_RULES = """
Preference rules:
- Prefer transforming or rewriting expressions over directly computing values
- If a symbolic or structural simplification exists, use it before numeric evaluation
- Avoid expanding large numbers early if a simpler form can be obtained
"""

DECOMP_PROMPT_JSON = """You are an Expert Tutor utilizing Dynamic Programming.
Your goal is to guide the student through the problem by breaking it down into detailed, pedagogical sub-problems.
Each sub-problem should be a clear, instructional step that teaches a specific concept or operation required for the solution.

Rules:
1. DO NOT just solve the problem. Break it into teachable components.
2. Each step should explain *what* to do and *why* (briefly), but not give the final answer immediately.
3. Use a "Divide and Conquer" approach: recursive breakdown.
4. Output strict JSON.

Example:
User: "51^2 - 49^2"
Result:
{{
  "subtasks": [
    "Recall the algebraic identity for the difference of two squares: a^2 - b^2 = (a-b)(a+b)",
    "Identify the values for 'a' and 'b' from the given expression (a=51, b=49)",
    "Set up the substitution: replace a with 51 and b with 49 in the identity",
    "Calculate the difference term (51 - 49)",
    "Calculate the sum term (51 + 49)",
    "Multiply the calculated difference and sum to find the final result"
  ]
}}

User query:
{query}

Respond in STRICT JSON format used in the example.
"""

DECOMP_PROMPT_SIMPLE = """Break the problem into step-by-step subproblems.
1. Identify key concepts.
2. Structure the approach.
3. Perform the necessary calculations.

User query:
{query}

Output the steps as a numbered list.
"""

DECOMP_PROMPT_SIMPLE = f"""Generate EXECUTABLE SOLUTION STEPS.

{SOLVER_PREFERENCE_RULES}

Rules:
- Each step must directly operate on the content of the query
- NO meta reasoning
- NO planning
- NO explanations
- One step per line

User query:
'''{{query}}'''
"""

DECOMP_PROMPT_MIN = f"""Return 3–6 EXECUTABLE SOLUTION STEPS.

{SOLVER_PREFERENCE_RULES}

Rules:
- Steps must transform the problem toward a solution
- NO meta reasoning
- NO planning
- NO classification
- NO explanation

User query:
'''{{query}}'''
"""

# ===================== LEARNING PROMPT =====================
DECOMP_PROMPT_LEARNING = """You are designing a LEARNING PATH.

User intent: learn a topic.

Rules:
- Steps should build understanding progressively
- Start from prerequisites
- Introduce key concepts before tools
- Reference documentation or ecosystem where relevant
- Avoid installation-only steps
- Avoid meta reasoning ("identify the problem")

Good step examples:
- "Understand what Pydantic is and why it is used"
- "Learn core Pydantic concepts like BaseModel and validation"
- "Explore how Logfire integrates with Pydantic"
- "Read official documentation and examples"
- "Build a small example to connect concepts"

Return an ordered list of LEARNING steps (one per line).

User query:
'''{query}'''
"""

# ===================== STEP FILTERING =====================
META_PATTERNS = (
    "identify the type",
    "determine the relevance",
    "research",
    "analyze the question",
    "decide the approach",
    "understand the problem",
    "high-level",
    "plan",
    "strategy",
    "consider whether",
    "review concepts",
)

def looks_like_bruteforce(step: str) -> bool:
    s = step.lower()
    return (
        ("calculate" in s or "evaluate" in s)
        and any(ch.isdigit() for ch in s)
    )

def filter_steps(steps):
    filtered = []
    for s in steps:
        sl = s.lower()

        # drop meta reasoning
        if any(p in sl for p in META_PATTERNS):
            continue

        # drop early brute-force numeric expansion
        if looks_like_bruteforce(sl):
            continue

        filtered.append(s)

    return filtered

# ===================== LLM CALL =====================
def call_llm(prompt: str, model_name: str = MODEL) -> str:
    # Allow dynamic model switching
    current_llm = ChatGroq(model=model_name, api_key=os.environ.get("GROQ_API_KEY"))
    
    try:
        out = current_llm.invoke(prompt)
        if hasattr(out, "content"):
            print(f"[DEBUG] call_llm content ({model_name}): {out.content[:100]}...")
            return out.content
        if isinstance(out, str):
            print(f"[DEBUG] call_llm str ({model_name}): {out[:100]}...")
            return out
        return str(out)
    except Exception as e:
        print(f"[DEBUG] call_llm failed for {model_name}: {e}")
        raise e

# ===================== PARSERS =====================
def _parse_json_subtasks(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    
    # aggressive validation cleaning
    # remove markdown code blocks
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("subtasks"), list):
            return [s.strip() for s in obj["subtasks"] if isinstance(s, str) and s.strip()]
    except Exception:
        pass

    m = re.search(r'\{[^}]*"subtasks"[^}]*\}', raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("subtasks"), list):
                return [s.strip() for s in obj["subtasks"] if isinstance(s, str) and s.strip()]
        except Exception:
            pass

    return None

def _lines_from_text(raw: str):
    if not raw:
        return []
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove numbering: "1. ", "1)", "- "
        line = re.sub(r'^[\d+\.\)\-\*]+\s+', '', line)
        if 3 <= len(line) <= 300:
            lines.append(line)
    return lines

# ===================== FALLBACKS =====================
def solver_fallback(query: str):
    # dynamic fallback based on query?
    return [
        f"Analyze the problem: {query}",
        "Break it down into smaller components",
        "Solve each component",
        "Combine the results"
    ]

# ===================== MAIN DECOMPOSER =====================
def decompose_query(query: str):
    intent = detect_intent(query)
    
    # Models to try in order
    models_to_try = [MODEL, "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    
    for model in models_to_try:
        print(f"[DEBUG] Trying decomposition with model: {model}")
        
        # 1. Try JSON Prompt
        try:
            raw = call_llm(DECOMP_PROMPT_JSON.format(query=query), model_name=model)
            parsed = _parse_json_subtasks(raw)
            if parsed:
                parsed = filter_steps(parsed)
                if parsed:
                    return parsed
        except Exception:
            pass
            
        # 2. Try Simple Prompt (List based)
        try:
            raw2 = call_llm(DECOMP_PROMPT_SIMPLE.format(query=query), model_name=model)
            lines = filter_steps(_lines_from_text(raw2))
            if len(lines) >= 2:
                return lines
        except Exception:
            pass

    print("[DEBUG] All models/prompts failed. Returning fallback.")
    return solver_fallback(query)

# ===================== PUBLIC API =====================
def run_decomposer(query: str):
    subtasks = decompose_query(query)
    print("Subtasks:")
    for i, s in enumerate(subtasks, 1):
        print(f" {i}. {s}")
    return subtasks

# ===================== ADAPTIVE RESPONSE =====================
def generate_adaptive_response(query: str, steps: list, ce_score: float, diagnostics: dict):
    # Select tone based on CE score
    if ce_score < 0.5:
        tone = "supportive, detailed, and encouraging. Focus on building foundational understanding."
    elif ce_score < 0.8:
        tone = "balanced and structured. Focus on reinforcing the logic."
    else:
        tone = "concise, advanced, and challenging. Focus on extension and mastery."

    prompt = f"""
    You are an AI Tutor.
    Student Goal: {query}
    Cognitive Efficiency Score: {ce_score:.2f} (Range 0-1)
    Diagnostics: {json.dumps(diagnostics)}
    
    The student has followed these steps:
    {json.dumps(steps, indent=2)}
    
    Generate a final "Learning Summary" for the student.
    Tone: {tone}
    
    Structure:
    1.  **Concept Check**: Briefly review the core concept used.
    2.  **Step-by-Step Walkthrough**: Synthesize the steps into a coherent narrative solution.
    3.  **Growth Tip**: Advice based on the diagnostics.
    
    Keep it within 300 words. Format with Markdown.
    """
    return call_llm(prompt)
