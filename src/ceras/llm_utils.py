import json
import re
from langchain.llms import Ollama

# ===================== MODEL =====================
MODEL = "llama3.2"
llm = Ollama(model=MODEL)

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

DECOMP_PROMPT_JSON = f"""You are generating EXECUTABLE SOLUTION STEPS.

{SOLVER_PREFERENCE_RULES}

Rules (MANDATORY):
- Each step MUST directly act on objects, symbols, or entities in the user query
- Steps must transform the problem toward a solution
- NO meta reasoning
- NO planning
- NO classification
- NO research
- NO explanation
- NO commentary

FORBIDDEN STEP TYPES (DO NOT OUTPUT):
- "Identify the type of problem"
- "Determine relevance"
- "Research concepts"
- "Choose an approach"
- "Understand the question"
- "High-level plan"

GOOD STEP EXAMPLES:
- "Rewrite the expression using a suitable identity"
- "Substitute the given values into the rewritten expression"
- "Evaluate the resulting expression"

RETURN STRICT JSON ONLY, EXACTLY:
{{"subtasks": ["step 1", "step 2", "..."]}}

User query:
\"\"\"{{query}}\"\"\"
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
def call_llm(prompt: str) -> str:
    out = llm(prompt)
    if isinstance(out, str):
        return out
    if hasattr(out, "text"):
        return out.text
    return str(out)

# ===================== PARSERS =====================
def _parse_json_subtasks(raw: str):
    if not raw:
        return None
    raw = raw.strip()

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
        line = re.sub(r'^[0-9]+[)\.\-\s]+', '', line)
        line = re.sub(r'^[-*\u2022]\s*', '', line)
        if 3 <= len(line) <= 300:
            lines.append(line)
    return lines

# ===================== FALLBACKS =====================
def solver_fallback(query: str):
    return [
        "Rewrite the problem into a simpler equivalent form",
        "Substitute the given values into the rewritten form",
        "Perform the required operations to obtain the result"
    ]

def learning_fallback():
    return [
        "Understand the core purpose of the topic",
        "Learn foundational concepts and terminology",
        "Study official documentation and examples",
        "Explore advanced or related tools",
        "Build a small practical example"
    ]

# ===================== MAIN DECOMPOSER =====================
def decompose_query(query: str):
    intent = detect_intent(query)

    # ---------- LEARNING MODE ----------
    if intent == "learning":
        try:
            raw = call_llm(DECOMP_PROMPT_LEARNING.format(query=query))
        except Exception:
            raw = ""

        steps = _lines_from_text(raw)
        steps = filter_steps(steps)
        return steps if steps else learning_fallback()

    # ---------- SOLVING MODE ----------
    try:
        raw = call_llm(DECOMP_PROMPT_JSON.format(query=query))
    except Exception:
        raw = ""

    parsed = _parse_json_subtasks(raw)
    if parsed:
        parsed = filter_steps(parsed)
        if parsed:
            return parsed

    try:
        raw2 = call_llm(DECOMP_PROMPT_SIMPLE.format(query=query))
    except Exception:
        raw2 = ""

    lines = filter_steps(_lines_from_text(raw2))
    if len(lines) >= 2:
        return lines

    try:
        raw3 = call_llm(DECOMP_PROMPT_MIN.format(query=query))
    except Exception:
        raw3 = ""

    lines3 = filter_steps(_lines_from_text(raw3))
    if lines3:
        return lines3

    return solver_fallback(query)

# ===================== PUBLIC API =====================
def run_decomposer(query: str):
    subtasks = decompose_query(query)
    print("Subtasks:")
    for i, s in enumerate(subtasks, 1):
        print(f" {i}. {s}")
    return subtasks
