import json
import re
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        ChatOpenAI = None
from dotenv import load_dotenv
import os

load_dotenv()

# ===================== MODEL =====================
MODEL = "llama-3.3-70b-versatile"
# Default fallback to environment variable if no config passed
# llm = ChatGroq(model=MODEL, api_key=os.environ.get("GROQ_API_KEY"))

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

DECOMP_PROMPT_JSON = """You are an Expert Problem Solver using Chain-of-Thought reasoning.
Your goal is to break down the complex problem into a logical sequence of EXECUTABLE steps.

Rules:
1.  **No Direct Solution**: Do not solve the problem yet. Just list the steps.
2.  **Granularity**: Each step must be a single, distinct action (e.g., "Identify...", "Calculate...", "Compare...").
3.  **Logical Flow**: Steps must follow a strict dependency order.
4.  **Verification**: Include a final step to verify the result.
5.  **Output**: Return a JSON object with a "subtasks" key containing the list of strings.

Example:
User: "Calculate the kinetic energy of a 5kg mass moving at 10m/s."
Result:
{{
  "subtasks": [
    "Identify the given mass (m) and velocity (v) from the problem statement.",
    "Recall the formula for kinetic energy: KE = 0.5 * m * v^2.",
    "Substitute the values of m = 5 kg and v = 10 m/s into the formula.",
    "Calculate the square of the velocity (10^2).",
    "Multiply the mass by the squared velocity.",
    "Multiply the result by 0.5 to get the final kinetic energy.",
    "Verify the units are in Joules."
  ]
}}

User query:
{query}

Respond in STRICT JSON format.
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
DECOMP_PROMPT_LEARNING = """You are an Expert Curriculum Designer creating a DEEP DIVE LEARNING ROADMAP.
User intent: detailed acquisition of knowledge on a specific topic.

Goal: Break down the learning process into granular, actionable, and comprehensive steps.
The user wants to know EXACTLY how to go from zero to hero on this topic.

Rules:
1.  **Granularity**: Do not give high-level abstract steps like "Learn Python". Break it down: "Master Python control flow (if/else, loops) and basic data structures (lists, dicts)."
2.  **Resources & Actions**: For each step, suggest *how* to learn it (e.g., "Read documentation on X", "Implement a small script to do Y", "Watch a visualization of Z").
3.  **Logical Flow**:
    *   **Phase 1: Conceptual Foundations** (Prerequisites, mental models, basic definitions)
    *   **Phase 2: Core Mechanics** (How it works deeply, key theorems/laws/functions)
    *   **Phase 3: Practical Application** (How to use it, real-world examples, solving standard problems)
    *   **Phase 4: Advanced/Nuance** (Edge cases, limitations, conflicting theories)
4.  **No Meta-Fluff**: Avoid generic steps like "Analyze the problem". Every step must be a learning action.

Example Output format (List of strings):
1.  "Foundations: Study the history of [Topic] to understand the motivation behind its invention (e.g., Resource X or Y)."
2.  "Core Concept: Master the specific mathematical definition of [Sub-topic], specifically focusing on [Detail]."
3.  "Practice: Solve 5 introductory problems involving [Scenario] to solidify intuition."
4.  "Deep Dive: Read the seminal paper/chapter on [Advanced Aspect] to understand the theoretical limits."

User query:
'''{query}'''

Return a list of 6-10 highly detailed, executable learning actions.
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
def get_llm_instance(model_name: str, api_config: dict = None):
    """
    Factory to return the appropriate LLM instance based on config.
    """
    provider = "Groq"
    api_key = None
    
    if api_config:
        provider = api_config.get("main_provider", "Groq")
        if provider == "Groq":
            api_key = api_config.get("groq_api_key")
        elif provider == "Gemini":
            api_key = api_config.get("gemini_api_key")
        elif provider == "OpenAI":
            api_key = api_config.get("openai_api_key")
    
    # Fallback to env if specific key not provided
    if not api_key:
        if provider == "Groq":
            api_key = os.environ.get("GROQ_API_KEY")
        elif provider == "Gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
        elif provider == "OpenAI":
            api_key = os.environ.get("OPENAI_API_KEY")

    if provider == "Gemini":
        if not api_key:
            print("[WARN] Gemini selected but no API key provided. Trying env GOOGLE_API_KEY.")
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        # Use gemini-2.5-flash as requested
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    elif provider == "OpenAI":
        if not api_key:
             api_key = os.environ.get("OPENAI_API_KEY")
        if not ChatOpenAI:
             raise ImportError("LangChain OpenAI integration not found. Please install `langchain-openai` or `langchain`.")
        return ChatOpenAI(model=model_name, api_key=api_key)
    else:
        # Groq
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        return ChatGroq(model=model_name, api_key=api_key)


def call_llm(prompt: str, model_name: str = MODEL, api_config: dict = None) -> str:
    # Allow dynamic model switching
    try:
        current_llm = get_llm_instance(model_name, api_config)
    except Exception as e:
         print(f"[ERROR] Failed to instantiate LLM: {e}")
         return str(e)

    try:
        out = current_llm.invoke(prompt)
        if hasattr(out, "content"):
            print(f"[DEBUG] call_llm content ({model_name} / {current_llm.__class__.__name__}): {out.content[:100]}...")
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
        # Increased max length to 500 to allow for detailed learning steps
        if 3 <= len(line) <= 500:
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

def check_connection(provider: str, api_key: str, model_name: str = None) -> bool:
    """
    Verifies connectivity to the LLM provider with the given key.
    Returns True if a simple LLM call succeeds, False otherwise.
    Uses BaseException to catch ALL errors and prevent server crashes.
    """
    if not api_key:
        return False
    
    try:
        config = {
            "main_provider": provider,
            "groq_api_key": api_key if provider == "Groq" else None,
            "gemini_api_key": api_key if provider == "Gemini" else None,
            "openai_api_key": api_key if provider == "OpenAI" else None
        }
        
        # Use a lightweight/common model for check if not specified
        if not model_name:
            if provider == "Groq":
                model_name = "llama-3.1-8b-instant"
            elif provider == "Gemini":
                model_name = "gemini-2.5-flash"
            elif provider == "OpenAI":
                model_name = "gpt-3.5-turbo"

        llm = get_llm_instance(model_name, api_config=config)
        # Simple invocation to verify the key works
        llm.invoke("Hello")
        return True
    except BaseException as e:
        print(f"[ERROR] Connection check failed for {provider}: {e}")
        return False

# ===================== MAIN DECOMPOSER =====================
def decompose_query(query: str, api_config: dict = None):
    intent = detect_intent(query)
    
    # Determine which model to use as primary
    primary_model = MODEL # default
    if api_config and "main_model" in api_config:
         primary_model = api_config["main_model"]
    
    # Models to try in order: User Selected -> Llama 3 -> Mixtral
    models_to_try = [primary_model]
    if primary_model != "llama-3.1-8b-instant":
         models_to_try.append("llama-3.1-8b-instant")
    if primary_model != "mixtral-8x7b-32768":
         models_to_try.append("mixtral-8x7b-32768")
    
    for model in models_to_try:
        print(f"[DEBUG] Trying decomposition with model: {model} | Intent: {intent}")
        
        # 0. If Learning Intent, try Learning Prompt first
        if intent == "learning":
            try:
                raw_learn = call_llm(DECOMP_PROMPT_LEARNING.format(query=query), model_name=model, api_config=api_config)
                lines = filter_steps(_lines_from_text(raw_learn))
                if len(lines) >= 3:
                    return lines
            except Exception as e:
                print(f"[DEBUG] Learning prompt failed: {e}")
                pass

        # 1. Try JSON Prompt (Standard Solver)
        try:
            raw = call_llm(DECOMP_PROMPT_JSON.format(query=query), model_name=model, api_config=api_config)
            parsed = _parse_json_subtasks(raw)
            if parsed:
                parsed = filter_steps(parsed)
                if parsed:
                    return parsed
        except Exception:
            pass
            
        # 2. Try Simple Prompt (List based fallback)
        try:
            raw2 = call_llm(DECOMP_PROMPT_SIMPLE.format(query=query), model_name=model, api_config=api_config)
            lines = filter_steps(_lines_from_text(raw2))
            if len(lines) >= 2:
                return lines
        except Exception:
            pass

    print("[DEBUG] All models/prompts failed. Returning fallback.")
    return solver_fallback(query)

# ===================== PUBLIC API =====================
def run_decomposer(query: str, api_config: dict = None):
    subtasks = decompose_query(query, api_config=api_config)
    print("Subtasks:")
    for i, s in enumerate(subtasks, 1):
        print(f" {i}. {s}")
    return subtasks

# ===================== ADAPTIVE RESPONSE =====================
def generate_adaptive_response(query: str, steps: list, ce_score: float, diagnostics: dict, api_config: dict = None):
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
    return call_llm(prompt, api_config=api_config)


# ===================== SOCRATIC FOLLOW-UP =====================
SOCRATIC_SYSTEM = """You are a Socratic mentor. Your role is to help the student discover the answer themselves through questions.

RULES — follow these strictly:
1. You must NEVER directly answer the student's question.
2. Instead, ask ONE thoughtful probing question that guides their thinking.
3. Base your question on the learning steps they have been given.
4. Keep it concise (1-2 sentences).
5. If the student seems stuck, give a small nudge/hint framed as a question.
6. Never say "Great question!" or similar filler — go straight to your probing question.
"""

def generate_socratic_followup(user_message: str, context: dict, history: list, api_config: dict = None):
    """
    Generate a Socratic follow-up response. Never directly answers — only asks guiding questions.
    Returns (response_text, prompt_tokens_est, completion_tokens_est).
    """
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(context.get("steps", [])))
    ce_score = context.get("ce_score", 0.5)

    history_text = ""
    for msg in history[-6:]:  # keep last 6 turns for context
        role = "Student" if msg["role"] == "user" else "Mentor"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""{SOCRATIC_SYSTEM}

Context:
- Student's original question: {context.get("prompt", "")}
- CE Score: {ce_score:.2f}
- Learning steps the student received:
{steps_text}

Conversation so far:
{history_text}
Student: {user_message}

Respond with ONE probing question (1-2 sentences):"""

    response = call_llm(prompt, api_config=api_config)

    # Estimate tokens (len/4 heuristic)
    prompt_tokens = max(1, int(len(prompt) / 4))
    completion_tokens = max(1, int(len(response) / 4))

    return response, prompt_tokens, completion_tokens


# ===================== LEARNING PLAN GENERATOR =====================
def generate_learning_plan(query: str, steps: list, ce_score: float, diagnostics: dict, api_config: dict = None):
    """
    Generate a structured learning/action plan in Markdown.
    Returns (plan_markdown, prompt_tokens_est, completion_tokens_est).
    """
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))

    prompt = f"""You are an Expert Learning Architect. Based on the student's session below, generate a structured **Action Plan** in Markdown.

Student's Question: {query}
Cognitive Efficiency Score: {ce_score:.2f}
Diagnostics: {json.dumps(diagnostics)}

Learning Steps Already Generated:
{steps_text}

Create a plan with this EXACT structure:

# 🗺 Learning Action Plan

## Phase 1: Foundation (Days 1-3)
- Goal: ...
- Tasks:
  - [ ] Task 1
  - [ ] Task 2
- Resources: ...

## Phase 2: Core Understanding (Days 4-7)
- Goal: ...
- Tasks:
  - [ ] Task 1
  - [ ] Task 2
- Resources: ...

## Phase 3: Practice & Application (Days 8-12)
- Goal: ...
- Tasks:
  - [ ] Task 1
  - [ ] Task 2
- Resources: ...

## Phase 4: Mastery & Review (Days 13-14)
- Goal: ...
- Tasks:
  - [ ] Task 1
  - [ ] Task 2

## 🎯 Milestones
| Milestone | Target | Metric |
|-----------|--------|--------|
| ... | ... | ... |

Keep the plan actionable, specific to the topic, and within 400 words. Format in clean Markdown.
"""
    response = call_llm(prompt, api_config=api_config)

    prompt_tokens = max(1, int(len(prompt) / 4))
    completion_tokens = max(1, int(len(response) / 4))

    return response, prompt_tokens, completion_tokens
