import json
import time
import uuid
from typing import Any, Dict, List, Union
from llm_utils import run_decomposer

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

VERIFICATION_MODEL = 'llama-3.1-8b-instant'
# Global instance removed/commented out to support dynamic instantiation
# llm_verify = ChatGroq(model=VERIFICATION_MODEL, api_key=os.environ.get("GROQ_API_KEY"))  


def _make_subtask_dict(id_: str = None, inp: str = "", out: str = "") -> Dict[str, str]:
    return {"id": id_ or str(uuid.uuid4()), "input": inp or "", "output": out or ""}


def normalize_subtasks(raw_subtasks: Union[str, Dict, List, Any]) -> List[Dict[str, str]]:
    if isinstance(raw_subtasks, list):
        normalized = []
        for item in raw_subtasks:
            if isinstance(item, dict):
                sid = item.get("id") or item.get("name") or str(uuid.uuid4())
                inp = item.get("input") if item.get("input") is not None else item.get("prompt", "")
                out = item.get("output") if item.get("output") is not None else item.get("result") or item.get("answer") or ""
                normalized.append(_make_subtask_dict(sid, str(inp), str(out)))
            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    inp, out = item
                    normalized.append(_make_subtask_dict(str(uuid.uuid4()), str(inp), str(out)))
                elif len(item) >= 3:
                    sid, inp, out = item[0], item[1], item[2]
                    normalized.append(_make_subtask_dict(str(sid), str(inp), str(out)))
                else:
                    normalized.append(_make_subtask_dict(None, "", str(item)))
            else:
                normalized.append(_make_subtask_dict(None, "", str(item)))
        return normalized

    if isinstance(raw_subtasks, dict):
        if all(not isinstance(v, (dict, list, tuple)) for v in raw_subtasks.values()):
            normalized = []
            for k, v in raw_subtasks.items():
                normalized.append(_make_subtask_dict(str(k), "", str(v)))
            return normalized
        else:
            sid = raw_subtasks.get("id") or raw_subtasks.get("name") or str(uuid.uuid4())
            inp = raw_subtasks.get("input") or raw_subtasks.get("prompt") or ""
            out = raw_subtasks.get("output") or raw_subtasks.get("result") or raw_subtasks.get("answer") or ""
            return [_make_subtask_dict(sid, str(inp), str(out))]

    return [_make_subtask_dict(None, "", str(raw_subtasks))]


def extract_json_from_text(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            pass
    s = text.find("[")
    e = text.rfind("]")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            pass
    raise ValueError("No JSON found in verifier output.")


# --- compact verifier prompt ---
def build_quick_verifier_prompt(original_prompt: str, subtasks: List[Dict[str, Any]], domain_hint: str = "") -> str:
    brief = ""
    for i, s in enumerate(subtasks, start=1):
        text = s.get("input") or s.get("output") or ""
        brief += f"{i}. {text}\n"

    prompt = (
        "You are a fast verifier. Given the original task and the list of subtasks below, answer in JSON only:\n"
        " - 'accept_all': boolean, are these subtasks sufficient to solve the main task?\n"
        " - 'missing': short list of missing concepts or checks (if any)\n"
        " - 'suggestions': short list of short suggested subtasks to add (if any)\n\n"
        f"Domain: {domain_hint}\n"
        f"Original task: {original_prompt}\n"
        f"Subtasks:\n{brief}\n"
        "Return EXACT JSON like: {\"accept_all\": true, \"missing\": [], \"suggestions\": []}\n"
        "Output only JSON."
    )
    return prompt


def get_verifier_llm(api_config: dict = None):
    provider = "Groq"
    api_key = None
    if api_config:
        provider = api_config.get("verifier_provider", "Groq")
        if provider == "Groq":
            api_key = api_config.get("groq_api_key")
        elif provider == "Gemini":
            api_key = api_config.get("gemini_api_key")
    
    # Fallback
    if not api_key:
        if provider == "Groq":
             api_key = os.environ.get("GROQ_API_KEY")
        elif provider == "Gemini":
             api_key = os.environ.get("GOOGLE_API_KEY")

    # Determine model
    if provider == "Gemini":
        model_name = "gemini-2.5-flash" # default
    else:
        model_name = VERIFICATION_MODEL # default
        
    if api_config and api_config.get("verifier_model"):
        model_name = api_config.get("verifier_model")

    if provider == "Gemini":
         if not api_key:
              api_key = os.environ.get("GOOGLE_API_KEY")
         return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    else:
         if not api_key:
              api_key = os.environ.get("GROQ_API_KEY")
         return ChatGroq(model=model_name, api_key=api_key)


# --- fast verifier call ---
def fast_verify_subtasks(original_prompt: str, raw_subtasks: Union[List[Any], Any], domain_hint: str = "", api_config: dict = None) -> Dict[str, Any]:
    subtasks = normalize_subtasks(raw_subtasks)
    prompt = build_quick_verifier_prompt(original_prompt, subtasks, domain_hint)
    
    try:
        llm_verify = get_verifier_llm(api_config)
        raw_msg = llm_verify.invoke(prompt)
        raw = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
    except Exception as e:
        return {
             "accept_all": False,
             "missing": ["verifier_error"],
             "suggestions": [f"Verifier failed: {str(e)}"],
             "subtasks": subtasks,
             "raw_verifier": str(e)
        }

    try:
        parsed = extract_json_from_text(raw)
    except Exception:
        return {
            "accept_all": False,
            "missing": ["verifier_parse_failed"],
            "suggestions": ["Please retry verification or use a different verifier model."],
            "subtasks": subtasks,
            "raw_verifier": raw
        }

    accept_all = bool(parsed.get("accept_all")) if isinstance(parsed, dict) else False
    missing = parsed.get("missing") if isinstance(parsed, dict) else []
    suggestions = parsed.get("suggestions") if isinstance(parsed, dict) else []

    return {
        "accept_all": accept_all,
        "missing": missing or [],
        "suggestions": suggestions or [],
        "subtasks": subtasks,
        "raw_verifier": parsed
    }


# --- updated run_inference_pipeline_fast that always returns final_subtasks ---
def run_inference_pipeline(
    prompt: str,
    domain_hint: str = "",
    auto_extend: bool = False,
    keep_suggestions_field: bool = True,
    api_config: dict = None
) -> Dict[str, Any]:
    """
    Returns a dict containing:
      - status: 'accepted'|'insufficient'
      - final_subtasks: canonical forwardable subtasks (possibly extended if auto_extend=True)
      - suggestions, missing, raw_verifier
    """
    raw = run_decomposer(prompt, api_config=api_config)
    print(f"[DEBUG] run_decomposer returned type={type(raw)}, preview={str(raw)[:300]}")

    check = fast_verify_subtasks(prompt, raw, domain_hint=domain_hint, api_config=api_config)

    canonical = check["subtasks"]  # list of canonical subtask dicts

    final_subtasks = [dict(s) for s in canonical]  # shallow copy

    if check["accept_all"]:
        status = "accepted"
        message = "Subtasks sufficient. Forwarding inference."
    else:
        status = "insufficient"
        message = "Verifier flagged missing points or insufficiencies."
        # If auto_extend, append suggestion strings as new subtasks with empty output
        if auto_extend and check.get("suggestions"):
            for s in check["suggestions"]:
                final_subtasks.append(_make_subtask_dict(None, s, ""))
        # else leave final_subtasks as canonical (caller can handle suggestions)

    result = {
        "status": status,
        "message": message,
        "final_subtasks": final_subtasks,
        "timestamp": time.time(),
        "raw_verifier": check.get("raw_verifier")
    }

    if keep_suggestions_field:
        result["suggestions"] = check.get("suggestions", [])
        result["missing"] = check.get("missing", [])

    return result