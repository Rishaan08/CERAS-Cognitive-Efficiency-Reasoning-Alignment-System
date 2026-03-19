"""
CERAS FastAPI Backend Server
Wraps existing Python pipeline, ML models, and LLM utils as REST API endpoints.
Models are loaded lazily in a background thread so the frontend loads instantly.
"""

import os
import sys
import time
import re
import json
import threading
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from collections import deque
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --------------- PATH SETUP ---------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src" / "ceras"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ASSET_DIR = BASE_DIR / "assets"

# Add src/ceras to path so we can import pipeline modules
sys.path.insert(0, str(SRC_DIR))

# --------------- LOGGING ---------------
LOG_BUFFER_SIZE = 500
log_buffer = deque(maxlen=LOG_BUFFER_SIZE)
log_buffer_lock = threading.Lock()


class InMemoryLogHandler(logging.Handler):
    def emit(self, record):
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                formatter = self.formatter or logging.Formatter()
                entry["exception"] = formatter.formatException(record.exc_info)
            with log_buffer_lock:
                log_buffer.append(entry)
        except Exception:
            self.handleError(record)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ceras-server")
memory_log_handler = InMemoryLogHandler()
memory_log_handler.setLevel(logging.INFO)
logger.addHandler(memory_log_handler)
root_logger = logging.getLogger()
handler_types = {type(handler) for handler in root_logger.handlers}
if type(memory_log_handler) not in handler_types:
    root_logger.addHandler(memory_log_handler)


def _log_event(level: str, message: str, **extra):
    payload = {"event": message}
    payload.update(extra)
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(payload, default=str))

# --------------- APP ---------------
app = FastAPI(title="CERAS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or str(uuid4())
    start = time.time()

    _log_event(
        "info",
        "request_started",
        trace_id=trace_id,
        method=request.method,
        path=request.url.path,
        query=str(request.url.query),
        client=getattr(request.client, "host", None),
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.time() - start) * 1000, 2)
        _log_event(
            "error",
            "request_failed",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise

    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Trace-Id"] = trace_id
    _log_event(
        "info",
        "request_completed",
        trace_id=trace_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response

# --------------- MODEL STATE ---------------
model_state = {
    "loaded": False,
    "loading": False,
    "error": None,
    "cepm_model": None,
    "cepm_scaler": None,
    "cnn_model": None,
    "cnn_scaler": None,
    "cepm_features": None,
    "cnn_features": None,
}


def _load_models_background():
    """Load ML models in a background thread."""
    import joblib
    import tensorflow as tf

    model_state["loading"] = True
    logger.info("⏳ Loading ML models in background...")
    try:
        model_state["cepm_model"] = joblib.load(str(ARTIFACT_DIR / "cepm_lightgbm.pkl"))
        model_state["cepm_scaler"] = joblib.load(str(ARTIFACT_DIR / "cepm_scaler.pkl"))
        model_state["cnn_model"] = tf.keras.models.load_model(str(ARTIFACT_DIR / "cnn_ce_model.keras"))
        model_state["cnn_scaler"] = joblib.load(str(ARTIFACT_DIR / "cnn_scaler.pkl"))
        model_state["cepm_features"] = np.load(str(ARTIFACT_DIR / "cepm_features.npy"), allow_pickle=True).tolist()
        model_state["cnn_features"] = np.load(str(ARTIFACT_DIR / "cnn_features.npy"), allow_pickle=True).tolist()
        model_state["loaded"] = True
        model_state["error"] = None
        logger.info("✅ All ML models loaded successfully.")
    except Exception as e:
        model_state["error"] = str(e)
        logger.error(f"❌ Model loading failed: {e}")
    finally:
        model_state["loading"] = False


@app.on_event("startup")
def startup_event():
    _log_event("info", "startup_models_loading_scheduled")
    thread = threading.Thread(target=_load_models_background, daemon=True)
    thread.start()


# --------------- FEATURE EXTRACTION ---------------
def extract_ceras_features(prompt_text: str) -> dict:
    words = prompt_text.split()
    prompt_length = int(np.clip(len(words), 1, 400))
    character_count = len(prompt_text)
    sentence_count = max(len(re.findall(r"[.!?]", prompt_text)), 1)
    unique_word_ratio = float(np.clip(len(set(words)) / (prompt_length + 1e-6), 0, 1))
    concept_density = float(np.clip(sum(1 for w in words if len(w) > 6) / (prompt_length + 1e-6), 0, 1))
    keystrokes = int(np.clip(character_count, 1, 2000))
    prompt_quality = float(np.clip(prompt_length / 150, 0, 1))

    if prompt_length < 20:
        prompt_type = 0
    elif prompt_length < 60:
        prompt_type = 1
    elif prompt_length < 120:
        prompt_type = 2
    else:
        prompt_type = 3

    return {
        "prompt_length": float(prompt_length),
        "sentence_count": float(sentence_count),
        "unique_word_ratio": unique_word_ratio,
        "concept_density": concept_density,
        "prompt_quality": prompt_quality,
        "character_count": float(character_count),
        "keystrokes": float(keystrokes),
        "prompt_type": float(prompt_type),
    }


# --------------- REQUEST / RESPONSE MODELS ---------------
class CheckConnectionRequest(BaseModel):
    provider: str
    api_key: str

class RunSessionRequest(BaseModel):
    prompt: str
    main_provider: str = "Groq"
    verifier_provider: str = "Groq"
    main_model: Optional[str] = None
    verifier_model: Optional[str] = None
    groq_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    formulation_time: Optional[float] = 0.0

class AdaptiveResponseRequest(BaseModel):
    prompt: str
    steps: List[str]
    ce_score: float
    diagnostics: Dict[str, Any]
    main_provider: str = "Groq"
    main_model: Optional[str] = None
    groq_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""


class FollowUpRequest(BaseModel):
    message: str
    context: Dict[str, Any]  # {prompt, steps, ce_score}
    history: List[Dict[str, str]] = []  # [{role, content}]
    main_provider: str = "Groq"
    main_model: Optional[str] = None
    groq_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""


class GeneratePlanRequest(BaseModel):
    prompt: str
    steps: List[str]
    ce_score: float
    diagnostics: Dict[str, Any]
    main_provider: str = "Groq"
    main_model: Optional[str] = None
    groq_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""


# --------------- TOKEN COST HELPER ---------------
# Rates in USD per 1M tokens: (input_rate, output_rate)
_COST_RATES = {
    "Groq":   (0.59,  0.79),    # Llama 3.3-70B on Groq
    "Gemini": (0.075, 0.30),    # Gemini 2.5 Flash
    "OpenAI": (0.15,  0.60),    # GPT-4o-mini
}

def _estimate_cost(prompt_tokens: int, completion_tokens: int, provider: str) -> float:
    inp_rate, out_rate = _COST_RATES.get(provider, (0.59, 0.79))
    return round((prompt_tokens * inp_rate + completion_tokens * out_rate) / 1_000_000, 8)


# --------------- ENDPOINTS ---------------

@app.get("/health")
@app.get("/api/health")
def health():
    _log_event(
        "info",
        "health_checked",
        models_loaded=model_state["loaded"],
        models_loading=model_state["loading"],
        has_error=bool(model_state["error"]),
    )
    return {
        "status": "ok",
        "models_loaded": model_state["loaded"],
        "models_loading": model_state["loading"],
        "model_error": model_state["error"],
        "timestamp": time.time(),
    }


@app.get("/api/logo")
def get_logo():
    logo_path = ASSET_DIR / "ceras_logo.png"
    if logo_path.exists():
        _log_event("info", "logo_served")
        return FileResponse(str(logo_path), media_type="image/png")
    _log_event("warning", "logo_missing", path=str(logo_path))
    raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/api/logs")
def get_logs(limit: int = 100, level: Optional[str] = None, contains: Optional[str] = None):
    safe_limit = max(1, min(limit, 500))

    with log_buffer_lock:
        entries = list(log_buffer)

    if level:
        level_upper = level.upper()
        entries = [entry for entry in entries if entry["level"] == level_upper]

    if contains:
        needle = contains.lower()
        entries = [
            entry for entry in entries
            if needle in entry["message"].lower() or needle in entry["logger"].lower()
        ]

    sliced_entries = entries[-safe_limit:]
    _log_event(
        "info",
        "logs_requested",
        requested_limit=limit,
        applied_limit=safe_limit,
        filter_level=level,
        contains=contains,
        returned=len(sliced_entries),
    )
    return {
        "count": len(sliced_entries),
        "total_buffered": len(entries),
        "buffer_capacity": LOG_BUFFER_SIZE,
        "logs": sliced_entries,
    }


@app.post("/api/check-connection")
def check_connection_endpoint(req: CheckConnectionRequest):
    _log_event("info", "connection_check_started", provider=req.provider)
    try:
        from llm_utils import check_connection
        result = check_connection(req.provider, req.api_key)
        _log_event("info", "connection_check_completed", provider=req.provider, connected=bool(result))
        return {"connected": result}
    except BaseException as e:
        logger.error(f"Connection check failed for {req.provider}: {e}")
        return {"connected": False, "error": str(e)}


@app.post("/api/run-session")
def run_session(req: RunSessionRequest):
    if not model_state["loaded"]:
        _log_event("warning", "run_session_blocked_models_loading")
        raise HTTPException(status_code=503, detail="Models are still loading. Please wait.")

    from pipeline_1 import main as run_infer
    from fusion import CERASFusion

    api_config = {
        "main_provider": req.main_provider,
        "verifier_provider": req.verifier_provider,
        "groq_api_key": req.groq_api_key,
        "gemini_api_key": req.gemini_api_key,
        "openai_api_key": req.openai_api_key,
        "main_model": req.main_model,
        "verifier_model": req.verifier_model,
    }

    _log_event(
        "info",
        "run_session_started",
        main_provider=req.main_provider,
        verifier_provider=req.verifier_provider,
        main_model=req.main_model,
        verifier_model=req.verifier_model,
        prompt_chars=len(req.prompt or ""),
    )

    t0 = time.time()
    result = run_infer(req.prompt, api_config=api_config)
    runtime = time.time() - t0
    _log_event(
        "info",
        "run_session_pipeline_completed",
        runtime_ms=round(runtime * 1000, 2),
        llm_calls_used=result.get("llm_calls_used", 0),
        strategy_used=result.get("strategy_used", ""),
    )

    final_steps = result.get("final_answer", [])
    features = extract_ceras_features(req.prompt)

    # CEPM Inference
    cepm_input = np.array([features[f] for f in model_state["cepm_features"]]).reshape(1, -1)
    cepm_input_scaled = model_state["cepm_scaler"].transform(cepm_input)
    cepm_score = float(np.clip(model_state["cepm_model"].predict(cepm_input_scaled)[0], 0, 1))

    # CNN Inference
    cnn_input = np.array([features[f] for f in model_state["cnn_features"]]).reshape(1, -1)
    cnn_input = model_state["cnn_scaler"].transform(cnn_input)
    if len(model_state["cnn_model"].input_shape) == 3:
        cnn_input = cnn_input.reshape(cnn_input.shape[0], cnn_input.shape[1], 1)
    cnn_score = float(np.clip(np.squeeze(model_state["cnn_model"].predict(cnn_input, verbose=0)), 0, 1))

    # Fusion
    fusion_engine = CERASFusion()
    fusion_df = fusion_engine.fuse(
        session_ids=["session_1"],
        cepm_scores=[cepm_score],
        cnn_scores=[cnn_score],
    )

    fused_score = float(fusion_df["fused_ce_score"].iloc[0])
    confidence = float(fusion_df["confidence"].iloc[0])
    diagnostics = fusion_df["diagnostics"].iloc[0]
    readiness = fusion_df["readiness_label"].iloc[0]

    # Token estimation
    est_prompt_tokens = int(len(req.prompt) / 4)
    est_response_tokens = int(len(str(final_steps)) / 4)
    total_tokens = est_prompt_tokens + est_response_tokens

    # Diagnostic logic
    strengths = []
    suggestions = []
    if cepm_score > 0.75:
        strengths.append("Strong structural complexity and adequate length.")
    else:
        suggestions.append("Try adding more specific constraints or context to increase structural density.")
    if cnn_score > 0.75:
        strengths.append("High semantic clarity; intent matches known high-performing patterns.")
    else:
        suggestions.append("Clarify the core intent. Use precise domain terminology to improve semantic alignment.")
    if not strengths:
        strengths.append("Prompt is functional but has room for optimization across all dimensions.")
    if not suggestions:
        suggestions.append("Excellent prompt! Maintains high cognitive efficiency.")

    _log_event(
        "info",
        "run_session_completed",
        runtime_ms=round(runtime * 1000, 2),
        total_tokens=total_tokens,
        llm_calls_used=result.get("llm_calls_used", 0),
        cepm_score=round(cepm_score, 4),
        cnn_score=round(cnn_score, 4),
        fused_score=round(fused_score, 4),
        readiness=readiness,
    )

    return {
        "final_steps": final_steps if isinstance(final_steps, list) else [str(final_steps)],
        "strategy_used": result.get("strategy_used", ""),
        "llm_calls_used": result.get("llm_calls_used", 0),
        "tree": result.get("tree"),
        "logs": result.get("logs", ""),
        "runtime": runtime,
        "formulation_time": req.formulation_time,
        "features": features,
        "feature_count": len(features),
        "total_tokens": total_tokens,
        "cepm_score": cepm_score,
        "cnn_score": cnn_score,
        "fused_score": fused_score,
        "confidence": confidence,
        "diagnostics": diagnostics,
        "readiness": readiness,
        "strengths": strengths,
        "suggestions": suggestions,
    }


@app.post("/api/adaptive-response")
def adaptive_response(req: AdaptiveResponseRequest):
    from llm_utils import generate_adaptive_response

    api_config = {
        "main_provider": req.main_provider,
        "verifier_provider": req.main_provider,
        "groq_api_key": req.groq_api_key,
        "gemini_api_key": req.gemini_api_key,
        "openai_api_key": req.openai_api_key,
        "main_model": req.main_model,
    }

    try:
        _log_event(
            "info",
            "adaptive_response_started",
            main_provider=req.main_provider,
            main_model=req.main_model,
            steps_count=len(req.steps),
            ce_score=round(req.ce_score, 4),
        )
        response = generate_adaptive_response(
            req.prompt,
            req.steps,
            req.ce_score,
            req.diagnostics,
            api_config=api_config,
        )
        _log_event("info", "adaptive_response_completed", response_chars=len(response or ""))
        return {"response": response}
    except Exception as e:
        _log_event("error", "adaptive_response_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --------------- NEW: FILE PARSING ---------------
@app.post("/api/parse-file")
async def parse_file(file: UploadFile = File(...)):
    """Extract text from uploaded PDF, DOCX, TXT, or CSV files."""
    filename = (file.filename or "").lower()
    content = await file.read()
    _log_event(
        "info",
        "file_parse_started",
        filename=file.filename,
        size_bytes=len(content),
    )

    try:
        if filename.endswith(".pdf"):
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif filename.endswith(".docx"):
            import docx
            import io
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)

        elif filename.endswith(".csv"):
            text = content.decode("utf-8", errors="replace")

        elif filename.endswith(".txt") or filename.endswith(".md"):
            text = content.decode("utf-8", errors="replace")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

        # Truncate to ~8000 chars to keep context manageable
        if len(text) > 8000:
            text = text[:8000] + "\n... [truncated]"

        _log_event(
            "info",
            "file_parse_completed",
            filename=file.filename,
            chars=len(text.strip()),
        )
        return {"text": text.strip(), "filename": file.filename, "chars": len(text.strip())}

    except HTTPException:
        _log_event("warning", "file_parse_rejected", filename=file.filename)
        raise
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        _log_event("error", "file_parse_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")


# --------------- NEW: SOCRATIC FOLLOW-UP ---------------
@app.post("/api/followup")
def followup_chat(req: FollowUpRequest):
    from llm_utils import generate_socratic_followup

    api_config = {
        "main_provider": req.main_provider,
        "verifier_provider": req.main_provider,
        "groq_api_key": req.groq_api_key,
        "gemini_api_key": req.gemini_api_key,
        "openai_api_key": req.openai_api_key,
        "main_model": req.main_model,
    }

    try:
        _log_event(
            "info",
            "followup_started",
            main_provider=req.main_provider,
            main_model=req.main_model,
            history_count=len(req.history),
            message_chars=len(req.message or ""),
        )
        response, prompt_tokens, completion_tokens = generate_socratic_followup(
            user_message=req.message,
            context=req.context,
            history=req.history,
            api_config=api_config,
        )
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = _estimate_cost(prompt_tokens, completion_tokens, req.main_provider)

        _log_event(
            "info",
            "followup_completed",
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            response_chars=len(response or ""),
        )

        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }
    except Exception as e:
        logger.error(f"Follow-up error: {e}")
        _log_event("error", "followup_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --------------- NEW: LEARNING PLAN GENERATOR ---------------
@app.post("/api/generate-plan")
def generate_plan(req: GeneratePlanRequest):
    from llm_utils import generate_learning_plan

    api_config = {
        "main_provider": req.main_provider,
        "verifier_provider": req.main_provider,
        "groq_api_key": req.groq_api_key,
        "gemini_api_key": req.gemini_api_key,
        "openai_api_key": req.openai_api_key,
        "main_model": req.main_model,
    }

    try:
        _log_event(
            "info",
            "plan_generation_started",
            main_provider=req.main_provider,
            main_model=req.main_model,
            steps_count=len(req.steps),
            ce_score=round(req.ce_score, 4),
        )
        plan, prompt_tokens, completion_tokens = generate_learning_plan(
            query=req.prompt,
            steps=req.steps,
            ce_score=req.ce_score,
            diagnostics=req.diagnostics,
            api_config=api_config,
        )
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = _estimate_cost(prompt_tokens, completion_tokens, req.main_provider)

        _log_event(
            "info",
            "plan_generation_completed",
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            plan_chars=len(plan or ""),
        )

        return {
            "plan": plan,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }
    except Exception as e:
        logger.error(f"Plan generation error: {e}")
        _log_event("error", "plan_generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
