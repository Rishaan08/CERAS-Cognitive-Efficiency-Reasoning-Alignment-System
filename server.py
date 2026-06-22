"""
CERAS FastAPI Backend Server
Wraps existing Python pipeline, ML models, and LLM utils as REST API endpoints.
Models are loaded lazily in a background thread so the frontend loads instantly.

Changes from original:
- Added JWT auth system (register, login, /auth/me) replacing Supabase auth
- Added get_current_user() dependency for protected endpoints
- Added DB saving to /api/run-session, /api/followup, /api/generate-plan via db.py
- db.py now uses asyncpg + Neon instead of Supabase client
"""

import os
from dotenv import load_dotenv

load_dotenv()

import sys
import time
import re
import json
import threading
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from uuid import uuid4

import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Request,
    Depends,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# JWT
from jose import JWTError, jwt
from passlib.context import CryptContext

# DB
from db import (
    save_session_to_db,
    save_followup_to_db,
    save_learning_plan_to_db,
    save_ml_training_row,
)

# NLP feature extraction (spaCy + textstat) — used for ml_training_data
# NLP feature extraction (spaCy + textstat) — used for ml_training_data
from nlp_features import extract_nlp_features

# Independent ground-truth CE score formula (refined Eq.1 from IEEE paper)
from ce_formula import compute_ce_score_label

# --------------- PATH SETUP ---------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src" / "ceras"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ASSET_DIR = BASE_DIR / "assets"

sys.path.insert(0, str(SRC_DIR))

# --------------- JWT CONFIG ---------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "10080"))  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

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
    logger.log(
        getattr(logging, level.upper(), logging.INFO), json.dumps(payload, default=str)
    )


# --------------- APP ---------------
app = FastAPI(title="CERAS API", version="2.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    import joblib
    import tensorflow as tf

    model_state["loading"] = True
    logger.info("⏳ Loading ML models in background...")
    try:
        model_state["cepm_model"] = joblib.load(str(ARTIFACT_DIR / "cepm_lightgbm.pkl"))
        model_state["cepm_scaler"] = joblib.load(str(ARTIFACT_DIR / "cepm_scaler.pkl"))
        model_state["cnn_model"] = tf.keras.models.load_model(
            str(ARTIFACT_DIR / "cnn_ce_model.keras")
        )
        model_state["cnn_scaler"] = joblib.load(str(ARTIFACT_DIR / "cnn_scaler.pkl"))
        model_state["cepm_features"] = np.load(
            str(ARTIFACT_DIR / "cepm_features.npy"), allow_pickle=True
        ).tolist()
        model_state["cnn_features"] = np.load(
            str(ARTIFACT_DIR / "cnn_features.npy"), allow_pickle=True
        ).tolist()
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


# --------------- JWT HELPERS ---------------
def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def _create_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# --------------- AUTH DEPENDENCY ---------------
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency — verifies JWT token on every protected endpoint.
    Use: def my_endpoint(current_user: dict = Depends(get_current_user))
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = _decode_token(credentials.credentials)
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"id": user_id, "email": email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")


# --------------- FEATURE EXTRACTION ---------------
def extract_ceras_features(prompt_text: str) -> dict:
    words = prompt_text.split()
    prompt_length = int(np.clip(len(words), 1, 400))
    character_count = len(prompt_text)
    sentence_count = max(len(re.findall(r"[.!?]", prompt_text)), 1)
    unique_word_ratio = float(np.clip(len(set(words)) / (prompt_length + 1e-6), 0, 1))
    concept_density = float(
        np.clip(sum(1 for w in words if len(w) > 6) / (prompt_length + 1e-6), 0, 1)
    )
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
    # Optional typing analytics from frontend
    typing_analytics: Optional[Dict[str, Any]] = None


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
    context: Dict[str, Any]
    history: List[Dict[str, str]] = []
    main_provider: str = "Groq"
    main_model: Optional[str] = None
    groq_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    # DB saving fields (optional — only saved if provided)
    message_id: Optional[str] = None


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
    # DB saving fields (optional)
    message_id: Optional[str] = None


# Auth request models
class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


# --------------- TOKEN COST HELPER ---------------
_COST_RATES = {
    "Groq": (0.59, 0.79),
    "Gemini": (0.075, 0.30),
    "OpenAI": (0.15, 0.60),
}


def _estimate_cost(prompt_tokens: int, completion_tokens: int, provider: str) -> float:
    inp_rate, out_rate = _COST_RATES.get(provider, (0.59, 0.79))
    return round(
        (prompt_tokens * inp_rate + completion_tokens * out_rate) / 1_000_000, 8
    )


# ================================================
# AUTH ENDPOINTS (new — replaces Supabase auth)
# ================================================


@app.post("/api/auth/register")
async def auth_register(req: RegisterRequest):
    """Register a new user. Returns JWT token + user object."""
    import asyncpg
    from db import get_connection

    conn = await get_connection()
    try:
        # Check if email already exists
        existing = await conn.fetchrow(
            "SELECT id FROM public.users WHERE email = $1", req.email
        )
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash password and insert user
        password_hash = _hash_password(req.password)
        user = await conn.fetchrow(
            """
            INSERT INTO public.users (email, password_hash, display_name)
            VALUES ($1, $2, $3)
            RETURNING id, email, display_name, created_at
            """,
            req.email,
            password_hash,
            req.display_name,
        )

        user_id = str(user["id"])
        token = _create_token(user_id, req.email)

        _log_event("info", "user_registered", user_id=user_id, email=req.email)

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": user["email"],
                "display_name": user["display_name"],
                "created_at": str(user["created_at"]),
            },
        }
    finally:
        await conn.close()


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    """Login with email + password. Returns JWT token + user object."""
    from db import get_connection

    conn = await get_connection()
    try:
        user = await conn.fetchrow(
            "SELECT id, email, password_hash, display_name, created_at FROM public.users WHERE email = $1",
            req.email,
        )
        if not user or not _verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user_id = str(user["id"])
        token = _create_token(user_id, req.email)

        _log_event("info", "user_logged_in", user_id=user_id, email=req.email)

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": user["email"],
                "display_name": user["display_name"],
                "created_at": str(user["created_at"]),
            },
        }
    finally:
        await conn.close()


@app.get("/api/auth/me")
async def auth_me(current_user: dict = Depends(get_current_user)):
    """Return current user info from JWT token."""
    from db import get_connection

    conn = await get_connection()
    try:
        user = await conn.fetchrow(
            "SELECT id, email, display_name, created_at FROM public.users WHERE id = $1",
            current_user["id"],
        )
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "id": str(user["id"]),
            "email": user["email"],
            "display_name": user["display_name"],
            "created_at": str(user["created_at"]),
        }
    finally:
        await conn.close()


# ================================================
# EXISTING ENDPOINTS (unchanged logic, DB saving added)
# ================================================


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
def get_logs(
    limit: int = 100, level: Optional[str] = None, contains: Optional[str] = None
):
    safe_limit = max(1, min(limit, 500))
    with log_buffer_lock:
        entries = list(log_buffer)
    if level:
        entries = [e for e in entries if e["level"] == level.upper()]
    if contains:
        needle = contains.lower()
        entries = [
            e
            for e in entries
            if needle in e["message"].lower() or needle in e["logger"].lower()
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
        _log_event(
            "info",
            "connection_check_completed",
            provider=req.provider,
            connected=bool(result),
        )
        return {"connected": result}
    except BaseException as e:
        logger.error(f"Connection check failed for {req.provider}: {e}")
        return {"connected": False, "error": str(e)}


# ------------------------------------------------
# Background task: save ml_training_data with full
# NLP feature extraction (spaCy + textstat).
# Runs AFTER the response has been sent to the user.
# ------------------------------------------------
async def _save_ml_training_background(
    message_id,
    session_id,
    user_id,
    prompt_text,
    features,
    typing_analytics,
    formulation_time,
    cepm_score,
    cnn_score,
    fused_score,
):
    try:
        ta = typing_analytics or {}

        avg_sentence_length = None
        if features.get("sentence_count") and features.get("prompt_length"):
            avg_sentence_length = features["prompt_length"] / max(
                features["sentence_count"], 1
            )

        # ---- NLP features (spaCy + textstat) ----
        nlp_feats = extract_nlp_features(prompt_text)

        # ---- Independent ground-truth CE score (refined Eq.1) ----
        # This is computed from raw features ONLY — never from
        # cepm_score/cnn_score/fused_score — so it stays a genuinely
        # independent label, not a copy of the model's own prediction.
        ce_score_label = compute_ce_score_label(
            prompt_length=features.get("prompt_length") or 0,
            unique_word_ratio=features.get("unique_word_ratio") or 0,
            keystrokes=ta.get("totalKeystrokes") or features.get("keystrokes") or 0,
            character_count=features.get("character_count") or 0,
            prompt_type=features.get("prompt_type") or 0,
            prompt_text=prompt_text,
        )

        ml_row = {
            "message_id": message_id,
            "session_id": session_id,
            "user_id": user_id,
            "prompt_text": prompt_text,
            # ---- Semantic Features (CNN) ----
            "prompt_length": features.get("prompt_length") or 0,
            "character_count": features.get("character_count") or 0,
            "sentence_count": features.get("sentence_count") or 0,
            "avg_sentence_length": avg_sentence_length or 0,
            "unique_word_ratio": features.get("unique_word_ratio") or 0,
            "multi_clause_count": nlp_feats.get("multi_clause_count") or 0,
            "cognitive_verb_count": nlp_feats.get("cognitive_verb_count") or 0,
            "lexical_diversity": nlp_feats.get("lexical_diversity")
            or features.get("unique_word_ratio")
            or 0,
            "readability_score": nlp_feats.get("readability_score") or 0,
            "stopword_ratio": nlp_feats.get("stopword_ratio") or 0,
            "punctuation_density": nlp_feats.get("punctuation_density") or 0,
            "named_entity_count": nlp_feats.get("named_entity_count") or 0,
            "keyword_density": nlp_feats.get("keyword_density") or 0,
            "topic_consistency_score": nlp_feats.get("topic_consistency_score") or 0,
            "coherence_score": nlp_feats.get("coherence_score") or 0,
            "prompt_type": features.get("prompt_type") or 0,
            "concept_density": features.get("concept_density") or 0,
            # ---- Behaviour Features (CEPM) — from useTypingAnalytics.js ----
            "keystrokes": ta.get("totalKeystrokes") or features.get("keystrokes") or 0,
            "typing_speed_wpm": ta.get("wpm") or 0,
            "typing_speed_cpm": ta.get("cpm") or 0,
            "avg_key_latency": ta.get("avgKeystrokeInterval") or 0,
            "latency_std": ta.get("interKeyDelayStd") or 0,
            "pause_count": ta.get("hesitations") or 0,
            "avg_pause_duration": ta.get("longestPause") or 0,
            "total_pauses_ms": (ta.get("hesitations") or 0)
            * (ta.get("longestPause") or 0),
            "typing_duration_ms": (ta.get("sessionDuration") or 0) * 1000,
            "idle_time": ta.get("currentPause") or 0,
            "burst_count": ta.get("burstCount") or 0,
            "burst_typing_ratio": ta.get("burstTypingRatio") or 0,
            "backspace_count": ta.get("deletions") or 0,
            "correction_rate": ta.get("deletionRatio") or 0,
            "rewrite_ratio": ta.get("deletionRatio") or 0,
            "delete_burst_count": 0,
            "error_rate": ta.get("deletionRatio") or 0,
            # first_input_delay is ALWAYS measurable (time from focus to
            # first keystroke/paste) — captured in useTypingAnalytics.js.
            # Never null/0 in real usage; default only covers the
            # edge case of a missing/old frontend payload.
            "first_input_delay": ta.get("firstInputDelay") or 0,
            "finalization_time": formulation_time or 0,
            "avg_inter_key_delay": ta.get("avgKeystrokeInterval") or 0,
            "inter_key_delay_std": ta.get("interKeyDelayStd") or 0,
            "hesitation_ratio": (
                ta.get("hesitations") / ta.get("totalKeystrokes")
                if ta.get("hesitations") and ta.get("totalKeystrokes")
                else 0
            ),
            # copy_paste_events: real count from useTypingAnalytics.js
            # registerPaste(), incremented on the textarea's native
            # paste event — 0 if the user typed everything themselves.
            "copy_paste_events": ta.get("copyPasteEvents") or 0,
            # cursor_movement_count and focus_loss_count were removed
            # from the schema entirely — weak/no signal for CE
            # prediction (wouldn't survive MI/RFE feature selection),
            # so computing them would just be storage cost with no
            # modeling benefit. See drop_unused_columns.sql.
            # ---- Targets ----
            # ce_score is the INDEPENDENT formula-based ground-truth label
            # (NOT a copy of fused_score) — see compute_ce_score_label()
            # above for why this distinction matters for model
            # comparison/retraining. Disagreement with fused_score is
            # EXPECTED right now: the deployed CEPM/CNN models were
            # trained on labels from the OLD formula, not this refined
            # one, so they're being compared against a target they've
            # never seen. This is not a sign the model is "better" —
            # it's a sign the model needs retraining on labels from
            # the new formula before the comparison is meaningful.
            "ce_score": ce_score_label,
            "prompt_quality": features.get("prompt_quality") or 0,
            # ---- Model Outputs ----
            "cepm_score": cepm_score,
            "cnn_score": cnn_score,
            "fused_score": fused_score,
            # input_mode: "paste" if the user pasted at least once
            # during this prompt, "typed" otherwise. Uses the real
            # copy_paste_events counter, not a fragile heuristic.
            "input_mode": "paste" if (ta.get("copyPasteEvents") or 0) > 0 else "typed",
        }

        await save_ml_training_row(ml_row)
        _log_event(
            "info",
            "ml_training_data_saved",
            message_id=message_id,
            ce_score_label=ce_score_label,
            fused_score=round(fused_score, 4),
            label_model_delta=round(ce_score_label - fused_score, 4),
        )
    except Exception as e:
        logger.error(f"ML training data background save failed (non-fatal): {e}")
        _log_event("warning", "ml_training_data_save_failed", error=str(e))


@app.post("/api/run-session")
async def run_session(
    req: RunSessionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Run CERAS ML pipeline on a prompt.
    Added: saves session + metrics + typing analytics to Neon after pipeline completes.
    """
    if not model_state["loaded"]:
        _log_event("warning", "run_session_blocked_models_loading")
        raise HTTPException(
            status_code=503, detail="Models are still loading. Please wait."
        )

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
        user_id=current_user["id"],
    )

    t0 = time.time()
    result = run_infer(req.prompt, api_config=api_config)
    runtime = time.time() - t0

    final_steps = result.get("final_answer", [])
    features = extract_ceras_features(req.prompt)

    # CEPM Inference
    cepm_input = np.array([features[f] for f in model_state["cepm_features"]]).reshape(
        1, -1
    )
    cepm_input_scaled = model_state["cepm_scaler"].transform(cepm_input)
    cepm_score = float(
        np.clip(model_state["cepm_model"].predict(cepm_input_scaled)[0], 0, 1)
    )

    # CNN Inference
    cnn_input = np.array([features[f] for f in model_state["cnn_features"]]).reshape(
        1, -1
    )
    cnn_input = model_state["cnn_scaler"].transform(cnn_input)
    if len(model_state["cnn_model"].input_shape) == 3:
        cnn_input = cnn_input.reshape(cnn_input.shape[0], cnn_input.shape[1], 1)
    cnn_score = float(
        np.clip(
            np.squeeze(model_state["cnn_model"].predict(cnn_input, verbose=0)), 0, 1
        )
    )

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
        suggestions.append(
            "Try adding more specific constraints or context to increase structural density."
        )
    if cnn_score > 0.75:
        strengths.append(
            "High semantic clarity; intent matches known high-performing patterns."
        )
    else:
        suggestions.append(
            "Clarify the core intent. Use precise domain terminology to improve semantic alignment."
        )
    if not strengths:
        strengths.append(
            "Prompt is functional but has room for optimization across all dimensions."
        )
    if not suggestions:
        suggestions.append("Excellent prompt! Maintains high cognitive efficiency.")

    # Save to Neon DB 
    db_ids = {"session_id": None, "message_id": None}
    try:
        db_result = await save_session_to_db(
            user_id=current_user["id"],
            prompt=req.prompt,
            result={
                "final_steps": final_steps
                if isinstance(final_steps, list)
                else [str(final_steps)],
                "strategy_used": result.get("strategy_used", ""),
                "llm_calls_used": result.get("llm_calls_used", 0),
                "cepm_score": cepm_score,
                "cnn_score": cnn_score,
                "fused_score": fused_score,
                "confidence": confidence,
                "readiness": readiness,
                "formulation_time": req.formulation_time,
                "runtime": runtime,
                "total_tokens": total_tokens,
                "features": features,
            },
            config=api_config,
            typing_analytics=req.typing_analytics,
        )
        db_ids = db_result
        _log_event("info", "run_session_saved_to_db", **db_ids)
    except Exception as e:
        logger.error(f"DB save failed (non-fatal): {e}")
        _log_event("warning", "run_session_db_save_failed", error=str(e))

    # ------------------------------------------------
    # Save to ml_training_data — scheduled as a BACKGROUND TASK.
    # NLP feature extraction (spaCy + textstat) takes 100-300ms;
    # we don't want the user waiting on that after their response
    # is already computed. This fires after the response is sent.
    # ------------------------------------------------
    background_tasks.add_task(
        _save_ml_training_background,
        message_id=db_ids.get("message_id"),
        session_id=db_ids.get("session_id"),
        user_id=current_user["id"],
        prompt_text=req.prompt,
        features=features,
        typing_analytics=req.typing_analytics or {},
        formulation_time=req.formulation_time,
        cepm_score=cepm_score,
        cnn_score=cnn_score,
        fused_score=fused_score,
    )

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
        "final_steps": final_steps
        if isinstance(final_steps, list)
        else [str(final_steps)],
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
        # Return DB IDs so frontend can use them for follow-up/plan saving
        "session_id": db_ids.get("session_id"),
        "message_id": db_ids.get("message_id"),
    }


@app.post("/api/adaptive-response")
def adaptive_response(
    req: AdaptiveResponseRequest,
    current_user: dict = Depends(get_current_user),
):
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
        _log_event(
            "info", "adaptive_response_completed", response_chars=len(response or "")
        )
        return {"response": response}
    except Exception as e:
        _log_event("error", "adaptive_response_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse-file")
async def parse_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    filename = (file.filename or "").lower()
    content = await file.read()
    _log_event(
        "info", "file_parse_started", filename=file.filename, size_bytes=len(content)
    )

    try:
        if filename.endswith(".pdf"):
            import pypdf, io

            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif filename.endswith(".docx"):
            import docx, io

            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)

        elif filename.endswith(".csv"):
            text = content.decode("utf-8", errors="replace")

        elif filename.endswith(".txt") or filename.endswith(".md"):
            text = content.decode("utf-8", errors="replace")

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {filename}"
            )

        if len(text) > 8000:
            text = text[:8000] + "\n... [truncated]"

        _log_event(
            "info",
            "file_parse_completed",
            filename=file.filename,
            chars=len(text.strip()),
        )
        return {
            "text": text.strip(),
            "filename": file.filename,
            "chars": len(text.strip()),
        }

    except HTTPException:
        _log_event("warning", "file_parse_rejected", filename=file.filename)
        raise
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        _log_event("error", "file_parse_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")


@app.post("/api/followup")
async def followup_chat(
    req: FollowUpRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Socratic follow-up chat.
    Added: saves user message + assistant response to followup_messages in Neon.
    """
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

        # Save to Neon if message_id provided
        if req.message_id:
            try:
                # Save user message
                await save_followup_to_db(
                    message_id=req.message_id,
                    user_id=current_user["id"],
                    role="user",
                    content=req.message,
                )
                # Save assistant response
                await save_followup_to_db(
                    message_id=req.message_id,
                    user_id=current_user["id"],
                    role="assistant",
                    content=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                )
                _log_event("info", "followup_saved_to_db", message_id=req.message_id)
            except Exception as e:
                logger.error(f"Followup DB save failed (non-fatal): {e}")
                _log_event("warning", "followup_db_save_failed", error=str(e))

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


@app.post("/api/generate-plan")
async def generate_plan(
    req: GeneratePlanRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Learning plan generator.
    Added: saves plan to learning_plans in Neon.
    """
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

        # Save to Neon if message_id provided
        if req.message_id:
            try:
                await save_learning_plan_to_db(
                    message_id=req.message_id,
                    user_id=current_user["id"],
                    plan_text=plan,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                )
                _log_event("info", "plan_saved_to_db", message_id=req.message_id)
            except Exception as e:
                logger.error(f"Plan DB save failed (non-fatal): {e}")
                _log_event("warning", "plan_db_save_failed", error=str(e))

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


# ================================================
# VAULT ENDPOINTS (replaces useVault.js Supabase calls)
# ================================================


class VaultSaveRequest(BaseModel):
    provider: str
    api_key: str
    key_label: str = "default"


class VaultVerifyRequest(BaseModel):
    is_valid: bool


@app.get("/api/vault/keys")
async def vault_get_keys(current_user: dict = Depends(get_current_user)):
    """Get all active API keys for the current user."""
    from db import get_connection

    conn = await get_connection()
    try:
        rows = await conn.fetch(
            """
            SELECT id, provider, key_label, is_active, is_valid, last_verified_at, created_at
            FROM public.api_keys
            WHERE user_id = $1 AND is_active = true
            ORDER BY created_at DESC
            """,
            current_user["id"],
        )
        return {"keys": [dict(r) for r in rows]}
    finally:
        await conn.close()


@app.post("/api/vault/save")
async def vault_save_key(
    req: VaultSaveRequest, current_user: dict = Depends(get_current_user)
):
    """Save or update an API key (upsert by user_id + provider + key_label)."""
    from db import get_connection

    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO public.api_keys (user_id, provider, api_key, key_label, is_active)
            VALUES ($1, $2, $3, $4, true)
            ON CONFLICT (user_id, provider, key_label)
            DO UPDATE SET api_key = EXCLUDED.api_key, is_active = true, updated_at = NOW()
            RETURNING id, provider, key_label, is_active, created_at
            """,
            current_user["id"],
            req.provider,
            req.api_key,
            req.key_label,
        )
        return {"key": dict(row)}
    finally:
        await conn.close()


@app.delete("/api/vault/delete/{key_id}")
async def vault_delete_key(key_id: str, current_user: dict = Depends(get_current_user)):
    """Delete an API key (only if it belongs to the current user)."""
    from db import get_connection

    conn = await get_connection()
    try:
        await conn.execute(
            "DELETE FROM public.api_keys WHERE id = $1 AND user_id = $2",
            key_id,
            current_user["id"],
        )
        return {"deleted": True}
    finally:
        await conn.close()


@app.patch("/api/vault/verify/{key_id}")
async def vault_verify_key(
    key_id: str, req: VaultVerifyRequest, current_user: dict = Depends(get_current_user)
):
    """Update verification status of an API key."""
    from db import get_connection

    conn = await get_connection()
    try:
        await conn.execute(
            """
            UPDATE public.api_keys
            SET is_valid = $1, last_verified_at = NOW()
            WHERE id = $2 AND user_id = $3
            """,
            req.is_valid,
            key_id,
            current_user["id"],
        )
        return {"updated": True}
    finally:
        await conn.close()


# ================================================
# HISTORY ENDPOINTS (replaces useHistory.js Supabase calls)
# ================================================


@app.get("/api/history")
async def get_history(
    limit: int = 50,
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Get chat session history for the current user, with optional search."""
    from db import get_connection

    conn = await get_connection()
    try:
        if search:
            rows = await conn.fetch(
                """
                SELECT
                    cs.id, cs.session_title, cs.main_provider, cs.verifier_provider,
                    cs.main_model, cs.verifier_model, cs.created_at,
                    json_agg(json_build_object(
                        'id', cm.id,
                        'prompt', cm.prompt,
                        'final_steps', cm.final_steps,
                        'strategy_used', cm.strategy_used,
                        'llm_calls_used', cm.llm_calls_used,
                        'created_at', cm.created_at
                    )) AS chat_messages
                FROM public.chat_sessions cs
                LEFT JOIN public.chat_messages cm ON cm.session_id = cs.id
                WHERE cs.user_id = $1 AND cm.prompt ILIKE $2
                GROUP BY cs.id
                ORDER BY cs.created_at DESC
                LIMIT $3
                """,
                current_user["id"],
                f"%{search}%",
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT
                    cs.id, cs.session_title, cs.main_provider, cs.verifier_provider,
                    cs.main_model, cs.verifier_model, cs.created_at,
                    json_agg(json_build_object(
                        'id', cm.id,
                        'prompt', cm.prompt,
                        'final_steps', cm.final_steps,
                        'strategy_used', cm.strategy_used,
                        'llm_calls_used', cm.llm_calls_used,
                        'created_at', cm.created_at
                    )) AS chat_messages
                FROM public.chat_sessions cs
                LEFT JOIN public.chat_messages cm ON cm.session_id = cs.id
                WHERE cs.user_id = $1
                GROUP BY cs.id
                ORDER BY cs.created_at DESC
                LIMIT $2
                """,
                current_user["id"],
                limit,
            )
        return {"sessions": [dict(r) for r in rows]}
    finally:
        await conn.close()


@app.delete("/api/history/delete/{session_id}")
async def delete_session(
    session_id: str, current_user: dict = Depends(get_current_user)
):
    """Delete a session (only if it belongs to the current user)."""
    from db import get_connection

    conn = await get_connection()
    try:
        await conn.execute(
            "DELETE FROM public.chat_sessions WHERE id = $1 AND user_id = $2",
            session_id,
            current_user["id"],
        )
        return {"deleted": True}
    finally:
        await conn.close()


# ================================================
# SAVE REPORT ENDPOINT (replaces Dashboard.jsx Supabase call)
# ================================================


class SaveReportRequest(BaseModel):
    session_id: str
    message_id: str
    report_content: str


@app.post("/api/save-report")
async def save_report(
    req: SaveReportRequest, current_user: dict = Depends(get_current_user)
):
    """Save a session report to Neon."""
    from db import get_connection

    conn = await get_connection()
    try:
        await conn.execute(
            """
            INSERT INTO public.session_reports (session_id, message_id, user_id, report_content)
            VALUES ($1, $2, $3, $4)
            """,
            req.session_id,
            req.message_id,
            current_user["id"],
            req.report_content,
        )
        return {"saved": True}
    finally:
        await conn.close()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
