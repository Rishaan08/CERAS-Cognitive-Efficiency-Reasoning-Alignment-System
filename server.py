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
from pathlib import Path
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ceras-server")

# --------------- APP ---------------
app = FastAPI(title="CERAS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# --------------- ENDPOINTS ---------------

@app.get("/health")
@app.get("/api/health")
def health():
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
        return FileResponse(str(logo_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


@app.post("/api/check-connection")
def check_connection_endpoint(req: CheckConnectionRequest):
    try:
        from llm_utils import check_connection
        result = check_connection(req.provider, req.api_key)
        return {"connected": result}
    except BaseException as e:
        logger.error(f"Connection check failed for {req.provider}: {e}")
        return {"connected": False, "error": str(e)}


@app.post("/api/run-session")
def run_session(req: RunSessionRequest):
    if not model_state["loaded"]:
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

    t0 = time.time()
    result = run_infer(req.prompt, api_config=api_config)
    runtime = time.time() - t0

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

    return {
        "final_steps": final_steps if isinstance(final_steps, list) else [str(final_steps)],
        "strategy_used": result.get("strategy_used", ""),
        "llm_calls_used": result.get("llm_calls_used", 0),
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
        response = generate_adaptive_response(
            req.prompt,
            req.steps,
            req.ce_score,
            req.diagnostics,
            api_config=api_config,
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
