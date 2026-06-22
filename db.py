"""
Neon PostgreSQL async client for CERAS server-side database operations.
Replaces Supabase client — uses asyncpg directly with Neon connection pool.
"""

import os
import asyncpg
from typing import Optional

DATABASE_URL = os.getenv("ASYNC_DATABASE_URL", "")


async def get_connection() -> asyncpg.Connection:
    """Get a single async connection to Neon."""
    if not DATABASE_URL:
        raise RuntimeError("ASYNC_DATABASE_URL must be set in .env")
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    return await asyncpg.connect(url)

async def save_session_to_db(
    user_id: str,
    prompt: str,
    result: dict,
    config: dict,
    typing_analytics: Optional[dict] = None,
) -> dict:
    """
    Save a complete session (chat + metrics + typing) to Neon.
    Called from the /api/save-session endpoint.
    Replaces the old Supabase version — same logic, async asyncpg queries.
    """
    conn = await get_connection()

    try:
        # ------------------------------------------------
        # 1. Create chat session
        # ------------------------------------------------
        session_id = await conn.fetchval(
            """
            INSERT INTO public.chat_sessions
              (user_id, session_title, main_provider, verifier_provider, main_model, verifier_model)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            user_id,
            prompt[:80],
            config.get("main_provider"),
            config.get("verifier_provider"),
            config.get("main_model"),
            config.get("verifier_model"),
        )

        # ------------------------------------------------
        # 2. Create chat message
        # ------------------------------------------------
        import json
        message_id = await conn.fetchval(
            """
            INSERT INTO public.chat_messages
              (session_id, user_id, prompt, final_steps, strategy_used, llm_calls_used)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            session_id,
            user_id,
            prompt,
            json.dumps(result.get("final_steps", [])),
            result.get("strategy_used", ""),
            result.get("llm_calls_used", 0),
        )

        # ------------------------------------------------
        # 3. Save session metrics
        # ------------------------------------------------
        features = result.get("features", {})
        ta = typing_analytics or {}

        await conn.execute(
            """
            INSERT INTO public.session_metrics (
              message_id, user_id,
              cepm_score, cnn_score, fused_score, confidence, readiness,
              formulation_time, runtime, total_tokens,
              prompt_length, character_count, sentence_count,
              unique_word_ratio, concept_density, prompt_quality,
              keystrokes, prompt_type,
              typing_speed_wpm, typing_speed_cpm,
              backspace_count, pause_count, avg_pause_duration,
              total_pauses_ms, typing_duration_ms, burst_count,
              api_provider_main, api_provider_verifier,
              model_main, model_verifier
            ) VALUES (
              $1,  $2,  $3,  $4,  $5,  $6,  $7,
              $8,  $9,  $10, $11, $12, $13,
              $14, $15, $16,
              $17, $18,
              $19, $20,
              $21, $22, $23,
              $24, $25, $26,
              $27, $28, $29, $30
            )
            """,
            message_id, user_id,
            result.get("cepm_score"),
            result.get("cnn_score"),
            result.get("fused_score"),
            result.get("confidence"),
            result.get("readiness"),
            result.get("formulation_time"),
            result.get("runtime"),
            result.get("total_tokens"),
            features.get("prompt_length") or 0,
            features.get("character_count") or 0,
            features.get("sentence_count") or 0,
            features.get("unique_word_ratio") or 0,
            features.get("concept_density") or 0,
            features.get("prompt_quality") or 0,
            features.get("keystrokes") or 0,
            features.get("prompt_type") or 0,
            ta.get("wpm") or 0,
            ta.get("cpm") or 0,
            ta.get("deletions") or 0,
            ta.get("hesitations") or 0,
            ta.get("longestPause") or 0,
            (ta.get("hesitations") or 0) * (ta.get("longestPause") or 0),
            (ta.get("sessionDuration") or 0) * 1000,
            ta.get("burstCount") or 0,
            config.get("main_provider"),
            config.get("verifier_provider"),
            config.get("main_model"),
            config.get("verifier_model"),
        )

        # ------------------------------------------------
        # 4. Save typing analytics (if provided)
        if typing_analytics:
            await conn.execute(
                """
                INSERT INTO public.typing_analytics (
                  message_id, user_id,
                  wpm, cpm, backspace_count, pause_count,
                  avg_pause_ms, total_pauses_ms, duration_ms, burst_count
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                message_id, user_id,
                ta.get("wpm") or 0,
                ta.get("cpm") or 0,
                ta.get("deletions") or 0,
                ta.get("hesitations") or 0,
                ta.get("longestPause") or 0,
                (ta.get("hesitations") or 0) * (ta.get("longestPause") or 0),
                (ta.get("sessionDuration") or 0) * 1000,
                ta.get("burstCount") or 0,
            )

        return {"session_id": str(session_id), "message_id": str(message_id)}

    finally:
        await conn.close()


async def save_followup_to_db(
    message_id: str,
    user_id: str,
    role: str,
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: Optional[float] = None,
) -> str:
    """Save a follow-up message to Neon. Returns the new followup id."""
    conn = await get_connection()
    try:
        followup_id = await conn.fetchval(
            """
            INSERT INTO public.followup_messages
              (message_id, user_id, role, content, prompt_tokens, completion_tokens, cost_usd)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            message_id, user_id, role, content,
            prompt_tokens, completion_tokens, cost_usd,
        )
        return str(followup_id)
    finally:
        await conn.close()


async def save_learning_plan_to_db(
    message_id: str,
    user_id: str,
    plan_text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: Optional[float] = None,
) -> str:
    """Save a learning plan to Neon. Returns the new plan id."""
    conn = await get_connection()
    try:
        plan_id = await conn.fetchval(
            """
            INSERT INTO public.learning_plans
              (message_id, user_id, plan_text, prompt_tokens, completion_tokens, cost_usd)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            message_id, user_id, plan_text,
            prompt_tokens, completion_tokens, cost_usd,
        )
        return str(plan_id)
    finally:
        await conn.close()


async def save_ml_training_row(row: dict) -> str:
    """Save a real user prompt's extracted features to ml_training_data.
    cursor_movement_count and focus_loss_count were removed from the
    schema — weak/no signal for CE prediction, wouldn't survive
    MI/RFE feature selection, so not worth the storage cost."""
    conn = await get_connection()
    try:
        record_id = await conn.fetchval(
            """
            INSERT INTO public.ml_training_data (
              message_id, session_id, user_id, prompt_text,
              prompt_length, character_count, sentence_count, avg_sentence_length,
              unique_word_ratio, multi_clause_count, cognitive_verb_count,
              lexical_diversity, readability_score, stopword_ratio,
              punctuation_density, named_entity_count, keyword_density,
              topic_consistency_score, coherence_score, prompt_type, concept_density,
              keystrokes, typing_speed_wpm, typing_speed_cpm,
              avg_key_latency, latency_std, pause_count, avg_pause_duration,
              total_pauses_ms, typing_duration_ms, idle_time, burst_count,
              burst_typing_ratio, backspace_count, correction_rate, rewrite_ratio,
              delete_burst_count, error_rate, first_input_delay, finalization_time,
              avg_inter_key_delay, inter_key_delay_std, hesitation_ratio,
              copy_paste_events,
              ce_score, prompt_quality, cepm_score, cnn_score, fused_score, input_mode
            ) VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
              $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
              $21,$22,$23,$24,$25,$26,$27,$28,$29,$30,
              $31,$32,$33,$34,$35,$36,$37,$38,$39,$40,
              $41,$42,$43,$44,$45,$46,$47,$48,$49,$50
            )
            RETURNING id
            """,
            row.get("message_id"), row.get("session_id"), row.get("user_id"),
            row.get("prompt_text"),
            row.get("prompt_length"), row.get("character_count"),
            row.get("sentence_count"), row.get("avg_sentence_length"),
            row.get("unique_word_ratio"), row.get("multi_clause_count"),
            row.get("cognitive_verb_count"), row.get("lexical_diversity"),
            row.get("readability_score"), row.get("stopword_ratio"),
            row.get("punctuation_density"), row.get("named_entity_count"),
            row.get("keyword_density"), row.get("topic_consistency_score"),
            row.get("coherence_score"), row.get("prompt_type"),
            row.get("concept_density"), row.get("keystrokes"),
            row.get("typing_speed_wpm"), row.get("typing_speed_cpm"),
            row.get("avg_key_latency"), row.get("latency_std"),
            row.get("pause_count"), row.get("avg_pause_duration"),
            row.get("total_pauses_ms"), row.get("typing_duration_ms"),
            row.get("idle_time"), row.get("burst_count"),
            row.get("burst_typing_ratio"), row.get("backspace_count"),
            row.get("correction_rate"), row.get("rewrite_ratio"),
            row.get("delete_burst_count"), row.get("error_rate"),
            row.get("first_input_delay"), row.get("finalization_time"),
            row.get("avg_inter_key_delay"), row.get("inter_key_delay_std"),
            row.get("hesitation_ratio"), row.get("copy_paste_events"),
            row.get("ce_score"), row.get("prompt_quality"),
            row.get("cepm_score"), row.get("cnn_score"),
            row.get("fused_score"), row.get("input_mode"),
        )
        return str(record_id)
    finally:
        await conn.close()