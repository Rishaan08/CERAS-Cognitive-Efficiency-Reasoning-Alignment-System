/**
 * saveSession.js — CERAS
 * Supabase removed entirely.
 * Session saving is now handled by server.py (db.py → Neon) inside /api/run-session.
 * This file is now a thin wrapper — it just calls /api/run-session and returns
 * the session_id + message_id that the backend already saved to Neon.
 */

import { authHeaders } from '../lib/auth';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

/**
 * Run session + save to Neon — all in one backend call.
 * Returns { session_id, message_id } for use in follow-up and plan saving.
 */
export async function saveSession({ userId, prompt, result, config, typingAnalytics }) {
  // session_id and message_id are returned directly by /api/run-session
  // after server.py saves everything to Neon via db.py.
  // Nothing to do here — just return what the backend already gave us.
  if (!result?.session_id || !result?.message_id) {
    console.warn('No session_id/message_id in result — DB save may have failed on backend.');
    return null;
  }

  return {
    session: { id: result.session_id },
    message: { id: result.message_id },
  };
}