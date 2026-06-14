/**
 * api.js — CERAS frontend API client
 * Updated: replaced Supabase auth with JWT token headers on every request.
 * Import authHeaders from auth.js instead of supabase.js.
 */

import { authHeaders } from './auth';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function checkConnection(provider, apiKey) {
  const res = await fetch(`${API_BASE}/check-connection`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
  return res.json();
}

export async function runSession(payload) {
  const res = await fetch(`${API_BASE}/run-session`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || 'Session failed');
  }
  return res.json();
}

export async function getAdaptiveResponse(payload) {
  const res = await fetch(`${API_BASE}/adaptive-response`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || 'Adaptive response failed');
  }
  return res.json();
}

export async function parseFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const token = localStorage.getItem('ceras_token');
  const res = await fetch(`${API_BASE}/parse-file`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'File parse failed' }));
    throw new Error(err.detail || 'File parse failed');
  }
  return res.json();
}

export async function sendFollowUp(payload) {
  const res = await fetch(`${API_BASE}/followup`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Follow-up failed' }));
    throw new Error(err.detail || 'Follow-up failed');
  }
  return res.json();
}

export async function generatePlan(payload) {
  const res = await fetch(`${API_BASE}/generate-plan`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Plan generation failed' }));
    throw new Error(err.detail || 'Plan generation failed');
  }
  return res.json();
}