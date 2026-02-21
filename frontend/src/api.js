console.log("API URL:", import.meta.env.VITE_API_URL);
const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api'; // fallback for local dev

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function checkConnection(provider, apiKey) {
  const res = await fetch(`${API_BASE}/check-connection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
  return res.json();
}

export async function runSession(payload) {
  const res = await fetch(`${API_BASE}/run-session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || 'Adaptive response failed');
  }

  return res.json();
}
