/**
 * auth.js — replaces supabase.js
 * JWT-based auth client for CERAS.
 * Drop-in replacement: delete supabase.js, use this instead.
 */

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

// Token helpers (localStorage)
export const getToken = () => localStorage.getItem('ceras_token');
export const setToken = (token) => localStorage.setItem('ceras_token', token);
export const removeToken = () => localStorage.removeItem('ceras_token');

export const getUser = () => {
  const raw = localStorage.getItem('ceras_user');
  try { return raw ? JSON.parse(raw) : null; } catch { return null; }
};
export const setUser = (user) => localStorage.setItem('ceras_user', JSON.stringify(user));
export const removeUser = () => localStorage.removeItem('ceras_user');

// Auth header helper — use this in every API call
export const authHeaders = () => {
  const token = getToken();
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

// Register
export async function register({ email, password, display_name }) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, display_name }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Registration failed');
  setToken(data.access_token);
  setUser(data.user);
  return data;
}

// Login
export async function login({ email, password }) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Login failed');
  setToken(data.access_token);
  setUser(data.user);
  return data;
}

// Logout
export function logout() {
  removeToken();
  removeUser();
}

// Get current user from token (verify with backend)
export async function getCurrentUser() {
  const token = getToken();
  if (!token) return null;
  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: authHeaders(),
    });
    if (!res.ok) { logout(); return null; }
    return await res.json();
  } catch {
    return null;
  }
}

// Check if user is logged in
export const isAuthenticated = () => !!getToken();