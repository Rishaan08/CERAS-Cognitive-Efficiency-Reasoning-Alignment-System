/**
 * useHistory.js — CERAS
 * Supabase removed. All operations now go through FastAPI endpoints.
 * Endpoints added to server.py: /api/history, /api/history/delete/:id
 */

import { useCallback, useEffect, useState } from 'react';
import { authHeaders } from '../lib/auth';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

export default function useHistory(userId) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchSessions = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: 50 });
      if (searchQuery.trim()) params.append('search', searchQuery.trim());

      const res = await fetch(`${API_BASE}/history?${params}`, {
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error('Failed to fetch history');
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (err) {
      console.error('Error fetching history:', err);
    } finally {
      setLoading(false);
    }
  }, [userId, searchQuery]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const deleteSession = async (sessionId) => {
    try {
      const res = await fetch(`${API_BASE}/history/delete/${sessionId}`, {
        method: 'DELETE',
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error('Failed to delete session');
      setSessions(prev => prev.filter(s => s.id !== sessionId));
    } catch (err) {
      console.error('Error deleting session:', err);
    }
  };

  return {
    sessions,
    loading,
    searchQuery,
    setSearchQuery,
    refresh: fetchSessions,
    deleteSession,
  };
}