/**
 * useVault.js — CERAS
 * Supabase removed. All operations now go through FastAPI endpoints.
 * Endpoints added to server.py: /api/vault/keys, /api/vault/save,
 * /api/vault/delete/:id, /api/vault/verify/:id
 */

import { useCallback, useEffect, useState } from 'react';
import { authHeaders } from '../lib/auth';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

export default function useVault(userId) {
  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchKeys = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/vault/keys`, {
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error('Failed to fetch keys');
      const data = await res.json();
      setKeys(data.keys || []);
    } catch (err) {
      console.error('Error fetching keys:', err);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const saveKey = async (provider, apiKey, label = 'default') => {
    if (!userId) return;
    try {
      const res = await fetch(`${API_BASE}/vault/save`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ provider, api_key: apiKey, key_label: label }),
      });
      if (!res.ok) throw new Error('Failed to save key');
      const data = await res.json();
      await fetchKeys();
      return data;
    } catch (err) {
      console.error('Error saving key:', err);
      throw err;
    }
  };

  const deleteKey = async (keyId) => {
    try {
      const res = await fetch(`${API_BASE}/vault/delete/${keyId}`, {
        method: 'DELETE',
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error('Failed to delete key');
      setKeys(prev => prev.filter(k => k.id !== keyId));
    } catch (err) {
      console.error('Error deleting key:', err);
    }
  };

  const updateVerification = async (keyId, isValid) => {
    try {
      const res = await fetch(`${API_BASE}/vault/verify/${keyId}`, {
        method: 'PATCH',
        headers: authHeaders(),
        body: JSON.stringify({ is_valid: isValid }),
      });
      if (!res.ok) throw new Error('Failed to update verification');
      await fetchKeys();
    } catch (err) {
      console.error('Error updating verification:', err);
    }
  };

  const getKeyForProvider = (provider) => {
    const found = keys.find(k => k.provider === provider && k.is_active);
    return found?.api_key || '';
  };

  return {
    keys,
    loading,
    saveKey,
    deleteKey,
    updateVerification,
    getKeyForProvider,
    refresh: fetchKeys,
  };
}