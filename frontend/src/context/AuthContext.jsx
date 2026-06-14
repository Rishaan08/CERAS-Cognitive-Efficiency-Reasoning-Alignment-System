/**
 * AuthContext.jsx — CERAS
 * Replaces Supabase auth with custom JWT auth from auth.js
 * API shape is IDENTICAL: { user, session, loading, signUp, signIn, signOut }
 * No other component needs to change.
 */

import { createContext, useContext, useEffect, useState } from 'react';
import {
  login,
  register,
  logout,
  getCurrentUser,
  getToken,
} from '../lib/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser]       = useState(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  // On mount: restore session from localStorage token
  useEffect(() => {
    const restoreSession = async () => {
      const token = getToken();
      if (!token) {
        setLoading(false);
        return;
      }
      // Verify token is still valid with backend
      const currentUser = await getCurrentUser();
      if (currentUser) {
        setUser(currentUser);
        setSession({ access_token: token, user: currentUser });
      } else {
        // Token expired or invalid — clear everything
        logout();
        setUser(null);
        setSession(null);
      }
      setLoading(false);
    };

    restoreSession();
  }, []);

  // signUp — replaces supabase.auth.signUp()
  const signUp = async (email, password, displayName) => {
    const data = await register({ email, password, display_name: displayName });
    setUser(data.user);
    setSession({ access_token: data.access_token, user: data.user });
    return data;
  };

  // signIn — replaces supabase.auth.signInWithPassword()
  const signIn = async (email, password) => {
    const data = await login({ email, password });
    setUser(data.user);
    setSession({ access_token: data.access_token, user: data.user });
    return data;
  };

  // signOut — replaces supabase.auth.signOut()
  const signOut = async () => {
    logout();
    setUser(null);
    setSession(null);
  };

  const value = {
    user,
    session,
    loading,
    signUp,
    signIn,
    signOut,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;
