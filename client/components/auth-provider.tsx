"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { fetchMe, clearToken, getCachedUser, type User } from "@/lib/auth";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  logout: () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Read cache synchronously — if present, loading starts as false and the
  // page renders immediately without waiting for a network round-trip.
  const cached = getCachedUser();
  const [user, setUser] = useState<User | null>(cached);
  const [loading, setLoading] = useState(cached === null);

  useEffect(() => {
    // If we had a valid cache hit, still revalidate in the background so the
    // cache stays fresh — but the page doesn't block on it.
    fetchMe()
      .then((u) => { if (u) setUser(u); else if (!cached) setUser(null); })
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const logout = useCallback(() => {
    clearToken();
    document.cookie = "token=; path=/; max-age=0";
    setUser(null);
    window.location.href = "/";
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
