"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { fetchMe, clearToken, getCachedUser, getToken, type User } from "@/lib/auth";

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
  const attemptsLeft = useRef(2); // allow 1 retry on transient failures

  useEffect(() => {
    const attempt = () => {
      fetchMe()
        .then((u) => {
          // null means 401/403 — token definitively rejected
          setUser(u);
          setLoading(false);
        })
        .catch(() => {
          // Transient error (5xx, network hiccup, timeout)
          attemptsLeft.current -= 1;
          if (attemptsLeft.current > 0) {
            // Retry once after 3 s
            setTimeout(attempt, 3_000);
          } else {
            // Exhausted retries — honour cache, or stay loading if we have
            // a token so we don't falsely redirect to /login on a transient outage.
            const stillHasToken = !!getToken();
            if (cached) {
              setUser(cached);
              setLoading(false);
            } else if (!stillHasToken) {
              setUser(null);
              setLoading(false);
            }
            // If token exists but server unreachable: keep loading=true
            // (spinner) rather than bouncing the user back to /login.
          }
        });
    };
    attempt();
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
