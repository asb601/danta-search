const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const USER_CACHE_KEY = "gchat_user";
const USER_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

export interface User {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
  is_admin: boolean;
  role: string;
  allowed_domains: string[] | null;
  organization_id: string | null;
}

/** Retrieve stored JWT token */
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("token");
}

/** Store JWT token */
export function setToken(token: string): void {
  localStorage.setItem("token", token);
}

/** Clear stored token */
export function clearToken(): void {
  localStorage.removeItem("token");
  localStorage.removeItem(USER_CACHE_KEY);
}

/** Read user synchronously from localStorage cache (zero network). */
export function getCachedUser(): User | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(USER_CACHE_KEY);
    if (!raw) return null;
    const { user, ts } = JSON.parse(raw) as { user: User; ts: number };
    if (Date.now() - ts > USER_CACHE_TTL_MS) return null; // stale
    return user;
  } catch {
    return null;
  }
}

/** Write user to localStorage cache. */
export function setCachedUser(user: User): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(USER_CACHE_KEY, JSON.stringify({ user, ts: Date.now() }));
}

/** Authenticated fetch wrapper */
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const token = getToken();
  const headers = new Headers(init?.headers);
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(`${API_URL}${path}`, { ...init, headers });
}

/** Fetch current user from the backend — waits up to 20 s.
 *  Returns null only when there is definitively no token or the server
 *  rejects it with 401/403 (token is cleared in that case).
 *  Throws for transient failures (5xx, network error, timeout) so callers
 *  can retry rather than treating the session as ended. */
export async function fetchMe(): Promise<User | null> {
  const token = getToken();
  if (!token) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20_000);
  try {
    const res = await apiFetch("/api/auth/me", { signal: controller.signal });
    if (res.status === 401 || res.status === 403) {
      clearToken();
      return null; // Definitive: token is invalid
    }
    if (!res.ok) throw new Error("transient");
    const user: User = await res.json();
    setCachedUser(user); // keep cache fresh
    return user;
  } catch (err) {
    if (err instanceof Error && err.message === "transient") throw err;
    throw new Error("transient"); // network error or abort (timeout)
  } finally {
    clearTimeout(timer);
  }
}

/** Get the Google OAuth login URL (points at FastAPI) */
export function getGoogleLoginUrl(): string {
  return `${API_URL}/api/auth/google/login`;
}
