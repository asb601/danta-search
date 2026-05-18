"""
Basic functional smoke test — single user, no load.

Checks that the server is up, auth works, folders work, and chat responds.
Run this against the main VM to confirm basic functionality before any load testing.

Usage
-----
    python testing/smoke_test.py --base-url https://your-vm --token YOUR_JWT_TOKEN

    # Without a token: only tests unauthenticated endpoints
    python testing/smoke_test.py --base-url https://your-vm

    # Also test chat (any question)
    python testing/smoke_test.py --base-url https://your-vm --token TOKEN --chat "what files do I have?"
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

PASS_COUNT = 0
FAIL_COUNT = 0


def _ok(label: str, detail: str = "") -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [PASS] {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    suffix = f"  ← {detail}" if detail else ""
    print(f"  [FAIL] {label}{suffix}")


def _check(label: str, cond: bool, detail: str = "") -> bool:
    if cond:
        _ok(label, detail)
    else:
        _fail(label, detail)
    return cond


# ── Individual checks ─────────────────────────────────────────────────────────

def check_health(client: httpx.Client, base: str) -> bool:
    print("\n── Health ──────────────────────────────────────────────────────────")
    try:
        t = time.perf_counter()
        r = client.get(f"{base}/api/health", timeout=10)
        ms = round((time.perf_counter() - t) * 1000)
        ok = _check("GET /api/health → 200", r.status_code == 200, f"{ms} ms")
        if ok:
            body = r.json()
            _check("health body has status=ok", body.get("status") == "ok", str(body))
        return ok
    except Exception as e:
        _fail("GET /api/health", str(e))
        return False


def check_metrics(client: httpx.Client, base: str) -> None:
    print("\n── Metrics ─────────────────────────────────────────────────────────")
    try:
        r = client.get(f"{base}/api/metrics", timeout=10)
        _check("GET /api/metrics → 200", r.status_code == 200, str(r.status_code))
    except Exception as e:
        _fail("GET /api/metrics", str(e))


def check_unauth_returns_401(client: httpx.Client, base: str) -> None:
    """Protected endpoints must return 401, not 500."""
    print("\n── Auth guard (no token) ───────────────────────────────────────────")
    for path in ["/api/auth/me", "/api/folders", "/api/users"]:
        try:
            r = httpx.get(f"{base}{path}", timeout=10)
            _check(
                f"GET {path} without token → 401",
                r.status_code == 401,
                f"got {r.status_code}",
            )
        except Exception as e:
            _fail(f"GET {path}", str(e))


def check_auth_me(client: httpx.Client, base: str) -> dict | None:
    print("\n── Auth ────────────────────────────────────────────────────────────")
    try:
        r = client.get(f"{base}/api/auth/me", timeout=10)
        ok = _check("GET /api/auth/me → 200", r.status_code == 200, f"got {r.status_code}")
        if not ok:
            _fail("cannot continue auth checks — invalid token?", r.text[:120])
            return None
        user = r.json()
        _check("response has email", "email" in user, str(list(user.keys())))
        _check("response has id", "id" in user, "")
        print(f"         Logged in as: {user.get('email')}  admin={user.get('is_admin')}")
        return user
    except Exception as e:
        _fail("GET /api/auth/me", str(e))
        return None


def check_folders(client: httpx.Client, base: str) -> None:
    print("\n── Folders ─────────────────────────────────────────────────────────")
    try:
        r = client.get(f"{base}/api/folders", timeout=10)
        ok = _check("GET /api/folders → 200", r.status_code == 200, f"got {r.status_code}")
        if ok:
            data = r.json()
            _check("response is a list", isinstance(data, list), type(data).__name__)
            print(f"         Folder count: {len(data)}")
    except Exception as e:
        _fail("GET /api/folders", str(e))


def check_chat(client: httpx.Client, base: str, question: str) -> None:
    print("\n── Chat stream ─────────────────────────────────────────────────────")
    payload = {"query": question, "conversation_id": None, "container_id": None}
    ttft: float | None = None
    full_text = ""
    t_start = time.perf_counter()
    try:
        with client.stream(
            "POST", f"{base}/api/chat/message/stream", json=payload, timeout=60
        ) as resp:
            status_ok = _check(
                f"POST /api/chat/message/stream → 200", resp.status_code == 200,
                f"got {resp.status_code}"
            )
            if not status_ok:
                print(f"         Response: {resp.read()[:200]}")
                return

            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                chunk = evt.get("content") or evt.get("delta") or evt.get("text") or ""
                full_text += chunk

        total_s = round(time.perf_counter() - t_start, 2)
        ttft_s = round(ttft, 2) if ttft else total_s
        _check("received non-empty answer", len(full_text) > 0, f"{len(full_text)} chars")
        _ok(f"TTFT {ttft_s}s  |  total {total_s}s")
        print(f"         Preview: {full_text[:120].replace(chr(10), ' ')}")

    except httpx.TimeoutException:
        _fail("chat stream timed out after 60s")
    except Exception as e:
        _fail("chat stream error", str(e)[:200])


def check_logs_api(client: httpx.Client, base: str) -> None:
    """Check the logs API is reachable (admin only)."""
    print("\n── Logs API ────────────────────────────────────────────────────────")
    try:
        r = client.get(f"{base}/api/logs/files", timeout=10)
        _check(
            "GET /api/logs/files → 200 or 404",
            r.status_code in (200, 404),
            f"got {r.status_code}",
        )
        if r.status_code == 200:
            print(f"         Available logs: {r.json()}")
    except Exception as e:
        _fail("GET /api/logs/files", str(e))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="G-CHAT basic smoke test")
    parser.add_argument("--base-url", required=True, help="Server base URL, e.g. https://your-vm")
    parser.add_argument("--token", default="", help="JWT bearer token (from browser devtools)")
    parser.add_argument(
        "--chat",
        default="",
        help="Optional: a question to test the chat endpoint with",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    print(f"\nSmoke test → {base}")
    print(f"Token:  {'provided' if args.token else 'NOT provided — auth checks skipped'}")

    # Unauthenticated client for auth-guard checks
    anon = httpx.Client(timeout=15)
    # Authenticated client for everything else
    auth = httpx.Client(headers=headers, timeout=30, follow_redirects=True)

    # 1. Health — server must be up first
    alive = check_health(auth, base)
    if not alive:
        print("\n\nServer is not responding. Check that it's running and the URL is correct.")
        sys.exit(1)

    # 2. Metrics — no auth needed
    check_metrics(auth, base)

    # 3. Auth guard — unauthenticated requests must return 401 not 500
    check_unauth_returns_401(anon, base)

    if args.token:
        # 4. Auth/me
        user = check_auth_me(auth, base)

        # 5. Folders list
        if user:
            check_folders(auth, base)

        # 6. Logs API (admin only — will 403 for non-admin, which is fine)
        check_logs_api(auth, base)

        # 7. Chat — only if --chat provided
        if args.chat:
            check_chat(auth, base, args.chat)
        else:
            print("\n── Chat ────────────────────────────────────────────────────────────")
            print("  (skipped — pass --chat 'your question' to test chat)")
    else:
        print("\n  Skipping auth/folders/chat — no --token provided")
        print("  Get your token: open the app in the browser → DevTools → Application → Local Storage → look for 'access_token' or check the Authorization header in any request")

    # Summary
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'─' * 65}")
    print(f"  Result: {PASS_COUNT}/{total} passed")
    if FAIL_COUNT:
        print(f"  {FAIL_COUNT} check(s) FAILED — see above for details")
        sys.exit(1)
    else:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
