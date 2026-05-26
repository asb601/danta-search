"""
Response-time benchmark for danta-search API.

Measures:
  • time-to-first-token (TTFT)
  • total time to complete response
  • tokens in final answer (rough)

Usage:
  python bench_response_time.py \
    --base-url https://genai.codeen.in.net \
    --token  <your JWT> \
    --question "how many rows are in the transactions table?" \
    --users 2 \
    --runs 3

The --token is your Bearer JWT from the browser:
  Open DevTools → Network → any /api/ request → copy the Authorization header value
  (everything after "Bearer ")
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx


# ── helpers ──────────────────────────────────────────────────────────────────

async def _stream_one(client: httpx.AsyncClient, base_url: str, question: str) -> dict:
    """POST to /api/chat/message/stream and measure timing."""
    url = f"{base_url}/api/chat/message/stream"
    payload = {"query": question}

    ttft: float | None = None
    full_text = ""
    t_start = time.perf_counter()

    async with client.stream("POST", url, json=payload, timeout=120) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # first token
            if ttft is None:
                ttft = time.perf_counter() - t_start

            chunk = evt.get("content") or evt.get("delta") or evt.get("text") or ""
            full_text += chunk

    t_total = time.perf_counter() - t_start
    return {
        "ttft_s": round(ttft or t_total, 3),
        "total_s": round(t_total, 3),
        "chars": len(full_text),
        "answer_preview": full_text[:120].replace("\n", " "),
    }


async def _run_user(
    user_id: int,
    base_url: str,
    token: str,
    question: str,
    runs: int,
) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    results = []
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        for i in range(runs):
            print(f"  [user {user_id}] run {i+1}/{runs} …", flush=True)
            try:
                r = await _stream_one(client, base_url, question)
                r["user"] = user_id
                r["run"] = i + 1
                results.append(r)
                print(
                    f"  [user {user_id}] run {i+1} → TTFT={r['ttft_s']}s  "
                    f"total={r['total_s']}s  chars={r['chars']}"
                )
            except Exception as exc:
                print(f"  [user {user_id}] run {i+1} ERROR: {exc}")
                results.append({"user": user_id, "run": i+1, "error": str(exc)})
    return results


async def _fetch_metrics(base_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            r = await client.get(f"{base_url}/api/metrics", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"error": str(exc)}


# ── main ─────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--token", required=True, help="JWT from browser devtools")
    ap.add_argument("--question", default="show me the top 5 rows of any table")
    ap.add_argument("--users", type=int, default=2, help="concurrent users")
    ap.add_argument("--runs", type=int, default=3, help="runs per user")
    args = ap.parse_args()

    print(f"\nBenchmark: {args.users} concurrent user(s) × {args.runs} run(s)")
    print(f"Target   : {args.base_url}")
    print(f"Question : {args.question}\n")

    # Snapshot metrics BEFORE
    print("── metrics snapshot (before) ──")
    before = await _fetch_metrics(args.base_url, args.token)
    if "error" not in before:
        counters = before.get("counters", {})
        print(f"  query_total={counters.get('query_total', 'n/a')}")
        print(f"  query_errors={counters.get('query_errors', 'n/a')}")
        latency = before.get("latency_ms", {})
        print(f"  p50={latency.get('p50_ms','n/a')}ms  p95={latency.get('p95_ms','n/a')}ms  p99={latency.get('p99_ms','n/a')}ms")
    else:
        print(f"  (metrics endpoint unavailable: {before['error']})")

    print()

    # Run concurrent users
    t0 = time.perf_counter()
    tasks = [
        _run_user(uid + 1, args.base_url, args.token, args.question, args.runs)
        for uid in range(args.users)
    ]
    all_results_nested = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - t0

    # Flatten
    all_results = [r for user_res in all_results_nested for r in user_res]
    ok = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]

    print(f"\n── results summary ──")
    print(f"  total requests : {len(all_results)}")
    print(f"  succeeded      : {len(ok)}")
    print(f"  errors         : {len(errors)}")
    print(f"  wall clock     : {round(wall_time, 2)}s")

    if ok:
        ttfts  = [r["ttft_s"]  for r in ok]
        totals = [r["total_s"] for r in ok]
        print(f"\n  time-to-first-token (TTFT)")
        print(f"    min={min(ttfts)}s  max={max(ttfts)}s  "
              f"mean={round(statistics.mean(ttfts),3)}s  "
              + (f"stdev={round(statistics.stdev(ttfts),3)}s" if len(ttfts) > 1 else ""))
        print(f"\n  total response time")
        print(f"    min={min(totals)}s  max={max(totals)}s  "
              f"mean={round(statistics.mean(totals),3)}s  "
              + (f"stdev={round(statistics.stdev(totals),3)}s" if len(totals) > 1 else ""))
        print(f"\n  sample answers:")
        for r in ok[:2]:
            print(f"    [user {r['user']} run {r['run']}] "{r['answer_preview']}…"")

    if errors:
        print(f"\n  errors:")
        for r in errors:
            print(f"    [user {r['user']} run {r['run']}] {r['error']}")

    # Snapshot metrics AFTER
    print("\n── metrics snapshot (after) ──")
    after = await _fetch_metrics(args.base_url, args.token)
    if "error" not in after:
        counters = after.get("counters", {})
        print(f"  query_total={counters.get('query_total', 'n/a')}")
        print(f"  query_errors={counters.get('query_errors', 'n/a')}")
        latency = after.get("latency_ms", {})
        print(f"  p50={latency.get('p50_ms','n/a')}ms  p95={latency.get('p95_ms','n/a')}ms  p99={latency.get('p99_ms','n/a')}ms")
    else:
        print(f"  (metrics endpoint unavailable: {after['error']})")

    print()


if __name__ == "__main__":
    asyncio.run(main())
