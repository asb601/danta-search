"""
danta-search Concurrent Stress Test — 30-40 virtual users.

Each virtual user:
  1. Fires a stream request for each question in the question bank.
  2. Measures time-to-first-token (TTFT) and total response time.
  3. Waits a random think-time between requests to avoid exact thundering-herd.

Usage
-----
# Step 1 — create test users on the server machine (or tunnel to DB)
cd server/
python -m testing.load_test_setup --count 40 --out /tmp/stress_tokens.json

# Step 2 — run the stress test against production
python testing/stress_test.py \\
    --tokens-file /tmp/stress_tokens.json \\
    --base-url https://genai.codeen.in.net \\
    --rounds 2 \\
    --think-time 2 \\
    --timeout 120 \\
    --out /tmp/stress_results.json

Flags
-----
--tokens-file   JSON produced by load_test_setup.py
--base-url      API base (no trailing slash)
--rounds        How many question-rounds each virtual user completes (default 2)
--think-time    Seconds of random pause between requests per user (default 2)
--timeout       Per-request timeout in seconds (default 120)
--out           Where to write the full JSON results (default /tmp/stress_results.json)
--questions     Comma-separated question overrides (optional)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import httpx

# ── Default question bank ─────────────────────────────────────────────────────
# Hard, multi-step ERP/finance questions that force the agent to:
#  - Join multiple files (invoices + vendors + cost centers)
#  - Do time-series calculations (MoM trends, aging buckets)
#  - Rank, filter, and aggregate in a single query
#  - Handle edge cases (nulls, date arithmetic, currency)
DEFAULT_QUESTIONS = [
    # Multi-file join: invoices + vendor master
    "Which vendors account for more than 20% of total AP spend in the last 90 days, "
    "and what is their average invoice payment cycle time in days?",

    # Time-series + ranking: requires GROUP BY month + RANK()
    "Show me a month-by-month breakdown of invoice volumes and total amounts for the "
    "past 6 months, highlighting the top 3 months by spend and any months where "
    "spend dropped more than 15% compared to the prior month.",

    # Aging analysis: requires date bucketing
    "Provide a full accounts payable aging report grouped into buckets: "
    "0-30 days, 31-60 days, 61-90 days, and over 90 days overdue. "
    "Include total amount and invoice count per bucket, and flag any vendor "
    "with more than 5 invoices in the 90+ days bucket.",

    # Multi-condition filter: requires compound WHERE + aggregation
    "List all purchase orders that were raised in Q1 of this fiscal year, "
    "are still open (not fully invoiced), have a value above 500000, "
    "and belong to the top 5 cost centers by total committed spend. "
    "Show PO number, vendor, cost center, committed amount, and age in days.",

    # Cross-file reconciliation
    "Find all invoices in the last 60 days where the invoiced amount differs "
    "from the corresponding purchase order amount by more than 10%. "
    "Show the PO number, invoice number, PO amount, invoice amount, "
    "variance percentage, and vendor name.",

    # Statistical outlier detection
    "Identify vendors whose average invoice amount in the current quarter "
    "is more than 2 standard deviations above their own 12-month historical average. "
    "List vendor name, 12-month average, current quarter average, and the z-score.",

    # Cost center drill-down with % contribution
    "For the top 10 cost centers by total spend this year, show their monthly "
    "spend trend for the past 4 months and calculate what percentage each "
    "cost center contributes to the overall company spend. "
    "Highlight any cost center that grew more than 25% month-over-month.",

    # Duplicate / anomaly detection
    "Are there any duplicate invoices in the system — same vendor, same amount, "
    "posted within 7 days of each other? List all duplicates with vendor name, "
    "invoice numbers, amounts, and posting dates, sorted by vendor.",
]


# ── Streaming request ─────────────────────────────────────────────────────────

async def _stream_request(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    timeout: float,
) -> dict[str, Any]:
    """POST to /api/chat/message/stream and collect timing + content."""
    url = f"{base_url}/api/chat/message/stream"
    payload = {"query": question, "conversation_id": None, "container_id": None}

    ttft: float | None = None
    full_text = ""
    t_start = time.perf_counter()

    try:
        async with client.stream(
            "POST", url, json=payload, timeout=timeout
        ) as resp:
            status = resp.status_code
            if status != 200:
                body = await resp.aread()
                return {
                    "ok": False,
                    "status": status,
                    "error": body.decode()[:200],
                    "ttft_s": None,
                    "total_s": round(time.perf_counter() - t_start, 3),
                }

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

                if ttft is None:
                    ttft = time.perf_counter() - t_start

                chunk = (
                    evt.get("content")
                    or evt.get("delta")
                    or evt.get("text")
                    or ""
                )
                full_text += chunk

        total_s = time.perf_counter() - t_start
        return {
            "ok": True,
            "status": 200,
            "ttft_s": round(ttft or total_s, 3),
            "total_s": round(total_s, 3),
            "chars": len(full_text),
            "answer_preview": full_text[:100].replace("\n", " "),
        }

    except httpx.TimeoutException as exc:
        return {
            "ok": False,
            "status": None,
            "error": f"TIMEOUT after {timeout}s",
            "ttft_s": round(ttft, 3) if ttft else None,
            "total_s": round(time.perf_counter() - t_start, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "error": str(exc)[:200],
            "ttft_s": None,
            "total_s": round(time.perf_counter() - t_start, 3),
        }


# ── Virtual user ──────────────────────────────────────────────────────────────

async def _virtual_user(
    user_index: int,
    token: str,
    email: str,
    base_url: str,
    questions: list[str],
    rounds: int,
    think_time: float,
    timeout: float,
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}"}
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        http2=False,
    ) as client:
        for round_num in range(1, rounds + 1):
            for q_idx, question in enumerate(questions):
                label = f"[u{user_index:02d} r{round_num} q{q_idx+1}]"
                print(f"  {label} → {question[:60]}…", flush=True)

                result = await _stream_request(client, base_url, question, timeout)
                result.update({
                    "user_index": user_index,
                    "email": email,
                    "round": round_num,
                    "question_index": q_idx,
                    "question": question,
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                results.append(result)

                if result["ok"]:
                    print(
                        f"  {label} ✓  TTFT={result['ttft_s']}s  "
                        f"total={result['total_s']}s  chars={result['chars']}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {label} ✗  status={result['status']}  "
                        f"err={result.get('error','?')[:80]}",
                        flush=True,
                    )

                # Think time between questions (jitter ±50%)
                if think_time > 0:
                    jitter = think_time * random.uniform(0.5, 1.5)
                    await asyncio.sleep(jitter)

    return results


# ── Metrics snapshot ──────────────────────────────────────────────────────────

async def _fetch_metrics(base_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            r = await client.get(f"{base_url}/api/metrics", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"error": str(exc)}


def _print_metrics(label: str, m: dict) -> None:
    print(f"\n── server metrics ({label}) ──")
    if "error" in m:
        print(f"  (unavailable: {m['error']})")
        return
    c = m.get("counters", {})
    lat = m.get("latency_ms", {})
    print(f"  query_total={c.get('query_total','n/a')}  "
          f"query_errors={c.get('query_errors','n/a')}")
    print(f"  p50={lat.get('p50_ms','n/a')}ms  "
          f"p95={lat.get('p95_ms','n/a')}ms  "
          f"p99={lat.get('p99_ms','n/a')}ms")


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(all_results: list[dict], wall_time: float) -> None:
    total = len(all_results)
    ok = [r for r in all_results if r.get("ok")]
    errors = [r for r in all_results if not r.get("ok")]
    timeouts = [r for r in errors if "TIMEOUT" in str(r.get("error", ""))]

    print("\n" + "=" * 65)
    print("STRESS TEST SUMMARY")
    print("=" * 65)
    print(f"  Total requests   : {total}")
    print(f"  Succeeded        : {len(ok)}")
    print(f"  Errors           : {len(errors)}  (timeouts: {len(timeouts)})")
    print(f"  Success rate     : {100*len(ok)/total:.1f}%")
    print(f"  Wall clock       : {round(wall_time, 1)}s")

    if ok:
        ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
        totals = [r["total_s"] for r in ok]

        def _pct(data: list[float], p: float) -> float:
            data_sorted = sorted(data)
            idx = int(len(data_sorted) * p / 100)
            return round(data_sorted[min(idx, len(data_sorted) - 1)], 3)

        print(f"\n  Time-to-first-token (TTFT)")
        if ttfts:
            print(f"    min={min(ttfts)}s   mean={round(statistics.mean(ttfts),3)}s   "
                  f"p50={_pct(ttfts,50)}s   p95={_pct(ttfts,95)}s   max={max(ttfts)}s")

        print(f"\n  Total response time")
        print(f"    min={min(totals)}s   mean={round(statistics.mean(totals),3)}s   "
              f"p50={_pct(totals,50)}s   p95={_pct(totals,95)}s   max={max(totals)}s")

        chars_list = [r["chars"] for r in ok if r.get("chars")]
        if chars_list:
            print(f"\n  Answer length (chars)")
            print(f"    min={min(chars_list)}   mean={round(statistics.mean(chars_list))}   "
                  f"max={max(chars_list)}")

    if errors:
        print(f"\n  Error breakdown:")
        by_type: dict[str, int] = {}
        for r in errors:
            key = str(r.get("error", "unknown"))[:60]
            by_type[key] = by_type.get(key, 0) + 1
        for msg, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {count}x  {msg}")

    print("=" * 65 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    ap = argparse.ArgumentParser(description="danta-search concurrent stress test")
    ap.add_argument(
        "--tokens-file",
        required=True,
        help="JSON file produced by load_test_setup.py",
    )
    ap.add_argument(
        "--base-url",
        default="https://genai.codeen.in.net",
        help="API base URL (no trailing slash)",
    )
    ap.add_argument("--rounds", type=int, default=2, help="Question rounds per user")
    ap.add_argument(
        "--think-time",
        type=float,
        default=2.0,
        help="Mean seconds between requests per user",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds",
    )
    ap.add_argument(
        "--out",
        default="/tmp/stress_results.json",
        help="Where to write full results JSON",
    )
    ap.add_argument(
        "--questions",
        default="",
        help="Comma-separated question overrides (optional)",
    )
    args = ap.parse_args()

    # Load tokens
    with open(args.tokens_file) as f:
        users = json.load(f)

    questions = (
        [q.strip() for q in args.questions.split(",") if q.strip()]
        if args.questions
        else DEFAULT_QUESTIONS
    )

    n_users = len(users)
    total_requests = n_users * args.rounds * len(questions)

    print(f"\n{'='*65}")
    print(f"danta-search STRESS TEST")
    print(f"{'='*65}")
    print(f"  Target       : {args.base_url}")
    print(f"  Virtual users: {n_users}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Questions/rnd: {len(questions)}")
    print(f"  Total reqs   : {total_requests}")
    print(f"  Think time   : ~{args.think_time}s (±50% jitter)")
    print(f"  Timeout      : {args.timeout}s")
    print(f"{'='*65}\n")

    # Server metrics before
    first_token = users[0]["token"]
    before_metrics = await _fetch_metrics(args.base_url, first_token)
    _print_metrics("before", before_metrics)
    print()

    # Fire all virtual users simultaneously
    t0 = time.perf_counter()
    tasks = [
        _virtual_user(
            user_index=i + 1,
            token=u["token"],
            email=u["email"],
            base_url=args.base_url,
            questions=questions,
            rounds=args.rounds,
            think_time=args.think_time,
            timeout=args.timeout,
        )
        for i, u in enumerate(users)
    ]

    print(f"Launching {n_users} virtual users simultaneously…\n")
    try:
        nested = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        print("\n[interrupted — collecting partial results…]")
        nested = []
    wall_time = time.perf_counter() - t0

    # return_exceptions=True means exceptions come back as values — filter them out
    all_results = [
        r
        for user_res in nested
        if isinstance(user_res, list)
        for r in user_res
    ]

    # Server metrics after
    after_metrics = await _fetch_metrics(args.base_url, first_token)
    _print_metrics("after", after_metrics)

    _print_summary(all_results, wall_time)

    # Write full results
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "n_users": n_users,
            "rounds": args.rounds,
            "questions": questions,
            "think_time": args.think_time,
            "timeout": args.timeout,
        },
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "results": all_results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full results written to {args.out}\n")


if __name__ == "__main__":
    asyncio.run(main())
