"""
Quality test — fires each hard question against the live server ONE AT A TIME
(no concurrency, no queue pressure) and captures:
  - Full answer text
  - TTFT, total time
  - Tool calls made (thinking events)
  - Data rows returned
  - Whether the answer is substantive or a "no data found" fallback

Results stored as JSON + a human-readable text report.

Usage
-----
cd server/
uv run python ../testing/quality_test.py \\
    --token-file /tmp/stress_tokens_10.json \\
    --base-url https://genai.codeen.in.net \\
    --out /tmp/quality_results.json \\
    --report /tmp/quality_report.txt
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import httpx

HARD_QUESTIONS = [
    {
        "id": "Q1",
        "label": "Vendor AP concentration + payment cycle",
        "question": (
            "Which vendors account for more than 20% of total AP spend in the last 90 days, "
            "and what is their average invoice payment cycle time in days?"
        ),
    },
    {
        "id": "Q2",
        "label": "MoM invoice trend with drop detection",
        "question": (
            "Show me a month-by-month breakdown of invoice volumes and total amounts for the "
            "past 6 months, highlighting the top 3 months by spend and any months where "
            "spend dropped more than 15% compared to the prior month."
        ),
    },
    {
        "id": "Q3",
        "label": "Full AP aging report with vendor flag",
        "question": (
            "Provide a full accounts payable aging report grouped into buckets: "
            "0-30 days, 31-60 days, 61-90 days, and over 90 days overdue. "
            "Include total amount and invoice count per bucket, and flag any vendor "
            "with more than 5 invoices in the 90+ days bucket."
        ),
    },
    {
        "id": "Q4",
        "label": "Open POs in Q1, top 5 cost centers, high value",
        "question": (
            "List all purchase orders that were raised in Q1 of this fiscal year, "
            "are still open (not fully invoiced), have a value above 500000, "
            "and belong to the top 5 cost centers by total committed spend. "
            "Show PO number, vendor, cost center, committed amount, and age in days."
        ),
    },
    {
        "id": "Q5",
        "label": "PO vs invoice variance reconciliation",
        "question": (
            "Find all invoices in the last 60 days where the invoiced amount differs "
            "from the corresponding purchase order amount by more than 10%. "
            "Show the PO number, invoice number, PO amount, invoice amount, "
            "variance percentage, and vendor name."
        ),
    },
    {
        "id": "Q6",
        "label": "Statistical outlier vendors (z-score)",
        "question": (
            "Identify vendors whose average invoice amount in the current quarter "
            "is more than 2 standard deviations above their own 12-month historical average. "
            "List vendor name, 12-month average, current quarter average, and the z-score."
        ),
    },
    {
        "id": "Q7",
        "label": "Cost center spend trend + MoM growth %",
        "question": (
            "For the top 10 cost centers by total spend this year, show their monthly "
            "spend trend for the past 4 months and calculate what percentage each "
            "cost center contributes to the overall company spend. "
            "Highlight any cost center that grew more than 25% month-over-month."
        ),
    },
    {
        "id": "Q8",
        "label": "Duplicate invoice detection",
        "question": (
            "Are there any duplicate invoices in the system — same vendor, same amount, "
            "posted within 7 days of each other? List all duplicates with vendor name, "
            "invoice numbers, amounts, and posting dates, sorted by vendor."
        ),
    },
]


def run_question(base_url: str, token: str, question: str, timeout: float = 120.0) -> dict:
    """Send one streaming question, return timing + full answer."""
    url = f"{base_url}/api/v1/message/stream"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {"query": question, "conversation_id": None}

    result = {
        "ok": False,
        "ttft_s": None,
        "total_s": None,
        "answer": "",
        "tool_calls": [],
        "data_rows": 0,
        "from_cache": False,
        "error": None,
    }

    t_start = time.perf_counter()
    first_token_at = None

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    result["error"] = f"HTTP {resp.status_code}"
                    result["total_s"] = round(time.perf_counter() - t_start, 3)
                    return result

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    evt_type = evt.get("event")

                    if evt_type == "token":
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        result["answer"] += evt.get("content", "")

                    elif evt_type == "thinking":
                        tool = evt.get("tool", "")
                        if tool and tool not in result["tool_calls"]:
                            result["tool_calls"].append(tool)

                    elif evt_type == "done":
                        payload = evt.get("result", {})
                        result["data_rows"] = payload.get("row_count", 0)
                        result["from_cache"] = payload.get("from_cache", False)
                        result["ok"] = True

    except httpx.TimeoutException:
        result["error"] = f"TIMEOUT after {timeout}s"
    except Exception as e:
        result["error"] = str(e)[:200]

    t_end = time.perf_counter()
    result["total_s"] = round(t_end - t_start, 3)
    if first_token_at is not None:
        result["ttft_s"] = round(first_token_at - t_start, 3)

    return result


def grade_answer(answer: str, question_id: str) -> str:
    """Simple heuristic quality grade based on answer content."""
    if not answer or len(answer) < 50:
        return "EMPTY"
    lower = answer.lower()
    no_data_phrases = [
        "no data", "no records", "no invoices", "no results",
        "not found", "could not find", "unable to find",
        "no matching", "i checked", "i was unable",
    ]
    if any(p in lower for p in no_data_phrases):
        return "NO_DATA"
    # Check for numbers — a real answer should have numeric content
    import re
    has_numbers = bool(re.search(r'\$[\d,]+|\d+[\.,]\d+|\d{3,}', answer))
    has_structure = any(x in answer for x in ["**", "|", "1.", "- ", "•"])
    if has_numbers and has_structure:
        return "GOOD"
    if has_numbers:
        return "PARTIAL"
    return "VAGUE"


def build_report(results: list[dict]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("QUALITY TEST REPORT")
    lines.append(f"Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 72)
    lines.append("")

    total = len(results)
    ok = sum(1 for r in results if r["ok"])
    grades = [r["grade"] for r in results]

    lines.append(f"Questions: {total}   Answered: {ok}   Failed: {total - ok}")
    lines.append(f"Grades: GOOD={grades.count('GOOD')}  PARTIAL={grades.count('PARTIAL')}  "
                 f"VAGUE={grades.count('VAGUE')}  NO_DATA={grades.count('NO_DATA')}  "
                 f"EMPTY={grades.count('EMPTY')}")
    lines.append("")

    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    if ttfts:
        lines.append(f"TTFT:  min={min(ttfts):.1f}s  avg={sum(ttfts)/len(ttfts):.1f}s  max={max(ttfts):.1f}s")
    lines.append("")
    lines.append("-" * 72)

    for r in results:
        lines.append(f"\n[{r['id']}] {r['label']}")
        lines.append(f"  Grade   : {r['grade']}")
        lines.append(f"  TTFT    : {r['ttft_s']}s   Total: {r['total_s']}s")
        lines.append(f"  Data rows returned: {r['data_rows']}")
        lines.append(f"  Tool calls: {', '.join(r['tool_calls']) or 'none'}")
        lines.append(f"  From cache: {r['from_cache']}")
        if r["error"]:
            lines.append(f"  ERROR   : {r['error']}")
        lines.append(f"  Question: {r['question'][:120]}")
        lines.append("")
        if r["answer"]:
            lines.append("  --- FULL ANSWER ---")
            for line in r["answer"].split("\n"):
                lines.append(f"  {line}")
            lines.append("  --- END ANSWER ---")
        else:
            lines.append("  (no answer)")
        lines.append("-" * 72)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", default="/tmp/stress_tokens_10.json")
    ap.add_argument("--base-url", default="https://genai.codeen.in.net")
    ap.add_argument("--out", default="/tmp/quality_results.json")
    ap.add_argument("--report", default="/tmp/quality_report.txt")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    with open(args.token_file) as f:
        tokens = json.load(f)
    token = tokens[0]["token"]  # use first user, serial (no concurrency)

    print(f"Target : {args.base_url}")
    print(f"User   : {tokens[0]['email']}")
    print(f"Questions: {len(HARD_QUESTIONS)}")
    print("=" * 60)

    all_results = []

    for q in HARD_QUESTIONS:
        print(f"\n[{q['id']}] {q['label']}")
        print(f"  Q: {q['question'][:100]}...")
        r = run_question(args.base_url, token, q["question"], timeout=args.timeout)
        r["id"] = q["id"]
        r["label"] = q["label"]
        r["question"] = q["question"]
        r["grade"] = grade_answer(r["answer"], q["id"])

        status = "✓" if r["ok"] else "✗"
        grade_color = {"GOOD": "GOOD", "PARTIAL": "PARTIAL", "NO_DATA": "NO_DATA",
                       "VAGUE": "VAGUE", "EMPTY": "EMPTY"}.get(r["grade"], "?")
        print(f"  {status}  TTFT={r['ttft_s']}s  Total={r['total_s']}s  Grade={grade_color}")
        print(f"     Tools: {r['tool_calls']}")
        if r["answer"]:
            print(f"     Answer preview: {r['answer'][:200]}...")
        if r["error"]:
            print(f"     Error: {r['error']}")

        all_results.append(r)

    # Write JSON
    with open(args.out, "w") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "results": all_results,
        }, f, indent=2)
    print(f"\nFull JSON written to {args.out}")

    # Write text report
    report = build_report(all_results)
    with open(args.report, "w") as f:
        f.write(report)
    print(f"Text report written to {args.report}")

    # Print summary
    print("\n" + "=" * 60)
    goods = sum(1 for r in all_results if r["grade"] == "GOOD")
    no_data = sum(1 for r in all_results if r["grade"] == "NO_DATA")
    errors = sum(1 for r in all_results if not r["ok"])
    print(f"GOOD: {goods}/{len(HARD_QUESTIONS)}   NO_DATA: {no_data}   ERRORS: {errors}")


if __name__ == "__main__":
    main()
