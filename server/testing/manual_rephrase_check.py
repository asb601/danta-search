"""Manual, REAL check of the query rephraser — see the ERP rewrite for yourself.

Unlike test_query_rephraser.py (which mocks the LLM), this script makes a REAL
gpt-4o-mini call so you can eyeball whether the rewrite meets your expectations.
It needs your Azure OpenAI env (.env) configured, same as running the server.

Run from the server/ directory:

    # one query
    PYTHONPATH=. uv run python testing/manual_rephrase_check.py "show me overdue invoces by vndr last month"

    # several queries at once
    PYTHONPATH=. uv run python testing/manual_rephrase_check.py "top 5 customers by sales" "open POs not yet received"

    # no args → runs the built-in sample prompts below
    PYTHONPATH=. uv run python testing/manual_rephrase_check.py

    # interactive — type a prompt, see the rewrite, repeat (Ctrl-D / 'quit' to stop)
    PYTHONPATH=. uv run python testing/manual_rephrase_check.py -i
"""
from __future__ import annotations

import asyncio
import os
import sys

# Make `app` importable no matter where this is run from (e.g. inside testing/).
# server/ is the parent of this file's directory.
_SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from app.core.config import get_settings  # noqa: E402
from app.services.query_rephraser import rephrase_query  # noqa: E402


# Messy / terse ERP-flavoured prompts to exercise the rewrite. Edit freely.
SAMPLE_PROMPTS = [
    "List assets using MACRS depreciation placed in service after 2020. -- OEBS  ",
]

# Domain appended to every rewrite (in the real app this comes from the selected
# folder's domain_tag / domain filter, NOT from the question text). Set to None
# to test without a domain. Edit to match how your folders are tagged.
DOMAIN = "OEBS"


def _print_result(query: str, result) -> None:
    print("─" * 78)
    print(f"ORIGINAL  : {query}")
    print(f"REWRITTEN : {result.text}")
    print(f"changed={result.changed}   reason={result.reason}")


async def _run(prompts: list[str]) -> None:
    s = get_settings()
    print(f"QUERY_REPHRASE_ENABLED = {getattr(s, 'QUERY_REPHRASE_ENABLED', None)}")
    rephrase_dep = getattr(s, "QUERY_REPHRASE_DEPLOYMENT", "") or s.AZURE_OPENAI_DEPLOYMENT_MINI
    print(f"rephrase deployment    = {rephrase_dep!r}"
          f"  (QUERY_REPHRASE_DEPLOYMENT={getattr(s, 'QUERY_REPHRASE_DEPLOYMENT', '')!r}, "
          f"falls back to mini={s.AZURE_OPENAI_DEPLOYMENT_MINI!r})")
    if not getattr(s, "QUERY_REPHRASE_ENABLED", False):
        print("\n⚠  Rephrasing is DISABLED — every result will return the original "
              "unchanged. Set QUERY_REPHRASE_ENABLED=true in .env to test the rewrite.")

    print(f"domain (appended)      = {DOMAIN!r}")
    for q in prompts:
        try:
            result = await rephrase_query(q, domain=DOMAIN)
        except Exception as exc:  # the function shouldn't raise, but be loud if it does
            print("─" * 78)
            print(f"ORIGINAL  : {q}")
            print(f"!! ERROR  : {exc!r}")
            continue
        _print_result(q, result)
    print("─" * 78)


async def _interactive() -> None:
    s = get_settings()
    rephrase_dep = getattr(s, "QUERY_REPHRASE_DEPLOYMENT", "") or s.AZURE_OPENAI_DEPLOYMENT_MINI
    print(f"Interactive rephrase check (QUERY_REPHRASE_ENABLED="
          f"{getattr(s, 'QUERY_REPHRASE_ENABLED', None)}, deployment={rephrase_dep!r}). "
          "Type a prompt; blank line, 'quit', or Ctrl-D to exit.\n")
    while True:
        try:
            q = input("query> ").strip()
        except EOFError:
            print()
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        result = await rephrase_query(q, domain=DOMAIN)
        _print_result(q, result)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in {"-i", "--interactive"}:
        asyncio.run(_interactive())
        return
    prompts = args if args else SAMPLE_PROMPTS
    asyncio.run(_run(prompts))


if __name__ == "__main__":
    main()
