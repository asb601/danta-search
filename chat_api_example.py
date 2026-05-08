"""
chat_api_example.py
────────────────────────────────────────────────────────────────────
How to call the G-CHAT streaming API from Python.

SETUP
-----
pip install httpx python-dotenv

Create a .env file (or export these vars):
  BASE_URL=https://your-server.com       # no trailing slash
  TOKEN=<your JWT token>                 # see "How to get the TOKEN" below

HOW TO GET THE TOKEN
────────────────────
There is no separate API key yet.  The TOKEN is your JWT issued
after Google OAuth login.

To copy it:
  1. Open the app in your browser and log in.
  2. Open DevTools → Application → Local Storage → your domain.
  3. Copy the value of the "token" key.
  4. Paste it as TOKEN=... in your .env.

Note: JWTs expire (default 60 min).  Re-copy after expiry or
      increase ACCESS_TOKEN_EXPIRE_MINUTES in the server .env.

──────────────────────────────────────────────────────────────────── 
"""

import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
TOKEN    = os.getenv("TOKEN")

if not TOKEN:
    raise SystemExit("ERROR: TOKEN is not set.  See the instructions above.")

ENDPOINT = f"{BASE_URL}/api/chat/message/stream"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# ── Payload ────────────────────────────────────────────────────────────────────
# query          (required) — the user's question
# conversation_id (optional) — pass a UUID to continue an existing conversation;
#                               omit (or set null) to start a new one.
# ──────────────────────────────────────────────────────────────────────────────
PAYLOAD = {
    "query": "What is the total invoice amount for last month?",
    "conversation_id": None,   # None = start a new conversation
}


def stream_chat(query: str, conversation_id: str | None = None):
    """
    Streams SSE events from the chat endpoint and prints them.

    Events you will receive (in order):
      started         → conversation created / resumed
      pipeline_step   → retrieval phase (how many files were searched)
      thinking        → which tool the agent is running
      token           → one chunk of the LLM's answer text
      done            → final result containing answer + data rows + chart meta
      error           → something went wrong
    """
    payload = {"query": query, "conversation_id": conversation_id}

    with httpx.stream("POST", ENDPOINT, headers=HEADERS, json=payload, timeout=120) as resp:
        if resp.status_code != 200:
            resp.read()  # load body so we can print it
            print(f"ERROR {resp.status_code}: {resp.text}")
            return

        buffer = ""
        for raw_chunk in resp.iter_text():
            buffer += raw_chunk
            # SSE lines are separated by \n; an event ends with a blank line.
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data: "):
                    continue

                try:
                    event = json.loads(line[6:])   # strip the "data: " prefix
                except json.JSONDecodeError:
                    continue

                evt = event.get("event")

                if evt == "started":
                    print(f"[started] conversation_id = {event.get('conversation_id')}")

                elif evt == "pipeline_step" and event.get("step") == "retrieval":
                    r = event.get("retrieved_files", 0)
                    t = event.get("total_files", 0)
                    print(f"[retrieval] searching {r}/{t} files…")

                elif evt == "thinking":
                    print(f"[thinking] running tool: {event.get('tool')}")

                elif evt == "token":
                    # Stream answer text to stdout without newlines between chunks
                    print(event.get("content", ""), end="", flush=True)

                elif evt == "done":
                    result = event.get("result", {})
                    print()   # newline after streamed tokens
                    print("\n── Final answer ──")
                    print(result.get("answer", ""))
                    rows = result.get("data", [])
                    if rows:
                        print(f"\n── Data ({len(rows)} rows) ──")
                        # Print column headers
                        cols = list(rows[0].keys())
                        print(" | ".join(cols))
                        print("-" * (len(" | ".join(cols)) + 2))
                        for row in rows[:10]:  # show first 10 rows
                            print(" | ".join(str(row.get(c, "")) for c in cols))
                        if len(rows) > 10:
                            print(f"… {len(rows) - 10} more rows")

                elif evt == "error":
                    print(f"\n[error] {event.get('detail')}")


if __name__ == "__main__":
    stream_chat(
        query=PAYLOAD["query"],
        conversation_id=PAYLOAD["conversation_id"],
    )
