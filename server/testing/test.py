"""Simple one-shot LLM call for quick manual checks.

Uses the project's Azure OpenAI client (app.core.openai_client.get_client),
so it honours the same endpoint, deployment, and the DISABLE_GPT4O cost rule
(routes to gpt-4o-mini) as the rest of the backend.

Run:
    cd server
    uv run python testing/test.py                 # uses PROMPT below
    uv run python testing/test.py "your prompt"    # or pass it as an argument
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the script runnable from any cwd (e.g. from inside testing/): add the
# server root (parent of this testing/ dir) to sys.path so `import app` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AzureOpenAI

from app.core.config import get_settings

# ── Model selection ───────────────────────────────────────────────────────────
# True  -> gpt-4o  (uses GPT4O_DEPLOYMENT below, or AZURE_OPENAI_DEPLOYMENT)
# False -> gpt-4o-mini (AZURE_OPENAI_DEPLOYMENT_MINI)
# NOTE: this is a test-only override. It does NOT change the global DISABLE_GPT4O
# cost rule — production chat calls still route to gpt-4o-mini.
USE_GPT4O = False

# The gpt-4o deployment NAME as it exists in your Azure resource. On docwave-098
# only "gpt-4o-mini" is deployed, so plain "gpt-4o" returns DeploymentNotFound.
# Create a gpt-4o deployment in the Azure portal and paste its exact name here.
GPT4O_DEPLOYMENT = ""  # e.g. "gpt-4o" or "gpt-4o-prod"


def get_test_client():
    """Build an AzureOpenAI client + deployment for the chosen model, bypassing
    the DISABLE_GPT4O mini-routing (test only)."""
    s = get_settings()
    if USE_GPT4O:
        deployment = GPT4O_DEPLOYMENT or s.AZURE_OPENAI_DEPLOYMENT
    else:
        deployment = s.AZURE_OPENAI_DEPLOYMENT_MINI
    client = AzureOpenAI(
        azure_endpoint=s.AZURE_OPENAI_ENDPOINT or s.AZURE_OPENAI_API_BASE,
        api_key=s.AZURE_OPENAI_KEY or s.AZURE_OPENAI_API_KEY,
        api_version=s.AZURE_OPENAI_API_VERSION,
    )
    return client, deployment


# ── Paste your prompt here (or pass it as a CLI argument) ─────────────────────
PROMPT = "Show invoices that are on hold, including invoice number, hold reason, invoice date, and who placed the hold. -- OEBS SQL"

# Optional system message — leave as "" to send only the user prompt.
SYSTEM = "Show invoices that are on hold, including invoice number, hold reason, invoice date, and who placed the hold. -- OEBS SQL"


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else PROMPT

    client, deployment = get_test_client()

    messages = []
    if SYSTEM:
        messages.append({"role": "system", "content": SYSTEM})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.2,
    )

    print(f"--- model: {deployment} ---")
    print(f"--- prompt ---\n{prompt}\n")
    print(f"--- response ---\n{resp.choices[0].message.content}")
    usage = getattr(resp, "usage", None)
    if usage:
        print(
            f"\n--- tokens: prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} ---"
        )


if __name__ == "__main__":
    main()
