"""
Step 1 — Create 40 dummy users and write their JWT tokens to a JSON file.

Usage (from server/):
    python -m testing.load_test_setup [--count 40] [--out /tmp/stress_tokens.json]

Output: writes a JSON array with user_id / email / token to --out (default: /tmp/stress_tokens.json)
Then run the stress test:
    python testing/stress_test.py --tokens-file /tmp/stress_tokens.json \\
        --base-url https://genai.codeen.in.net --rounds 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt
from sqlalchemy import select

# ── Make sure server/ is on the path when run directly ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import get_settings
from app.core.database import async_session
from app.models.user import User

_NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
    "Nova", "Vega", "Lyra", "Orion",
]

def _build_dummy_users(count: int) -> list[dict]:
    users = []
    for i in range(count):
        name = _NAMES[i % len(_NAMES)]
        suffix = f"{i // len(_NAMES) + 1}" if i >= len(_NAMES) else ""
        users.append({
            "email": f"stress.{name.lower()}{suffix}@loadtest.internal",
            "name": f"Stress {name}{suffix}",
        })
    return users


def _mint_token(user_id: str, email: str, name: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": expire,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def main(count: int, out_path: str) -> None:
    settings = get_settings()
    print(f"\nConnecting to DB: {settings.DATABASE_URL[:60]}...\n")

    dummy_users = _build_dummy_users(count)
    created: list[dict] = []

    async with async_session() as db:
        for u in dummy_users:
            # Check if already exists
            result = await db.execute(select(User).where(User.email == u["email"]))
            existing = result.scalar_one_or_none()

            if existing:
                user_id = existing.id
                print(f"  EXISTS  {u['email']}  (id={user_id})")
            else:
                user_id = str(uuid.uuid4())
                new_user = User(
                    id=user_id,
                    email=u["email"],
                    name=u["name"],
                    is_admin=True,
                    role="admin",
                    allowed_domains=None,  # unrestricted — can query any container
                )
                db.add(new_user)
                print(f"  CREATED {u['email']}  (id={user_id})")

            token = _mint_token(user_id, u["email"], u["name"])
            created.append({
                "user_id": user_id,
                "email": u["email"],
                "name": u["name"],
                "token": token,
            })

        await db.commit()

    with open(out_path, "w") as f:
        json.dump(created, f, indent=2)

    print(f"\n✓ {len(created)} users written to {out_path}")
    print(f"  Run the stress test with:")
    print(f"  python testing/stress_test.py --tokens-file {out_path} \\")
    print(f"      --base-url https://genai.codeen.in.net --rounds 3\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40, help="Number of test users to create")
    ap.add_argument("--out", default="/tmp/stress_tokens.json", help="Output JSON path")
    args = ap.parse_args()
    asyncio.run(main(args.count, args.out))
