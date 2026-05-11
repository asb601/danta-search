"""
Step 1 — Create 5 dummy admin users and print their JWT tokens.

Usage (from server/):
    python -m testing.load_test_setup

Output: prints a JSON block with user IDs + tokens ready to paste into load_test_run.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt
from sqlalchemy import select

# ── Make sure server/ is on the path when run directly ───────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import get_settings
from app.core.database import async_session
from app.models.user import User

DUMMY_USERS = [
    {"email": "loadtest.alpha@gmail.com",   "name": "Load Test Alpha"},
    {"email": "loadtest.beta@gmail.com",    "name": "Load Test Beta"},
    {"email": "loadtest.gamma@gmail.com",   "name": "Load Test Gamma"},
    {"email": "loadtest.delta@gmail.com",   "name": "Load Test Delta"},
    {"email": "loadtest.epsilon@gmail.com", "name": "Load Test Epsilon"},
]


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


async def main() -> None:
    settings = get_settings()
    print(f"\nConnecting to DB: {settings.DATABASE_URL[:60]}...\n")

    created: list[dict] = []

    async with async_session() as db:
        for u in DUMMY_USERS:
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

    print("\n" + "=" * 70)
    print("PASTE THIS INTO load_test_run.py as USERS = [...]")
    print("=" * 70)
    print(json.dumps(created, indent=2))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
