"""
Focused retrieval stability checks.

Run from server/:
    python3 -m testing._retrieval_stability_check
"""
from __future__ import annotations

import asyncio

from app.agent.catalog_hydration import hydrate_files
from app.core.orchestration_trace import OrchestrationTrace
from app.retrieval.fuzzy import fuzzy_search


class _Savepoint:
    def __init__(self, db: "_FailingDB") -> None:
        self._db = db

    async def __aenter__(self) -> "_Savepoint":
        self._db.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._db.exits.append(exc_type.__name__ if exc_type else None)
        return False


class _FailingDB:
    def __init__(self) -> None:
        self.entered = 0
        self.exits: list[str | None] = []

    def begin_nested(self) -> _Savepoint:
        return _Savepoint(self)

    async def execute(self, *args, **kwargs):
        raise RuntimeError("simulated optional read failure")


class _EmptyRows:
    def all(self) -> list:
        return []


class _TrgmUnavailableDB(_FailingDB):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def execute(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("function word_similarity(character varying, text) does not exist")
        return _EmptyRows()


async def test_fuzzy_failure_is_savepoint_scoped() -> None:
    db = _FailingDB()
    try:
        await fuzzy_search("invoice typo", "user-1", False, db)  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "simulated optional read failure" in str(exc)
    else:
        raise AssertionError("fuzzy_search should propagate to orchestrator after savepoint rollback")

    assert db.entered == 1
    assert db.exits == ["RuntimeError"]


async def test_hydration_failure_degrades_after_savepoint() -> None:
    db = _FailingDB()
    result = await hydrate_files(db, ["file-1"])  # type: ignore[arg-type]

    assert result == {}
    assert db.entered == 1
    assert db.exits == ["RuntimeError"]


async def test_fuzzy_pg_trgm_unavailable_uses_metadata_fallback() -> None:
    db = _TrgmUnavailableDB()
    result = await fuzzy_search("invoice matching", "user-1", False, db)  # type: ignore[arg-type]

    assert result == []
    assert db.calls == 2
    assert db.entered == 2
    assert db.exits == ["RuntimeError", None]


def test_retrieval_trace_records_stage_errors() -> None:
    trace = OrchestrationTrace(request_id="retrieval-stability-test")
    trace.set_retrieval_fusion(
        retrieved_with_scores=[],
        shortlist=[],
        resolver_pins=[],
        fallback=True,
        stage_errors=[{"stage": "fuzzy", "error": "pg_trgm unavailable"}],
    )

    payload = trace._stages["retrieval_fusion"]  # noqa: SLF001 - focused contract check
    assert payload["fallback"] is True
    assert payload["stage_error_count"] == 1
    assert payload["stage_errors"][0]["stage"] == "fuzzy"


async def main() -> None:
    await test_fuzzy_failure_is_savepoint_scoped()
    print("[PASS] test_fuzzy_failure_is_savepoint_scoped")
    await test_hydration_failure_degrades_after_savepoint()
    print("[PASS] test_hydration_failure_degrades_after_savepoint")
    await test_fuzzy_pg_trgm_unavailable_uses_metadata_fallback()
    print("[PASS] test_fuzzy_pg_trgm_unavailable_uses_metadata_fallback")
    test_retrieval_trace_records_stage_errors()
    print("[PASS] test_retrieval_trace_records_stage_errors")


if __name__ == "__main__":
    asyncio.run(main())