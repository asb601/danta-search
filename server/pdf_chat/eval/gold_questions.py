"""Gold-question dataclass + loader. Data lives in gold_set.json (not code)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_GOLD_PATH = Path(__file__).with_name("gold_set.json")


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    must_cite: bool = True
    expect_refusal: bool = False
    # Phase-6 routing category (DATA, not a rule): local | graph_traversal |
    # global_community | cross_domain | negative_claim. Defaults to "local" so
    # pre-Phase-6 rows (and ad-hoc constructions) stay valid.
    category: str = "local"


def load_gold_set(path: "Path | None" = None) -> list[GoldQuestion]:
    """Load the seed gold-question set from JSON."""
    raw = json.loads((path or _GOLD_PATH).read_text(encoding="utf-8"))
    return [GoldQuestion(**row) for row in raw]
