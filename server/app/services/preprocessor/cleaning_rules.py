"""Pluggable data-cleaning rule registry.

Design mirrors type_detection.py — each cleaning rule is an isolated class.
New rules are added by creating a new class and registering it.
Existing code is NOT modified to accommodate new rules.

Public API
----------
    get_cleaning_profile(
        extra_null_patterns: Sequence[str] = (),
        extra_garbage_re_patterns: Sequence[str] = (),
    ) -> CleaningProfile

    CleaningProfile
        .nullify_series(s: pd.Series) -> pd.Series
            Drop-in for the old _nullify_series(s) — cell-level null normalisation.
        .clean_rows(chunk: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]
            Drop garbage / empty rows and return an audit sample of what was dropped.

Per-container configuration
----------------------------
Store custom rules in ContainerConfig.cleaning_config (JSONB):

    {
        "extra_null_patterns":      ["k.a.", "N/V", "#VALUE!"],
        "extra_garbage_patterns":   [".*Zwischensumme.*", ".*balance forward.*"]
    }

Load at ingest time and pass to get_cleaning_profile() — no code deploy needed.

Extending the built-in rules
-----------------------------
    Add a new universal null pattern  → add one string to NullPatternRule.KNOWN_NULLS
    Add a new garbage-row pattern     → add one regex string to GarbageKeywordRule.PATTERNS
    Add a new row-level rule type     → implement RowClassifier, append to _DEFAULT_ROW_CLASSIFIERS

None of the above require changes to data_preprocessor.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import pandas as pd

from app.core.logger import ingest_logger

# Maximum number of dropped rows to keep as an audit sample.
# Stored in FileAnalytics.quarantine_sample — large values add DB write overhead.
MAX_QUARANTINE_SAMPLE = 20


# ══════════════════════════════════════════════════════════════════════════════
# RowClassifier Protocol
# ══════════════════════════════════════════════════════════════════════════════

class RowClassifier(Protocol):
    """Contract for a single row-level drop rule.

    Implementations must be stateless and cheap to instantiate.
    The mask() method must be vectorized — no Python row loops.
    Adding a new classifier = creating a new class, nothing else.
    """

    name: str

    def mask(self, str_df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series (same index as str_df) — True = drop this row.

        str_df: every cell has been .fillna("").astype(str).str.strip() already.
        Must NOT contain Python row loops — use pandas vectorized operations.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Built-in Rule 1 — All-empty rows
# ══════════════════════════════════════════════════════════════════════════════

class AllEmptyRowRule:
    """Drop rows where every cell is empty after stripping whitespace."""

    name = "all_empty"

    def mask(self, str_df: pd.DataFrame) -> pd.Series:
        return (str_df == "").all(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Built-in Rule 2 — Garbage keyword rows (totals, subtotals, report summaries)
# ══════════════════════════════════════════════════════════════════════════════

class GarbageKeywordRule:
    """Drop rows whose first non-empty cell starts with a known summary keyword.

    Covers: English, German (SAP/Oracle), French, Spanish/Portuguese ERP patterns.

    To add support for a new language or export format:
        Append regex alternatives to GarbageKeywordRule.PATTERNS.
        No other code changes needed.

    To add patterns for one specific container only:
        Set ContainerConfig.cleaning_config["extra_garbage_patterns"] = [...]
        These are passed to __init__(extra_patterns=...) at ingest time.
    """

    name = "garbage_keyword"

    # Each string is a regex alternative joined with | at __init__ time.
    # All lowercase; re.IGNORECASE applied at compile time.
    PATTERNS: tuple[str, ...] = (
        r"total",
        r"grand\s+total",
        r"subtotal",
        r"sub\s+total",
        r"sum",
        r"page\s+total",
        r"running\s+total",
        r"end\s+of\s+report",
        r"average",
        r"avg",
        r"mean",
        r"balance\s+forward",
        r"carried\s+forward",
        r"min",
        r"max",
        # ── German (SAP, Oracle EBS, other German ERP exports) ────────────────
        r"summe",
        r"gesamtsumme",
        r"gesamt",
        r"zwischensumme",
        r"durchschnitt",
        r"gesamt\s+ergebnis",
        # ── French ────────────────────────────────────────────────────────────
        # Build with literal codepoints instead of \uXXXX escapes so the pattern
        # is safe under both Python's re engine and PyArrow's RE2 (which does not
        # accept Python-style Unicode escapes inside character classes).
        r"total\s+g[e" + "\u00e9" + r"]n[e" + "\u00e9" + r"]ral",
        r"total\s+partiel",
        r"moyenne",
        # ── Spanish / Portuguese ───────────────────────────────────────────────
        r"total\s+general",
        r"suma\s+total",
        r"promedio",
        r"m[e" + "\u00e9" + r"]dia",
        # ── Japanese ERP ──────────────────────────────────────────────────────
        # Use literal Unicode chars, not \uXXXX escapes — RE2 (PyArrow) does not
        # accept Python-style \u escapes in regex patterns.
        "合計",   # gōkei — total
        "小計",   # shōkei — subtotal
    )

    def __init__(self, extra_patterns: Sequence[str] = ()) -> None:
        all_patterns = list(self.PATTERNS) + [p for p in extra_patterns if p.strip()]
        # (?:\b|$) so ASCII terms use word-boundary and CJK terms (not ASCII word
        # chars) fall back to end-of-string anchor.
        combined = r"^\s*(" + "|".join(all_patterns) + r")(?:\b|$)"
        self._re = re.compile(combined, re.IGNORECASE)

    def mask(self, str_df: pd.DataFrame) -> pd.Series:
        return str_df.iloc[:, 0].str.match(self._re, na=False)


# ══════════════════════════════════════════════════════════════════════════════
# Built-in Rule 3 — Separator rows (----, ====, ****)
# ══════════════════════════════════════════════════════════════════════════════

class SeparatorRowRule:
    """Drop visual separator rows inserted by ERP exports between report sections.

    A row is a separator row when:
      - it has at least one non-empty cell, AND
      - every non-empty cell consists entirely of separator characters.
    """

    name = "separator_row"

    _SEP_RE = re.compile(r"^[-=*_~\s|+]+$")

    def mask(self, str_df: pd.DataFrame) -> pd.Series:
        non_empty = str_df != ""
        sep_cell  = str_df.apply(lambda col: col.str.match(self._SEP_RE, na=False))
        return (~non_empty | sep_cell).all(axis=1) & non_empty.any(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# NullPatternRule — cell-level null normalisation (not a RowClassifier)
# ══════════════════════════════════════════════════════════════════════════════

class NullPatternRule:
    """Recognises null-like cell values and replaces them with None.

    Built-in set covers English, German (SAP/Oracle), French, Spanish/Portuguese,
    and common data-quality placeholders.

    Extending the universal set:
        Add one lowercase string to NullPatternRule.KNOWN_NULLS.
        No other code changes.

    Per-container extras (no code change needed):
        Pass extra_patterns to __init__().
        Source: ContainerConfig.cleaning_config["extra_null_patterns"].
    """

    KNOWN_NULLS: frozenset[str] = frozenset({
        # ── Standard ─────────────────────────────────────────────────────────
        "", "null", "none", "na", "n/a", "nan", "nil", "tbd", "n.a.", "n.a",
        "-", "--", "---", ".", "..", "?", "#", "#n/a", "#na", "#null!",
        "not available", "not applicable", "not provided", "not assigned",
        "missing", "unknown", "no data", "no value", "nd", "n.d.",
        "void", "blank", "empty",
        # ── Excel error values ────────────────────────────────────────────────
        "#value!", "#ref!", "#div/0!", "#name?", "#num!", "#error!",
        # ── SAP / German ERP ─────────────────────────────────────────────────
        "n/v",           # nicht verfügbar (not available)
        "k.a.", "k.a",   # keine Angabe (no info)
        "n.v.", "n.v",   # nicht vorhanden (not present)
        # ── Spanish / French ERP ──────────────────────────────────────────────
        "s/o", "s/n",    # sin orden, sin número
        "n/e",           # not entered / no especificado
        # ── Misc data-quality placeholders ───────────────────────────────────
        "tba",           # to be advised
        "wip",           # work in progress
        "pending",
        "n.a.a",
        "na.",
        "not entered",
        "not recorded",
        "not applicable.",
    })

    def __init__(self, extra_patterns: Sequence[str] = ()) -> None:
        extras = frozenset(p.strip().lower() for p in extra_patterns if p.strip())
        self.null_strings: frozenset[str] = self.KNOWN_NULLS | extras

    def nullify_series(self, s: pd.Series) -> pd.Series:
        """Vectorized null normalisation — one isin() call, no Python loop.

        Used as: chunk.apply(profile.nullify_series)
        """
        if s.dtype.kind not in ("O", "U"):
            return s
        lower     = s.astype(str).str.strip().str.lower()
        null_mask = lower.isin(self.null_strings) | s.isna()
        return s.where(~null_mask, other=None)


# ══════════════════════════════════════════════════════════════════════════════
# CleaningProfile — assembled view of all rules for one preprocessing run
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CleaningProfile:
    """Assembled cleaning configuration for one preprocessing run.

    Created by get_cleaning_profile() — do not instantiate directly.

    Attributes:
        null_rule:        cell-level null normalisation
        row_classifiers:  ordered list of row-level drop rules
    """

    null_rule:        NullPatternRule
    row_classifiers:  list  # list[RowClassifier]

    # ── Cell-level ──────────────────────────────────────────────────────────

    def nullify_series(self, s: pd.Series) -> pd.Series:
        """Drop-in replacement for the old _nullify_series(s).

        Usage:  chunk.apply(profile.nullify_series)
        """
        return self.null_rule.nullify_series(s)

    # ── Row-level ────────────────────────────────────────────────────────────

    def clean_rows(
        self,
        chunk: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Drop garbage / empty rows. Return (clean_chunk, quarantine_sample).

        quarantine_sample is a list of at most MAX_QUARANTINE_SAMPLE dicts:
            {"reason": "<classifier_name>", "row": {"col": "value", ...}}

        Stored in FileAnalytics.quarantine_sample for audit.

        Performance:
            - All masks computed vectorized (no Python row loops for bulk work).
            - Per-row reason lookup is O(classifiers × sample_size) — bounded by
              MAX_QUARANTINE_SAMPLE, so never touches the full chunk size.
        """
        if chunk.empty:
            return chunk, []

        str_df = chunk.fillna("").astype(str).apply(lambda s: s.str.strip())

        combined_mask          = pd.Series(False, index=chunk.index)
        classifier_masks: list[tuple[str, pd.Series]] = []

        for cls in self.row_classifiers:
            m = cls.mask(str_df)
            classifier_masks.append((cls.name, m))
            combined_mask |= m

        # Build audit sample for the first MAX_QUARANTINE_SAMPLE dropped rows.
        # Uses a short Python loop — capped by MAX_QUARANTINE_SAMPLE, not file size.
        quarantine: list[dict] = []
        dropped_indices        = combined_mask[combined_mask].index.tolist()

        for idx in dropped_indices[:MAX_QUARANTINE_SAMPLE]:
            reason = "unknown"
            for cls_name, m in classifier_masks:
                if idx in m.index and bool(m.loc[idx]):
                    reason = cls_name
                    break
            try:
                row_data = chunk.loc[idx].fillna("").astype(str).to_dict()
            except Exception:
                row_data = {}
            quarantine.append({"reason": reason, "row": row_data})

        clean = chunk[~combined_mask].copy()
        return clean, quarantine


# ══════════════════════════════════════════════════════════════════════════════
# Default registry (built-in rules in priority order)
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_ROW_CLASSIFIERS: list = [
    AllEmptyRowRule(),
    GarbageKeywordRule(),    # uses class-level PATTERNS, no per-container extras
    SeparatorRowRule(),
]


def get_cleaning_profile(
    extra_null_patterns: Sequence[str] = (),
    extra_garbage_re_patterns: Sequence[str] = (),
) -> CleaningProfile:
    """Build a CleaningProfile for one preprocessing run.

    Args:
        extra_null_patterns:
            Container-level null strings from
            ContainerConfig.cleaning_config["extra_null_patterns"].
            Merged with NullPatternRule.KNOWN_NULLS — no code change needed.

        extra_garbage_re_patterns:
            Container-level regex alternatives from
            ContainerConfig.cleaning_config["extra_garbage_patterns"].
            Each string is added as a new alternative in GarbageKeywordRule.

    Returns:
        A CleaningProfile ready to use in data_preprocessor.py.
    """
    null_rule = NullPatternRule(extra_patterns=extra_null_patterns)

    if extra_garbage_re_patterns:
        garbage_rule     = GarbageKeywordRule(extra_patterns=extra_garbage_re_patterns)
        row_classifiers  = [AllEmptyRowRule(), garbage_rule, SeparatorRowRule()]
        ingest_logger.debug(
            "cleaning_profile_custom",
            extra_nulls=len(list(extra_null_patterns)),
            extra_garbage=len(list(extra_garbage_re_patterns)),
        )
    else:
        row_classifiers = _DEFAULT_ROW_CLASSIFIERS

    return CleaningProfile(null_rule=null_rule, row_classifiers=row_classifiers)
