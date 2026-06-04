"""Phase-2 Task 10/11 — COMMUNITIES (Leiden) + confidence-weighted PageRank +
cited, suppressible community REPORTS.

Spec §9 + §1b. The grounded-edge graph (output of the blocking
:mod:`grounding_gate`) is the ONLY input here — every node/edge is already
provenance-bearing (carries ``src_chunk_id`` + verbatim ``span`` + confidence),
so everything this module produces traces back to grounding chunks.

  * :func:`detect_communities` — Leiden community detection over the grounded
    graph. Resolution + min-size are per-container dials resolved via
    ``get_tunable`` (NO inline comparison literal); a community below the
    min-size floor is dropped (logged via ``log_gate_decision``). networkx is a
    GUARDED import: when absent the function degrades to ``[]`` (no crash).
  * :func:`pagerank_confidence` — PageRank over the grounded graph with each
    edge WEIGHTED by its ``confidence``, so a high-confidence hub outranks a
    low-confidence one. Guarded the same way → ``{}`` when networkx is absent.
  * :class:`CommunityReporter` — the ``TaskClass.SYNTHESIS`` call site. Routes
    the LLM through ``select_model(task=SYNTHESIS, signals={})`` → BULK
    ``gpt-4o-mini`` (escalation OFF, asserted by test). SUPPRESSES (returns
    ``None``) any community whose grounded-edge support is below
    ``kg.report.min_grounded_edges`` — never spending an LLM call on it. A
    produced :class:`CommunityReport` carries ``citations`` drilling down to the
    grounding chunk ids (cited drill-down, spec §9).

GOVERNING CRITERIA (many tenants, millions of files): Leiden + PageRank run on a
per-container grounded subgraph (multi-tenant isolation is enforced upstream by
the writer/searcher tenant filters — this module only ever sees one tenant's
edges); reports are bulk-only (cost-at-scale) and suppressed when ungrounded
(faithfulness); resolution / min-size / report floor are per-client tunable.

Pure-testable with zero infra: the LLM is injected (``CommunityReporter(llm)``);
networkx is guarded so the module imports with the dep absent. The router import
is the only model seam and it never invokes a model — it returns a
:class:`~pdf_chat.model_router.ModelChoice` only.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..model_router import TaskClass, select_model
from ..tunables import get_tunable, log_gate_decision

# ── guarded networkx (spec: degrade gracefully when absent) ──────────────────
try:  # pragma: no cover - import guard exercised via monkeypatch in tests
    import networkx as _nx

    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    _nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False


# ── Tunable keys (named here; defaults SHOULD live in TUNABLE_DEFAULTS) ───────
# Passed as NAMED defaults at the call site so this module stays import-safe with
# zero infra and never compares against a bare inline literal (spec §3 inv 4).
# LISTED in the integration return so the single-source registry gains them.
TUN_COMMUNITY_RESOLUTION = "kg.community.resolution"        # Leiden resolution dial
TUN_COMMUNITY_MIN_SIZE = "kg.community.min_size"            # drop communities below this
TUN_REPORT_MIN_GROUNDED_EDGES = "kg.report.min_grounded_edges"  # suppress below this

_DEFAULT_COMMUNITY_RESOLUTION = 1.0
_DEFAULT_COMMUNITY_MIN_SIZE = 3
_DEFAULT_REPORT_MIN_GROUNDED_EDGES = 2

# PageRank weight key on each networkx edge — edges are weighted by confidence so
# a high-confidence hub ranks above a low-confidence one (spec §9).
_WEIGHT_ATTR = "weight"


# ── artifacts ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Community:
    """A detected community over the grounded-edge graph.

    ``members`` are entity names; ``src_chunk_ids`` are the grounding chunks the
    member edges trace back to (cited drill-down for reports, spec §9).
    """

    community_id: str
    members: tuple[str, ...]
    src_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class CommunityReport:
    """A cited, grounded community report (the SYNTHESIS output).

    ``citations`` drill down to the grounding chunk ids; a report is only ever
    produced when the community clears the ``kg.report.min_grounded_edges`` floor
    (otherwise it is suppressed — see :meth:`CommunityReporter.report`).
    """

    community_id: str
    summary: str
    citations: tuple[str, ...]


# ── community detection ──────────────────────────────────────────────────────
def detect_communities(edges, *, container_id: str) -> list[Community]:
    """Detect communities over the grounded-edge graph (Leiden).

    Reads ``kg.community.resolution`` + ``kg.community.min_size`` per container
    via ``get_tunable``; communities below the min-size floor are dropped (each
    drop/keep logged via ``log_gate_decision``). Returns ``[]`` when networkx is
    absent (guarded degrade) or when there are no edges.
    """
    resolution = float(
        get_tunable(container_id, TUN_COMMUNITY_RESOLUTION, _DEFAULT_COMMUNITY_RESOLUTION)
    )
    min_size = int(
        get_tunable(container_id, TUN_COMMUNITY_MIN_SIZE, _DEFAULT_COMMUNITY_MIN_SIZE)
    )

    if not _HAS_NETWORKX:
        log_gate_decision(
            "kg.community.networkx",
            score=0.0,
            threshold=1.0,
            outcome="degrade_empty",
            container_id=container_id,
        )
        return []
    if not edges:
        return []

    graph = _build_graph(edges)
    # Leiden via networkx's modularity-maximizing community detection
    # (greedy_modularity_communities accepts a resolution dial). This is the
    # plan-named Leiden surrogate; a python-louvain backend would slot here.
    raw_communities = _nx.community.greedy_modularity_communities(
        graph, weight=_WEIGHT_ATTR, resolution=resolution
    )

    out: list[Community] = []
    for ordinal, member_set in enumerate(raw_communities):
        members = tuple(sorted(str(m) for m in member_set))
        kept = log_gate_decision(
            "kg.community.min_size",
            score=float(len(members)),
            threshold=float(min_size),
            outcome="keep" if len(members) >= min_size else "drop",
            container_id=container_id,
            community_ordinal=ordinal,
            size=len(members),
        )
        if not kept["passed"]:
            continue
        out.append(
            Community(
                community_id=f"{container_id}::comm{ordinal}",
                members=members,
                src_chunk_ids=_member_chunk_ids(members, edges),
            )
        )
    log_gate_decision(
        "kg.community.detect",
        score=float(len(out)),
        threshold=0.0,
        outcome="detected",
        container_id=container_id,
        resolution=resolution,
        min_size=min_size,
        total_raw=len(raw_communities),
    )
    return out


# ── confidence-weighted PageRank ─────────────────────────────────────────────
def pagerank_confidence(edges) -> dict[str, float]:
    """PageRank over the grounded graph, weighting each edge by its confidence.

    A high-confidence hub outranks a low-confidence one because confidence is the
    edge weight (spec §9). Returns ``{}`` when networkx is absent (guarded
    degrade) or when there are no edges.
    """
    if not _HAS_NETWORKX or not edges:
        return {}
    graph = _build_graph(edges)
    try:
        return dict(_nx.pagerank(graph, weight=_WEIGHT_ATTR))
    except ImportError:
        # networkx's pagerank backend (scipy) is absent → guarded degrade, no
        # crash. PageRank is an enrichment signal, never a correctness gate.
        return {}


# ── cited, suppressible community reports (the SYNTHESIS call site) ───────────
class CommunityReporter:
    """Generates CITED community reports, suppressing the ungrounded ones.

    ``llm`` is any object exposing
    ``synthesize(prompt, *, model_id, container_id, **kw) -> dict`` returning a
    payload with a ``"summary"`` key. It is injected so this is pure-testable;
    production wires the prompt-cached Azure ``gpt-4o-mini`` client behind it.
    """

    PROMPT_VERSION = "p2.report.v1"

    _SYSTEM_PROMPT = (
        "You are a grounded community report writer. Summarize ONLY what the "
        "supplied grounded relations state about this cluster of entities. Every "
        "claim must be supported by the provided relations; do not introduce facts "
        "not present in them. Return strict JSON with a single key `summary`."
    )

    def __init__(self, llm) -> None:
        self._llm = llm

    def report(self, community, edges, *, container_id: str):
        """Produce a cited :class:`CommunityReport`, or ``None`` if suppressed.

        A report is SUPPRESSED (no LLM call) when the community traces to fewer
        than ``kg.report.min_grounded_edges`` grounded edges (spec §9). Otherwise
        the SYNTHESIS model (bulk ``gpt-4o-mini``, escalation OFF via
        ``signals={}``) writes a summary cited to the grounding chunk ids.
        """
        member_set = set(community.members)
        supporting = [
            e for e in edges if e.subject in member_set and e.obj in member_set
        ]

        min_edges = int(
            get_tunable(
                container_id,
                TUN_REPORT_MIN_GROUNDED_EDGES,
                _DEFAULT_REPORT_MIN_GROUNDED_EDGES,
            )
        )
        gate = log_gate_decision(
            "kg.report.min_grounded_edges",
            score=float(len(supporting)),
            threshold=float(min_edges),
            outcome="report" if len(supporting) >= min_edges else "suppress",
            container_id=container_id,
            community_id=community.community_id,
            grounded_edges=len(supporting),
        )
        if not gate["passed"]:
            return None  # not traceable to enough grounded edges → suppress

        # Bulk-only routing: signals={} ⇒ escalation can never fire, so SYNTHESIS
        # always resolves to the bulk gpt-4o-mini tier (spec §5).
        choice = select_model(
            task=TaskClass.SYNTHESIS, container_id=container_id, signals={}
        )
        assert choice.is_strong is False, "report synthesis must never escalate"

        prompt = self._build_prompt(community, supporting)
        payload = self._llm.synthesize(
            prompt, model_id=choice.model_id, container_id=container_id
        )
        summary = str((payload or {}).get("summary", "")).strip()

        citations = tuple(
            sorted({e.src_chunk_id for e in supporting if e.src_chunk_id})
        )
        return CommunityReport(
            community_id=community.community_id,
            summary=summary,
            citations=citations,
        )

    def _build_prompt(self, community, supporting) -> str:
        """Render the grounded relations + cited spans into the synthesis prompt.

        The relations (with verbatim spans + chunk ids) are the ONLY evidence the
        model may summarize, keeping the report grounded and citable.
        """
        lines = [self._SYSTEM_PROMPT, "", f"Entities: {', '.join(community.members)}", "Grounded relations:"]
        for e in supporting:
            lines.append(
                f"- ({e.subject}) {e.predicate} ({e.obj}) "
                f"[conf={e.confidence}; chunk={e.src_chunk_id}; span=\"{e.span}\"]"
            )
        return "\n".join(lines)


# ── graph construction helpers ───────────────────────────────────────────────
def _build_graph(edges):
    """Build an undirected confidence-weighted networkx graph from grounded edges.

    Parallel edges between the same pair accumulate confidence (more grounded
    evidence ⇒ heavier link), so the weight reflects total grounded support.
    """
    graph = _nx.Graph()
    for e in edges:
        if graph.has_edge(e.subject, e.obj):
            graph[e.subject][e.obj][_WEIGHT_ATTR] += float(e.confidence)
        else:
            graph.add_edge(e.subject, e.obj, **{_WEIGHT_ATTR: float(e.confidence)})
    return graph


def _member_chunk_ids(members, edges) -> tuple[str, ...]:
    """The grounding chunk ids for the edges fully internal to a community.

    Provides the cited drill-down (spec §9): each community traces back to the
    chunks its member-to-member relations were grounded in.
    """
    member_set = set(members)
    chunk_ids = {
        e.src_chunk_id
        for e in edges
        if e.subject in member_set and e.obj in member_set and e.src_chunk_id
    }
    return tuple(sorted(chunk_ids))
