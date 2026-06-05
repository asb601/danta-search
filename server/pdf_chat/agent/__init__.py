"""Public agent surface: state, the legacy runner, deps, the graph builder,
and the Phase-3 agentic runtime (tool registry C3 + ``run_pdf_query`` C4)."""
from __future__ import annotations

from pdf_chat.agent.state import PdfChatState
from pdf_chat.agent.graph import (
    AgentDeps,
    Deps,
    PdfQueryResult,
    build_agent_graph,
    build_default_deps,
    build_graph,
    run_pdf_chat,
    run_pdf_query,
)
from pdf_chat.agent.tools import (
    PHASE3_TOOL_NAMES,
    RESERVED_TOOL_NAMES,
    TOOL_REGISTRY,
    Tool,
    register_tool,
)

__all__ = [
    # State
    "PdfChatState",
    # Legacy state-machine surface (preserved)
    "Deps",
    "run_pdf_chat",
    "build_graph",
    "build_default_deps",
    # Phase-3 agentic runtime (C4)
    "AgentDeps",
    "PdfQueryResult",
    "run_pdf_query",
    "build_agent_graph",
    # Phase-3 tool registry (C3)
    "Tool",
    "TOOL_REGISTRY",
    "register_tool",
    "PHASE3_TOOL_NAMES",
    "RESERVED_TOOL_NAMES",
]
