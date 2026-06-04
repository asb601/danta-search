"""pytest config for pdf_chat tests.

Registers the ``infra`` marker so infra-dependent tests (Redis/Neo4j/Azure) are
opt-in. Default runs exclude them with ``-m "not infra"``.
"""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "infra: requires live infra (Redis/Neo4j/Azure); excluded by default",
    )
