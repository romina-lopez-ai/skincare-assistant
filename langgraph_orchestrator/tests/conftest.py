"""conftest.py — Shared pytest fixtures for workflow scenario tests.

The graph fixture is scoped to the session so the StateGraph is compiled
only once across all test runs — compilation is the expensive step.
"""

from __future__ import annotations

import pytest

from langgraph_orchestrator.build_graph import build_graph


@pytest.fixture(scope="session")
def graph():
    """Compile and return the LangGraph workflow.

    Session-scoped: compiled once, shared across every test in the suite.
    All tests receive the same compiled graph object (read-only — state is
    passed in as a fresh dict per test, so there is no shared mutable state).
    """
    return build_graph()
