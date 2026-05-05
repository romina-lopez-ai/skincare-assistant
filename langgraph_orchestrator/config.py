"""config.py — Backend configuration for the dermatology chatbot.

This is the single file a teammate edits to switch from mock services to
real model implementations.  Nothing else in the codebase needs to change.

Usage
-----
Set each flag to True to use the real backend, or False to stay on the mock.
Flags are independent — you can run a real LLM with a mock vision model, for
example, while the vision integration is still in progress.

    USE_MOCK_LLM    = True   # use hardcoded LLM responses
    USE_MOCK_VISION = False  # use real DenseNet inference
    USE_MOCK_RAG    = True   # use hardcoded retrieval context

Environment-variable override
------------------------------
Each flag can also be overridden at launch time by setting the corresponding
environment variable to "0" (real) or "1" (mock):

    USE_MOCK_LLM=0 USE_MOCK_VISION=0 uvicorn langgraph_orchestrator.web.app:app

Environment variables take precedence over the values defined below, which
makes it easy to switch backends in CI or Docker without editing source.

Adding a new backend
---------------------
1. Add a new flag here (e.g. USE_MOCK_EMBEDDINGS = True).
2. Read it in the relevant service file with the same _env_flag() pattern.
3. Implement _real_<function>() and _mock_<function>() in that service.
4. Gate the public function on the flag.
"""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean flag from an environment variable, falling back to default.

    Treats the env-var value "0" as False and anything else as True.
    If the variable is not set, returns default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw != "0"


# ---------------------------------------------------------------------------
# Backend flags — edit these to switch between mock and real implementations.
# ---------------------------------------------------------------------------

# LLM service (symptom questions, diagnosis text, follow-up answers)
USE_MOCK_LLM: bool = _env_flag("USE_MOCK_LLM", default=False)

# Vision service (DenseNet inference + projection layer)
USE_MOCK_VISION: bool = _env_flag("USE_MOCK_VISION", default=False)

# RAG service (vector-store retrieval for follow-up questions)
# Still mock — no vector store is deployed yet.
USE_MOCK_RAG: bool = _env_flag("USE_MOCK_RAG", default=False)
