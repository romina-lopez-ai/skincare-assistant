"""schemas.py — Pydantic request/response models for all API endpoints."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# /session/start
# ---------------------------------------------------------------------------

class StartSessionResponse(BaseModel):
    session_id: str
    message: str
    phase: str


# ---------------------------------------------------------------------------
# /chat/message
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    phase: str
    symptoms_complete: bool
    image_uploaded: bool
    diagnosis_done: bool
    rag_context_used: bool
    nodes_visited: list[str]


# ---------------------------------------------------------------------------
# /chat/upload-image
# ---------------------------------------------------------------------------

class ImageUploadResponse(BaseModel):
    session_id: str
    response: str
    image_url: str
    phase: str
    symptoms_complete: bool
    image_uploaded: bool
    diagnosis_done: bool
    nodes_visited: list[str]


# ---------------------------------------------------------------------------
# /session/reset
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    session_id: str
    message: str
    phase: str


# ---------------------------------------------------------------------------
# GET /session/{session_id}/state
# ---------------------------------------------------------------------------

class SymptomSlotsOut(BaseModel):
    location: Optional[str] = None
    duration: Optional[str] = None
    appearance: Optional[str] = None
    severity: Optional[str] = None
    triggers: Optional[str] = None


class SessionStateResponse(BaseModel):
    session_id: str
    phase: str
    symptom_slots: SymptomSlotsOut
    symptom_turns: int
    symptoms_complete: bool
    image_uploaded: bool
    image_path: Optional[str] = None
    image_processed: bool
    densenet_output: Optional[dict[str, Any]] = None
    diagnosis_done: bool
    diagnosis_summary: Optional[str] = None
    rag_context: Optional[str] = None
    fallback_triggered: bool
    uncertainty_flag: bool
    message_count: int
    messages: list[dict[str, Any]]
