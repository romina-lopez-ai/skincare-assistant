"""app.py — FastAPI application for the dermatology chatbot test interface.

Run locally
-----------
From the skincare_model/ directory:

    pip install fastapi uvicorn python-multipart jinja2
    python -m uvicorn langgraph_orchestrator.web.app:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langgraph_orchestrator.web.api import schemas, service

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"
UPLOADS_DIR   = BASE_DIR / "uploads"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dermatology Chatbot — Local Test Interface",
    description="Test UI for the LangGraph multimodal dermatology chatbot workflow.",
    version="0.1.0",
)

app.mount("/static",  StaticFiles(directory=STATIC_DIR),  name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ---------------------------------------------------------------------------
# HTML entry point
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the single-page chat interface."""
    return templates.TemplateResponse(request=request, name="index.html")


# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------

@app.post("/session/start", response_model=schemas.StartSessionResponse)
async def start_session() -> schemas.StartSessionResponse:
    """Create a new session and return its ID.

    Called automatically when the page loads.
    """
    state = service.start_session()
    return schemas.StartSessionResponse(
        session_id=state["session_id"],
        message="Session started. Describe your skin concern to begin.",
        phase=state["phase"],
    )


# ---------------------------------------------------------------------------
# POST /chat/message
# ---------------------------------------------------------------------------

@app.post("/chat/message", response_model=schemas.ChatResponse)
async def send_message(req: schemas.ChatRequest) -> schemas.ChatResponse:
    """Process one user message through the LangGraph workflow.

    Appends the user message to conversation history, runs the graph,
    and returns the assistant response plus current state metadata.
    """
    if not service.get_session(req.session_id):
        service.start_session(req.session_id)

    result = service.run_turn(req.session_id, req.message)

    return schemas.ChatResponse(
        session_id=req.session_id,
        response=result["response"],
        phase=result["phase"],
        symptoms_complete=result["symptoms_complete"],
        image_uploaded=result["image_uploaded"],
        diagnosis_done=result["diagnosis_done"],
        rag_context_used=result["rag_context_used"],
        nodes_visited=result["nodes_visited"],
        diagnosis_summary=result["diagnosis_summary"],  # Summary: progressive summary card
    )


# ---------------------------------------------------------------------------
# POST /chat/upload-image
# ---------------------------------------------------------------------------

@app.post("/chat/upload-image", response_model=schemas.ImageUploadResponse)
async def upload_image(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> schemas.ImageUploadResponse:
    """Accept an image upload, save it to disk, and update session state.

    The vision pipeline (DenseNet → projection → diagnosis) is NOT triggered
    immediately.  It fires on the next /chat/message call once both
    symptoms_complete and image_uploaded are True.
    """
    if not service.get_session(session_id):
        service.start_session(session_id)

    # Save file with a session-scoped unique name
    suffix   = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{session_id[:8]}_{uuid.uuid4().hex[:8]}{suffix}"
    dest     = UPLOADS_DIR / filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result   = service.store_image(session_id, str(dest))
    image_url = f"/uploads/{filename}"

    return schemas.ImageUploadResponse(
        session_id=session_id,
        response=result["response"],
        image_url=image_url,
        phase=result["phase"],
        symptoms_complete=result["symptoms_complete"],
        image_uploaded=result["image_uploaded"],
        diagnosis_done=result["diagnosis_done"],
        nodes_visited=result["nodes_visited"],
        diagnosis_summary=result["diagnosis_summary"],  # Summary: progressive summary card
    )


# ---------------------------------------------------------------------------
# POST /session/reset
# ---------------------------------------------------------------------------

@app.post("/session/reset", response_model=schemas.ResetResponse)
async def reset_session(req: schemas.ResetRequest) -> schemas.ResetResponse:
    """Reset a session to its initial blank state."""
    state = service.reset_session(req.session_id)
    return schemas.ResetResponse(
        session_id=req.session_id,
        message="Session reset. Start a new conversation.",
        phase=state["phase"],
    )


# ---------------------------------------------------------------------------
# GET /session/{session_id}/state
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/state", response_model=schemas.SessionStateResponse)
async def get_state(session_id: str) -> schemas.SessionStateResponse:
    """Return the full state snapshot for the debug panel."""
    summary = service.get_state_summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return schemas.SessionStateResponse(**summary)
