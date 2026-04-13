"""
main.py — FastAPI Application Entry Point
REST endpoints + WebSocket gateway for the voice agent
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import init_db, seed_mock_data, SessionLocal, CallSession, EscalationEvent
from app.websocket_handler import handle_voice_websocket, get_active_sessions, initialize_models

# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize DB on startup."""
    print("[App] Initializing database...")
    init_db()
    seed_mock_data(100)  # Populate dashboard with mock data on first run

    print("[App] Loading STT model (faster-whisper large-v3)...")
    from app.stt_engine import STTEngine
    stt = STTEngine(model_size="large-v3", device="cuda", compute_type="float16")

    print("[App] Loading TTS model (Coqui XTTS-v2)...")
    from app.tts_engine import TTSEngine
    tts = TTSEngine(use_gpu=True)

    print("[App] Pre-loading intent classifier and RAG engine...")
    from app.intent_classifier import get_classifier
    from app.rag_engine import get_rag_engine
    get_classifier()
    get_rag_engine()

    # Inject shared models into WebSocket handler
    initialize_models(stt, tts)

    print("[App] All systems ready. Voice agent online. 🎙️")
    yield
    print("[App] Shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Groww Multilingual AI Voice Agent",
    description="D2C Customer Support Voice Agent — Hindi/English, FAISS RAG, VADER Sentiment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check for load balancers and Docker healthcheck."""
    return {"status": "ok", "service": "Groww Voice Agent", "version": "1.0.0"}


@app.get("/metrics", tags=["System"])
async def metrics():
    """Returns live KPI snapshot from the database."""
    db = SessionLocal()
    try:
        total = db.query(CallSession).count()
        resolved = db.query(CallSession).filter(CallSession.resolution_status == "resolved").count()
        escalated = db.query(CallSession).filter(CallSession.escalated == True).count()
        active = len(get_active_sessions())

        # Average CSAT
        csat_scores = [
            r.csat_score for r in db.query(CallSession).all()
            if r.csat_score is not None
        ]
        avg_csat = round(sum(csat_scores) / len(csat_scores), 2) if csat_scores else 0.0

        fcr = round(resolved / total * 100, 1) if total > 0 else 0.0

        return {
            "total_calls": total,
            "active_calls": active,
            "first_contact_resolution_pct": fcr,
            "escalation_rate_pct": round(escalated / total * 100, 1) if total > 0 else 0.0,
            "avg_csat": avg_csat,
        }
    finally:
        db.close()


@app.get("/sessions/active", tags=["Sessions"])
async def active_sessions():
    """Returns list of currently active WebSocket sessions."""
    return get_active_sessions()


@app.post("/sessions/new", tags=["Sessions"])
async def create_session():
    """Generate a new session ID for a WebSocket connection."""
    return {"session_id": str(uuid.uuid4())[:12]}


# ── WebSocket Gateway ─────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """
    Full-duplex WebSocket endpoint for voice calls.

    Connect: ws://localhost:8000/ws/{session_id}

    Message format:
      Send: {"type": "text", "data": "Your question here"}   # test mode
      Send: {"type": "audio", "data": "<base64 WAV>"}         # production mode
      Send: {"type": "end"}                                    # close session

    Receive: JSON with response text, audio, intent, sentiment, escalation flag
    """
    await handle_voice_websocket(websocket, session_id)


@app.websocket("/ws/test/{session_id}")
async def test_websocket(websocket: WebSocket, session_id: str):
    """
    Text-only WebSocket for testing without audio hardware.
    Same protocol as /ws but accepts only {"type": "text", "data": "..."} messages.
    STT is bypassed — text is used directly.
    """
    await handle_voice_websocket(websocket, session_id)
