"""
websocket_handler.py — Full-Duplex WebSocket Session Manager
Handles raw audio streaming → STT → Agent Pipeline → TTS → audio response
"""

import asyncio
import base64
import json
import traceback
from typing import Dict

from fastapi import WebSocket, WebSocketDisconnect

from app.agent_pipeline import AgentSession
from app.stt_engine import STTEngine
from app.tts_engine import TTSEngine

# Audio buffer config
AUDIO_CHUNK_SILENCE_TIMEOUT = 1.5   # seconds without audio before processing

# Shared model instances (loaded once at startup)
_stt: STTEngine = None
_tts: TTSEngine = None
_active_sessions: Dict[str, AgentSession] = {}


def initialize_models(stt: STTEngine, tts: TTSEngine):
    """Called from main.py startup to inject shared model instances."""
    global _stt, _tts
    _stt = stt
    _tts = tts


async def handle_voice_websocket(websocket: WebSocket, session_id: str):
    """
    Main WebSocket handler for a voice call session.

    Protocol (JSON messages):
      Client → Server:
        {"type": "audio", "data": "<base64-encoded WAV bytes>"}
        {"type": "text",  "data": "raw text (test mode, skips STT)"}
        {"type": "end"}

      Server → Client:
        {"type": "greeting", "text": "...", "audio": "<base64 WAV>"}
        {"type": "response", "text": "...", "audio": "<base64 WAV>",
         "intent": "...", "sentiment": {...}, "escalate": false}
        {"type": "escalation", "payload": {...}}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    print(f"[WS] Session {session_id} connected.")

    session = AgentSession(session_id=session_id)
    _active_sessions[session_id] = session

    try:
        # Send greeting
        greeting_text = session.get_greeting()
        greeting_audio = _tts.synthesize(greeting_text, target_language="en")
        await websocket.send_text(json.dumps({
            "type": "greeting",
            "text": greeting_text,
            "audio": base64.b64encode(greeting_audio).decode("utf-8"),
            "session_id": session_id,
        }))

        # Main message loop
        async for raw_message in websocket.iter_text():
            try:
                msg = json.loads(raw_message)
                msg_type = msg.get("type", "text")

                if msg_type == "end":
                    session.close_session(resolution="resolved")
                    await websocket.send_text(json.dumps({"type": "goodbye", "session_id": session_id}))
                    break

                # ── Get transcript ──────────────────────────────────────
                if msg_type == "audio":
                    # Real audio: base64-decode then run STT
                    audio_bytes = base64.b64decode(msg["data"])
                    stt_result = await asyncio.get_event_loop().run_in_executor(
                        None, _stt.transcribe, audio_bytes
                    )
                    customer_text = stt_result["text"]
                    language = stt_result["language"]
                else:
                    # Text override mode (testing / demo)
                    customer_text = msg.get("data", "").strip()
                    language = msg.get("language", "en")

                if not customer_text:
                    continue

                print(f"[WS] [{session_id}] ({language}) > {customer_text}")

                # ── Process through agent pipeline ──────────────────────
                result = session.process_turn(customer_text, language=language)

                # ── Synthesize TTS response ─────────────────────────────
                response_audio = await asyncio.get_event_loop().run_in_executor(
                    None, _tts.synthesize, result["response_text"], language
                )

                # ── Send response ───────────────────────────────────────
                response_msg = {
                    "type": "response",
                    "text": result["response_text"],
                    "audio": base64.b64encode(response_audio).decode("utf-8"),
                    "intent": result["intent"],
                    "intent_confidence": result["intent_confidence"],
                    "sentiment": result["sentiment"],
                    "escalate": result["escalate"],
                    "turn": result["turn"],
                    "session_id": session_id,
                }

                if result["escalate"]:
                    response_msg["type"] = "escalation"
                    response_msg["escalation_payload"] = result["escalation_payload"]
                    session.close_session(resolution="escalated")
                    await websocket.send_text(json.dumps(response_msg))
                    break
                else:
                    await websocket.send_text(json.dumps(response_msg))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON message format."
                }))
            except Exception as e:
                traceback.print_exc()
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                }))

    except WebSocketDisconnect:
        print(f"[WS] Session {session_id} disconnected.")
        session.close_session(resolution="unresolved")
    finally:
        _active_sessions.pop(session_id, None)


def get_active_sessions() -> Dict[str, dict]:
    """Returns summary of currently active sessions (for dashboard)."""
    return {
        sid: {
            "turn": s.turn_number,
            "intent": s.primary_intent,
            "avg_sentiment": s.sentiment.average_sentiment(),
        }
        for sid, s in _active_sessions.items()
    }
