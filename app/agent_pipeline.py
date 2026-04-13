"""
agent_pipeline.py — Central Orchestrator
Routes each customer turn through: Intent → RAG/Canned → Sentiment → DB log
"""

import uuid
from datetime import datetime
from typing import Optional

from app.intent_classifier import get_classifier
from app.rag_engine import get_rag_engine
from app.sentiment_engine import SentimentTracker
from app.database import SessionLocal, CallSession, CallTurn, EscalationEvent

# ── Canned responses for non-RAG intents ─────────────────────────────────────
CANNED_RESPONSES = {
    "order_status": (
        "I can look up your order right away. Could you please share your order ID "
        "or the registered mobile number?"
    ),
    "refund_request": (
        "I understand you'd like a refund. Refunds are typically processed within "
        "5–7 business days. I'm raising a ticket for your request now."
    ),
    "account_issue": (
        "I'm sorry you're having trouble accessing your account. "
        "I've sent a password reset link to your registered email and mobile number."
    ),
    "payment_failure": (
        "I can see a pending transaction on your account. "
        "Your money is completely safe. It will be refunded to your bank within 24–48 hours."
    ),
    "general_complaint": (
        "I sincerely apologize for the inconvenience caused. "
        "I'm escalating this to our senior support team who will contact you within 2 hours."
    ),
}

GREETING = (
    "Hello! Welcome to Groww customer support. I'm your virtual assistant. "
    "How can I help you today?"
)


class AgentSession:
    """
    Manages a single customer call session end-to-end.
    One instance per WebSocket connection.
    """

    def __init__(self, session_id: Optional[str] = None, detected_language: str = "en"):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.detected_language = detected_language
        self.turn_number = 0
        self.primary_intent: Optional[str] = None

        self.classifier = get_classifier()
        self.rag = get_rag_engine()
        self.sentiment = SentimentTracker(self.session_id)

        # Create DB record
        self._init_db_session()

    def _init_db_session(self):
        db = SessionLocal()
        try:
            record = CallSession(
                session_id=self.session_id,
                start_time=datetime.utcnow(),
                detected_language=self.detected_language,
            )
            db.add(record)
            db.commit()
        finally:
            db.close()

    def process_turn(self, customer_text: str, language: str = "en") -> dict:
        """
        Process a single customer utterance through the full pipeline.

        Args:
            customer_text: English transcript from STT (always in English).
            language: Detected ISO language code for TTS targeting.

        Returns:
            {
                "response_text": str,       # English agent response
                "intent": str,
                "intent_confidence": float,
                "sentiment": dict,
                "escalate": bool,
                "escalation_payload": dict | None,
                "session_id": str,
                "turn": int,
            }
        """
        self.turn_number += 1
        self.detected_language = language

        # 1. Intent Classification
        intent, confidence = self.classifier.classify(customer_text)
        if self.primary_intent is None:
            self.primary_intent = intent

        # 2. Generate response
        if intent == "product_query":
            response_text = self.rag.answer(customer_text, self.session_id)
            response_source = "rag"
        else:
            response_text = CANNED_RESPONSES.get(intent, CANNED_RESPONSES["general_complaint"])
            response_source = "canned"

        # 3. Sentiment analysis
        turn_sentiment = self.sentiment.analyze_turn(customer_text)
        should_escalate, escalation_reason = self.sentiment.should_escalate()

        escalation_payload = None
        if should_escalate:
            esc_payload = self.sentiment.generate_escalation_payload(
                call_summary=f"Customer called about: {self.primary_intent}. Turn {self.turn_number}."
            )
            escalation_payload = esc_payload.to_dict()
            response_text = (
                "I completely understand your frustration and I sincerely apologize. "
                "I'm connecting you right now with a senior Groww specialist who can resolve this immediately."
            )
            response_source = "escalation"
            self._save_escalation(esc_payload, escalation_payload)

        # 4. Log turn to DB
        self._log_turn(
            customer_text=customer_text,
            intent=intent,
            confidence=confidence,
            sentiment=turn_sentiment,
            response=response_text,
            source=response_source,
        )

        return {
            "response_text": response_text,
            "intent": intent,
            "intent_confidence": round(confidence, 4),
            "sentiment": {
                "compound": turn_sentiment.compound,
                "label": turn_sentiment.label,
            },
            "escalate": should_escalate,
            "escalation_payload": escalation_payload,
            "session_id": self.session_id,
            "turn": self.turn_number,
        }

    def _log_turn(self, customer_text, intent, confidence, sentiment, response, source):
        db = SessionLocal()
        try:
            turn = CallTurn(
                session_id=self.session_id,
                turn_number=self.turn_number,
                timestamp=datetime.utcnow(),
                customer_text=customer_text,
                detected_language=self.detected_language,
                intent=intent,
                intent_confidence=confidence,
                sentiment_compound=sentiment.compound,
                sentiment_label=sentiment.label,
                agent_response=response,
                response_source=source,
            )
            db.add(turn)
            db.commit()
        finally:
            db.close()

    def _save_escalation(self, esc_payload, payload_dict):
        import json
        db = SessionLocal()
        try:
            esc = EscalationEvent(
                session_id=self.session_id,
                trigger_time=datetime.utcnow(),
                trigger_turn=self.turn_number,
                reason=esc_payload.reason,
                compound_score=esc_payload.compound_score,
                payload_json=json.dumps(payload_dict),
            )
            db.add(esc)
            db.commit()
        finally:
            db.close()

    def close_session(self, resolution: str = "resolved"):
        """Mark session as ended and compute final metrics."""
        db = SessionLocal()
        try:
            session_record = db.query(CallSession).filter(
                CallSession.session_id == self.session_id
            ).first()
            if session_record:
                now = datetime.utcnow()
                duration = int((now - session_record.start_time).total_seconds())
                session_record.end_time = now
                session_record.duration_seconds = duration
                session_record.primary_intent = self.primary_intent
                session_record.resolution_status = resolution
                session_record.avg_sentiment = self.sentiment.average_sentiment()
                final = self.sentiment.history[-1].label if self.sentiment.history else "neutral"
                session_record.final_sentiment_label = final
                session_record.escalated = self.sentiment._escalated
                db.commit()
            self.rag.clear_session(self.session_id)
        finally:
            db.close()

    def get_greeting(self) -> str:
        return GREETING
