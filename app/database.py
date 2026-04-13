"""
database.py — SQLAlchemy models for KPI logging + 100-call seed data
Stores call session metadata, per-turn data, and escalation events.
Used by both the FastAPI agent and Streamlit dashboard.
"""

import json
import os
import random
import time
from datetime import datetime, timedelta

from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, create_engine, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Database setup ────────────────────────────────────────────────────────────
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "voice_agent.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class CallSession(Base):
    """Represents a single customer support call."""
    __tablename__ = "call_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True)
    phone_number = Column(String(20), nullable=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Detected language
    detected_language = Column(String(10), default="en")

    # Outcome
    primary_intent = Column(String(50), nullable=True)
    resolution_status = Column(String(20), default="unresolved")  # resolved, unresolved, escalated
    csat_score = Column(Integer, nullable=True)   # 1–5 (simulated post-call IVR score)
    escalated = Column(Boolean, default=False)

    # Final sentiment
    avg_sentiment = Column(Float, nullable=True)
    final_sentiment_label = Column(String(20), nullable=True)

    # Relations
    turns = relationship("CallTurn", back_populates="session", cascade="all, delete-orphan")
    escalation = relationship("EscalationEvent", back_populates="session", uselist=False)


class CallTurn(Base):
    """Represents a single customer utterance + agent response turn."""
    __tablename__ = "call_turns"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("call_sessions.session_id"))
    turn_number = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # STT output
    customer_text = Column(Text)
    detected_language = Column(String(10), default="en")

    # Classification
    intent = Column(String(50), nullable=True)
    intent_confidence = Column(Float, nullable=True)

    # Sentiment
    sentiment_compound = Column(Float, nullable=True)
    sentiment_label = Column(String(20), nullable=True)

    # Agent response
    agent_response = Column(Text, nullable=True)
    response_source = Column(String(20), nullable=True)  # "rag", "canned", "escalation"

    session = relationship("CallSession", back_populates="turns")


class EscalationEvent(Base):
    """Represents a human escalation triggered during a call."""
    __tablename__ = "escalation_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("call_sessions.session_id"), unique=True)
    trigger_time = Column(DateTime, default=datetime.utcnow)
    trigger_turn = Column(Integer)
    reason = Column(String(50))          # hard_threshold, trend, manual
    compound_score = Column(Float)
    payload_json = Column(Text)          # Full EscalationPayload as JSON string

    session = relationship("CallSession", back_populates="escalation")


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Tables created at {DB_PATH}")


def get_db():
    """FastAPI dependency: yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Seed 100 realistic mock calls ─────────────────────────────────────────────

INTENTS = ["order_status", "refund_request", "product_query", "account_issue",
           "payment_failure", "general_complaint"]
LANGUAGES = ["en", "hi", "hi", "en", "bn", "en"]  # English-Hindi weighted
RESOLUTIONS = ["resolved", "resolved", "resolved", "unresolved", "escalated"]

SAMPLE_CUSTOMER_TEXTS = {
    "order_status": ["Where is my order?", "Mera order kab aayega?", "Track my shipment please"],
    "refund_request": ["I want a refund", "Paise wapas chahiye", "Refund status check karo"],
    "product_query": ["What is SIP minimum?", "SIP kaise kaam karta hai?", "How to withdraw from mutual fund?"],
    "account_issue": ["Login nahi ho raha", "I forgot my MPIN", "Account locked hai"],
    "payment_failure": ["Payment fail ho gaya", "Money deducted but not invested", "UPI failed"],
    "general_complaint": ["This is unacceptable!", "Bahut bura service hai", "I am very frustrated"],
}

SAMPLE_AGENT_RESPONSES = {
    "order_status": "I can see your order is currently in transit. Expected delivery is within 2 business days.",
    "refund_request": "I've initiated your refund. It will be credited to your account within 5-7 business days.",
    "product_query": "The minimum SIP amount on Groww is ₹100 per month. You can start with any fund of your choice.",
    "account_issue": "I've sent a password reset link to your registered email. Please check your inbox.",
    "payment_failure": "I can see the transaction. Your money is safe and will be refunded within 24 hours.",
    "general_complaint": "I sincerely apologize for the inconvenience. Let me escalate this to our senior team.",
}

def seed_mock_data(n_calls: int = 100):
    """Populate DB with n_calls realistic mock call sessions."""
    init_db()
    db = SessionLocal()

    try:
        # Don't double-seed
        if db.query(CallSession).count() >= n_calls:
            print(f"[DB] Already seeded with {db.query(CallSession).count()} calls. Skipping.")
            return

        print(f"[DB] Seeding {n_calls} mock calls...")
        base_time = datetime.utcnow() - timedelta(days=30)

        for i in range(n_calls):
            import uuid
            session_id = str(uuid.uuid4())[:12]
            intent = random.choice(INTENTS)
            lang = random.choice(LANGUAGES)
            resolution = random.choice(RESOLUTIONS)
            escalated = resolution == "escalated"

            n_turns = random.randint(2, 8)
            start = base_time + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(8, 22),
                minutes=random.randint(0, 59),
            )
            duration = n_turns * random.randint(20, 45)

            # Generate sentiment trajectory
            base_sentiment = random.uniform(-0.3, 0.6)
            if escalated:
                base_sentiment = random.uniform(-0.8, -0.4)

            sentiments = [
                max(-1.0, min(1.0, base_sentiment + random.gauss(0, 0.15)))
                for _ in range(n_turns)
            ]
            avg_sent = sum(sentiments) / len(sentiments)
            final_label = "negative" if avg_sent < -0.05 else ("positive" if avg_sent > 0.05 else "neutral")

            session = CallSession(
                session_id=session_id,
                phone_number=f"+91{random.randint(7000000000, 9999999999)}",
                start_time=start,
                end_time=start + timedelta(seconds=duration),
                duration_seconds=duration,
                detected_language=lang,
                primary_intent=intent,
                resolution_status=resolution,
                csat_score=random.randint(1, 5) if resolution == "resolved" else random.randint(1, 3),
                escalated=escalated,
                avg_sentiment=round(avg_sent, 4),
                final_sentiment_label=final_label,
            )
            db.add(session)
            db.flush()

            # Add turns
            texts = SAMPLE_CUSTOMER_TEXTS.get(intent, ["Tell me more"])
            for j in range(n_turns):
                sentiment = sentiments[j]
                turn = CallTurn(
                    session_id=session_id,
                    turn_number=j + 1,
                    timestamp=start + timedelta(seconds=j * (duration // n_turns)),
                    customer_text=random.choice(texts),
                    detected_language=lang,
                    intent=intent if j == 0 else None,
                    intent_confidence=round(random.uniform(0.55, 0.97), 3) if j == 0 else None,
                    sentiment_compound=round(sentiment, 4),
                    sentiment_label="negative" if sentiment < -0.05 else ("positive" if sentiment > 0.05 else "neutral"),
                    agent_response=SAMPLE_AGENT_RESPONSES.get(intent, "How can I help?"),
                    response_source="rag" if intent == "product_query" else "canned",
                )
                db.add(turn)

            # Add escalation event if needed
            if escalated:
                esc = EscalationEvent(
                    session_id=session_id,
                    trigger_time=start + timedelta(seconds=duration // 2),
                    trigger_turn=n_turns // 2,
                    reason=random.choice(["hard_threshold", "trend"]),
                    compound_score=round(base_sentiment, 4),
                    payload_json=json.dumps({"session_id": session_id, "reason": "escalated"}),
                )
                db.add(esc)

        db.commit()
        print(f"[DB] Seeded {n_calls} calls successfully.")

    except Exception as e:
        db.rollback()
        print(f"[DB] Seed error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_mock_data(100)
    db = SessionLocal()
    print(f"\n=== DB Stats ===")
    print(f"Call Sessions : {db.query(CallSession).count()}")
    print(f"Call Turns    : {db.query(CallTurn).count()}")
    print(f"Escalations   : {db.query(EscalationEvent).count()}")
    db.close()
