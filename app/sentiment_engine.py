"""
sentiment_engine.py — Real-Time VADER Sentiment Analysis + Escalation Logic
Per-session stateful tracker. Zero-dependency NLP, instant inference.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass, field
from typing import List, Optional
import time

# ── Escalation thresholds ────────────────────────────────────────────────────
HARD_THRESHOLD = -0.6        # Single turn compound score triggers immediate escalation
SOFT_THRESHOLD = -0.3        # Used for trend detection
TREND_WINDOW = 3             # Consecutive negative turns to trigger trend escalation
TREND_NEGATIVE_LIMIT = -0.2  # Each turn must be below this for trend escalation


@dataclass
class TurnSentiment:
    """Sentiment result for a single conversation turn."""
    turn_id: int
    text: str
    compound: float    # [-1, 1] — overall sentiment
    positive: float    # [0, 1] — positive word ratio
    negative: float    # [0, 1] — negative word ratio
    neutral: float     # [0, 1] — neutral word ratio
    label: str         # "positive", "neutral", "negative"
    timestamp: float = field(default_factory=time.time)

    @property
    def is_negative(self) -> bool:
        return self.compound < TREND_NEGATIVE_LIMIT


@dataclass
class EscalationPayload:
    """Structured payload sent to human supervisor on escalation."""
    session_id: str
    reason: str                      # "hard_threshold" or "trend"
    trigger_turn_id: int
    compound_score: float
    turn_history: List[TurnSentiment]
    summary: str                     # Human-readable summary of call so far
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "escalation_reason": self.reason,
            "trigger_turn": self.trigger_turn_id,
            "compound_score": self.compound_score,
            "sentiment_history": [
                {
                    "turn": t.turn_id,
                    "text": t.text[:120],
                    "compound": t.compound,
                    "label": t.label,
                }
                for t in self.turn_history
            ],
            "call_summary": self.summary,
            "timestamp": self.timestamp,
        }


class SentimentTracker:
    """
    Stateful per-session sentiment tracker using VADER.

    Usage:
        tracker = SentimentTracker(session_id="abc123")
        result = tracker.analyze_turn("I have been waiting for 3 days!")
        if tracker.should_escalate():
            payload = tracker.generate_escalation_payload("...")
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.analyzer = SentimentIntensityAnalyzer()
        self.history: List[TurnSentiment] = []
        self._escalated = False

    def analyze_turn(self, text: str) -> TurnSentiment:
        """
        Analyze a transcribed customer utterance and record sentiment.

        Args:
            text: English transcript of the customer's utterance.

        Returns:
            TurnSentiment with compound score and label.
        """
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        turn = TurnSentiment(
            turn_id=len(self.history) + 1,
            text=text,
            compound=round(compound, 4),
            positive=round(scores["pos"], 4),
            negative=round(scores["neg"], 4),
            neutral=round(scores["neu"], 4),
            label=label,
        )
        self.history.append(turn)
        return turn

    def should_escalate(self) -> tuple[bool, str]:
        """
        Check if escalation should be triggered.

        Returns:
            Tuple of (should_escalate: bool, reason: str)
        """
        if self._escalated:
            return False, "already_escalated"

        if not self.history:
            return False, ""

        latest = self.history[-1]

        # Rule 1: Hard threshold — single extremely negative turn
        if latest.compound < HARD_THRESHOLD:
            return True, "hard_threshold"

        # Rule 2: Trend — last N consecutive turns all negative
        if len(self.history) >= TREND_WINDOW:
            recent = self.history[-TREND_WINDOW:]
            if all(t.is_negative for t in recent):
                return True, "trend"

        return False, ""

    def generate_escalation_payload(self, call_summary: str = "") -> EscalationPayload:
        """
        Generate a structured escalation JSON payload for supervisor handoff.

        Args:
            call_summary: Brief summary of the conversation so far (from LLM).

        Returns:
            EscalationPayload with full context.
        """
        self._escalated = True
        _, reason = self.should_escalate()
        latest = self.history[-1] if self.history else None

        payload = EscalationPayload(
            session_id=self.session_id,
            reason=reason if reason else "manual",
            trigger_turn_id=latest.turn_id if latest else 0,
            compound_score=latest.compound if latest else 0.0,
            turn_history=self.history,
            summary=call_summary or "Customer expressed significant frustration. Human review needed.",
        )
        return payload

    def average_sentiment(self) -> float:
        """Returns the rolling average compound sentiment score."""
        if not self.history:
            return 0.0
        return round(sum(t.compound for t in self.history) / len(self.history), 4)

    def get_trend(self) -> list[float]:
        """Returns list of compound scores over time for chart rendering."""
        return [t.compound for t in self.history]


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = SentimentTracker(session_id="demo_session_001")

    conversation = [
        "Hi, I want to check on my order.",
        "It's been 5 days and nothing has arrived.",
        "This is really frustrating, I needed it urgently.",
        "I've called three times already and no one helps me.",
        "This is absolutely unacceptable! I demand a refund immediately!",
    ]

    print("Turn | Compound | Label    | Escalate? | Text")
    print("-" * 80)
    for utterance in conversation:
        result = tracker.analyze_turn(utterance)
        should_esc, reason = tracker.should_escalate()
        print(
            f"  {result.turn_id}  | {result.compound:+.3f}  | {result.label:<8} | "
            f"{'YES ← ' + reason if should_esc else 'No':<20} | {utterance[:40]}"
        )

    print(f"\nAverage sentiment: {tracker.average_sentiment():.3f}")

    if tracker.should_escalate()[0]:
        payload = tracker.generate_escalation_payload("Customer upset about delayed order.")
        import json
        print("\n=== Escalation Payload ===")
        print(json.dumps(payload.to_dict(), indent=2))
