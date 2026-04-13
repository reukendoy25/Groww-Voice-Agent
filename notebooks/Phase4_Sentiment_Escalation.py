# ============================================================
# Phase 4: Real-Time NLP Sentiment Analysis + Escalation Logic
# Groww AI Voice Agent · Google Colab (CPU sufficient)
# ============================================================

# %% [markdown]
"""
# 😤 Phase 4: VADER Sentiment Analysis + Escalation Logic

**Model**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
**Zero ML inference** — pure lexicon lookup, instant
**Escalation Rules**:
- Hard threshold: compound score < -0.6 → immediate escalation
- Trend: 3 consecutive negative turns → escalation
"""

# %% Step 1 — Install
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vaderSentiment", "plotly"], check=True)
print("✅ Installed")


# %% Step 2 — VADER Demo
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

sample_texts = [
    "Hello, I want to check my order status.",
    "It's been 3 days and nothing has arrived.",
    "This is very frustrating, I needed it urgently.",
    "I've called multiple times and nobody helps.",
    "This is absolutely unacceptable! I demand a refund!",
    "Mujhe abhi refund chahiye! Bahut bura service hai!",   # Hindi — VADER scores English loanwords
    "Thank you for your help, that's very clear!",
    "Still waiting, quite unhappy about the delay.",
]

print(f"\n{'Text':<55} {'Compound':>10} {'Label'}")
print("─" * 80)
for text in sample_texts:
    scores = analyzer.polarity_scores(text)
    c = scores["compound"]
    label = "🟢 Positive" if c >= 0.05 else ("🔴 Negative" if c <= -0.05 else "🟡 Neutral")
    print(f"{text[:53]:<55} {c:>+8.3f}  {label}")


# %% Step 3 — Stateful SentimentTracker Class
HARD_THRESHOLD = -0.6
TREND_WINDOW = 3
TREND_LIMIT   = -0.2

class SentimentTracker:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.analyzer = SentimentIntensityAnalyzer()
        self.history = []
        self._escalated = False

    def analyze_turn(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        turn = {"turn": len(self.history)+1, "text": text, "compound": round(compound,4), "label": label}
        self.history.append(turn)
        return turn

    def should_escalate(self):
        if self._escalated or not self.history:
            return False, ""
        latest = self.history[-1]
        if latest["compound"] < HARD_THRESHOLD:
            return True, f"hard_threshold ({latest['compound']:.3f} < {HARD_THRESHOLD})"
        if len(self.history) >= TREND_WINDOW:
            recent = self.history[-TREND_WINDOW:]
            if all(t["compound"] < TREND_LIMIT for t in recent):
                return True, f"trend (last {TREND_WINDOW} turns all negative)"
        return False, ""

    def generate_payload(self, summary: str) -> dict:
        self._escalated = True
        _, reason = self.should_escalate() if not self._escalated else (None, "triggered")
        return {
            "session_id": self.session_id,
            "reason": reason or "triggered",
            "turn_count": len(self.history),
            "final_compound": self.history[-1]["compound"],
            "avg_sentiment": round(sum(t["compound"] for t in self.history)/len(self.history), 4),
            "summary": summary,
            "history": self.history,
        }


# %% Step 4 — Simulate a Frustrated Customer Call
conversation = [
    ("Hi, I placed an order last week.", "neutral"),
    ("It's been 7 days and still not delivered.", "negative"),
    ("I called support twice and got no resolution.", "negative"),
    ("This delay is completely unacceptable for my business!", "very negative"),
    ("I am absolutely furious. I want a full refund NOW or I'm disputing with my bank!", "extreme"),
]

tracker = SentimentTracker("demo_001")
print("\n" + "="*70)
print("📞 SIMULATED CALL — Frustrated Customer Scenario")
print("="*70)

for text, _ in conversation:
    result = tracker.analyze_turn(text)
    should_esc, reason = tracker.should_escalate()

    print(f"\n  Turn {result['turn']}: Customer: \"{text}\"")
    print(f"           Sentiment : {result['compound']:+.3f}  ({result['label']})")
    if should_esc:
        print(f"  🚨 ESCALATION TRIGGERED: {reason}")
        payload = tracker.generate_payload(f"Customer upset about delayed order. {len(tracker.history)} turns.")
        import json
        print(f"  📦 Escalation Payload:\n{json.dumps(payload, indent=4)}")
        break
    else:
        print(f"           Escalate  : No")


# %% Step 5 — Visualize Sentiment Timeline
try:
    import plotly.graph_objects as go

    compounds = [t["compound"] for t in tracker.history]
    turns = list(range(1, len(compounds)+1))
    colors = ["#f6546a" if c < -0.05 else ("#00b09b" if c > 0.05 else "#718096") for c in compounds]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=turns, y=compounds, marker_color=colors, name="Sentiment"))
    fig.add_hline(y=HARD_THRESHOLD, line_dash="dot", line_color="red",
                  annotation_text=f"Hard Threshold ({HARD_THRESHOLD})")
    fig.add_hline(y=TREND_LIMIT, line_dash="dash", line_color="orange",
                  annotation_text=f"Trend Threshold ({TREND_LIMIT})")
    fig.update_layout(
        title="Customer Sentiment per Turn — Escalation Detection",
        xaxis_title="Turn", yaxis_title="VADER Compound Score",
        yaxis=dict(range=[-1.1, 1.1]),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="#e2e8f0"
    )
    fig.show()
except Exception as e:
    print(f"Plotly not available: {e}")

print("\n✅ Phase 4 complete! Phase 5 runs the Streamlit dashboard.")
