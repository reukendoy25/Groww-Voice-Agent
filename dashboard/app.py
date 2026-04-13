"""
Streamlit Operations Dashboard — Groww Multilingual AI Voice Agent
Displays real-time KPIs: FCR, CSAT, call volume by intent, sentiment trends.
Run: streamlit run dashboard/app.py
"""

import os
import sys

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import func

from app.database import (
    init_db, seed_mock_data, SessionLocal,
    CallSession, CallTurn, EscalationEvent
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Groww Voice Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark gradient background */
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    [data-testid="metric-container"] label { color: #a0aec0 !important; font-size: 0.8rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #fff !important; font-size: 2rem !important; font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: rgba(10,10,25,0.95) !important; }

    /* Headers */
    h1, h2, h3 { color: #e2e8f0 !important; }

    /* Groww brand accent */
    .groww-accent { color: #00b09b; }

    /* Status badges */
    .badge-resolved  { background:#00b09b22; color:#00b09b; padding:2px 8px; border-radius:8px; font-size:0.8rem; }
    .badge-escalated { background:#f6546a22; color:#f6546a; padding:2px 8px; border-radius:8px; font-size:0.8rem; }
    .badge-active    { background:#4299e122; color:#4299e1; padding:2px 8px; border-radius:8px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Init DB on first run ──────────────────────────────────────────────────────
@st.cache_resource
def setup_db():
    init_db()
    seed_mock_data(100)
    return True

setup_db()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Groww Voice Agent")
    st.markdown("**Operations Dashboard**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Call Volume", "😊 Sentiment Trends", "🔴 Live Monitor"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    # Date range filter
    st.markdown("**Date Range**")
    days_back = st.slider("Last N days", 1, 30, 30)
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Powered by · faster-whisper · Mistral-7B · FAISS · VADER")


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_sessions(cutoff_ts: str) -> pd.DataFrame:
    cutoff = datetime.fromisoformat(cutoff_ts)
    db = SessionLocal()
    try:
        rows = db.query(CallSession).filter(CallSession.start_time >= cutoff).all()
        return pd.DataFrame([{
            "session_id": r.session_id,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "duration_seconds": r.duration_seconds or 0,
            "detected_language": r.detected_language or "en",
            "primary_intent": r.primary_intent or "unknown",
            "resolution_status": r.resolution_status or "unresolved",
            "csat_score": r.csat_score,
            "escalated": r.escalated,
            "avg_sentiment": r.avg_sentiment or 0.0,
            "final_sentiment_label": r.final_sentiment_label or "neutral",
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=30)
def load_turns(cutoff_ts: str) -> pd.DataFrame:
    cutoff = datetime.fromisoformat(cutoff_ts)
    db = SessionLocal()
    try:
        rows = db.query(CallTurn).filter(CallTurn.timestamp >= cutoff).all()
        return pd.DataFrame([{
            "session_id": r.session_id,
            "turn_number": r.turn_number,
            "timestamp": r.timestamp,
            "intent": r.intent,
            "sentiment_compound": r.sentiment_compound or 0.0,
            "sentiment_label": r.sentiment_label or "neutral",
            "response_source": r.response_source or "canned",
        } for r in rows])
    finally:
        db.close()


cutoff_str = cutoff.isoformat()
df_sessions = load_sessions(cutoff_str)
df_turns = load_turns(cutoff_str)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🏠 Operations Overview")
    st.markdown(f"*Last {days_back} days · {len(df_sessions)} total calls*")

    if df_sessions.empty:
        st.info("No call data found. Run `python app/database.py` to seed data.")
        st.stop()

    # ── KPI Metrics Row ───────────────────────────────────────────────────────
    total = len(df_sessions)
    resolved = (df_sessions["resolution_status"] == "resolved").sum()
    escalated = df_sessions["escalated"].sum()
    avg_csat = df_sessions["csat_score"].dropna().mean()
    avg_duration = df_sessions["duration_seconds"].mean()
    fcr = resolved / total * 100 if total > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📞 Total Calls", f"{total:,}")
    c2.metric("✅ FCR", f"{fcr:.1f}%", delta=f"+{fcr-75:.1f}% vs target")
    c3.metric("⭐ Avg CSAT", f"{avg_csat:.2f}/5.0")
    c4.metric("🚨 Escalations", f"{int(escalated)}", delta=f"{escalated/total*100:.1f}%")
    c5.metric("⏱️ Avg Duration", f"{avg_duration:.0f}s")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Resolution Donut
    with col_left:
        st.markdown("### Resolution Breakdown")
        res_counts = df_sessions["resolution_status"].value_counts().reset_index()
        res_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(
            res_counts, values="Count", names="Status", hole=0.55,
            color="Status",
            color_discrete_map={"resolved": "#00b09b", "escalated": "#f6546a", "unresolved": "#718096"},
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend=dict(font_color="#a0aec0"),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Language Distribution
    with col_right:
        st.markdown("### Language Distribution")
        lang_map = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu"}
        df_sessions["language_name"] = df_sessions["detected_language"].map(lambda x: lang_map.get(x, x.upper()))
        lang_counts = df_sessions["language_name"].value_counts().reset_index()
        lang_counts.columns = ["Language", "Count"]
        fig_lang = px.bar(
            lang_counts, x="Language", y="Count",
            color="Count", color_continuous_scale=["#00b09b", "#96c93d"],
        )
        fig_lang.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    # CSAT by intent
    st.markdown("### CSAT Score by Intent")
    csat_df = df_sessions.dropna(subset=["csat_score"])
    if not csat_df.empty:
        csat_intent = csat_df.groupby("primary_intent")["csat_score"].mean().reset_index()
        csat_intent.columns = ["Intent", "Avg CSAT"]
        fig_csat = px.bar(
            csat_intent.sort_values("Avg CSAT", ascending=True),
            x="Avg CSAT", y="Intent", orientation="h",
            color="Avg CSAT", color_continuous_scale=["#f6546a", "#f6d365", "#00b09b"],
            range_color=[1, 5],
        )
        fig_csat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_csat, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: CALL VOLUME
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Call Volume":
    st.markdown("# 📊 Call Volume Analysis")

    if df_sessions.empty:
        st.info("No data available.")
        st.stop()

    col1, col2 = st.columns(2)

    # Intent Distribution
    with col1:
        st.markdown("### Calls by Intent")
        intent_counts = df_sessions["primary_intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        intent_counts["Intent"] = intent_counts["Intent"].str.replace("_", " ").str.title()
        fig_intent = px.bar(
            intent_counts, x="Count", y="Intent", orientation="h",
            color="Count", color_continuous_scale=["#4299e1", "#00b09b"],
        )
        fig_intent.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
        )
        st.plotly_chart(fig_intent, use_container_width=True)

    # Hourly heatmap
    with col2:
        st.markdown("### Hourly Call Volume")
        df_sessions["hour"] = pd.to_datetime(df_sessions["start_time"]).dt.hour
        df_sessions["day"] = pd.to_datetime(df_sessions["start_time"]).dt.day_name()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heatmap_df = df_sessions.groupby(["day","hour"]).size().reset_index(name="calls")
        heatmap_pivot = heatmap_df.pivot(index="day", columns="hour", values="calls").fillna(0)
        heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
        fig_heat = px.imshow(
            heatmap_pivot,
            color_continuous_scale=["#1a1a2e", "#00b09b"],
            labels=dict(x="Hour of Day", y="Day", color="Calls"),
            aspect="auto",
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Daily call trend
    st.markdown("### Daily Call Volume Trend")
    df_sessions["date"] = pd.to_datetime(df_sessions["start_time"]).dt.date
    daily = df_sessions.groupby("date").size().reset_index(name="calls")
    fig_trend = px.area(
        daily, x="date", y="calls",
        color_discrete_sequence=["#00b09b"],
    )
    fig_trend.update_traces(fill="tozeroy", fillcolor="rgba(0,176,155,0.15)")
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Response source breakdown
    st.markdown("### Response Source (RAG vs Canned)")
    if not df_turns.empty:
        src_counts = df_turns["response_source"].value_counts().reset_index()
        src_counts.columns = ["Source", "Count"]
        fig_src = px.pie(
            src_counts, values="Count", names="Source", hole=0.4,
            color_discrete_sequence=["#00b09b", "#4299e1", "#f6546a"],
        )
        fig_src.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig_src, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: SENTIMENT TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "😊 Sentiment Trends":
    st.markdown("# 😊 Real-Time Sentiment Trends")

    if df_sessions.empty:
        st.info("No data available.")
        st.stop()

    # Overall sentiment distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sentiment Distribution")
        sent_counts = df_sessions["final_sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig_sent = px.pie(
            sent_counts, values="Count", names="Sentiment", hole=0.5,
            color="Sentiment",
            color_discrete_map={"positive": "#00b09b", "neutral": "#718096", "negative": "#f6546a"},
        )
        fig_sent.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        st.markdown("### Avg Sentiment by Intent")
        sent_intent = df_sessions.groupby("primary_intent")["avg_sentiment"].mean().reset_index()
        sent_intent.columns = ["Intent", "Avg Sentiment"]
        sent_intent["Intent"] = sent_intent["Intent"].str.replace("_", " ").str.title()
        fig_si = px.bar(
            sent_intent.sort_values("Avg Sentiment"),
            x="Intent", y="Avg Sentiment",
            color="Avg Sentiment",
            color_continuous_scale=["#f6546a", "#718096", "#00b09b"],
            range_color=[-1, 1],
        )
        fig_si.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
        )
        st.plotly_chart(fig_si, use_container_width=True)

    # Sentiment over time with escalation markers
    st.markdown("### Sentiment Timeline with Escalation Events")
    df_sessions["date"] = pd.to_datetime(df_sessions["start_time"]).dt.date
    daily_sent = df_sessions.groupby("date")["avg_sentiment"].mean().reset_index()

    escalated_dates = (
        df_sessions[df_sessions["escalated"] == True]
        .groupby("date").size().reset_index(name="escalations")
    )

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=daily_sent["date"], y=daily_sent["avg_sentiment"],
        mode="lines+markers", name="Avg Sentiment",
        line=dict(color="#00b09b", width=2),
        marker=dict(size=6),
    ))
    # Zero line
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig_timeline.add_hline(y=-0.6, line_dash="dot", line_color="#f6546a",
                           annotation_text="Hard Escalation Threshold",
                           annotation_font_color="#f6546a")
    # Escalation markers
    if not escalated_dates.empty:
        esc_sent = daily_sent[daily_sent["date"].isin(escalated_dates["date"])]
        fig_timeline.add_trace(go.Scatter(
            x=esc_sent["date"], y=esc_sent["avg_sentiment"],
            mode="markers", name="Escalation Event",
            marker=dict(color="#f6546a", size=12, symbol="x"),
        ))
    fig_timeline.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[-1, 1]),
        legend=dict(font_color="#a0aec0"),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Escalation reasons
    db = SessionLocal()
    try:
        esc_rows = db.query(EscalationEvent).all()
        if esc_rows:
            st.markdown("### Escalation Reasons Breakdown")
            esc_df = pd.DataFrame([{"reason": r.reason, "score": r.compound_score} for r in esc_rows])
            reason_counts = esc_df["reason"].value_counts().reset_index()
            reason_counts.columns = ["Reason", "Count"]
            fig_reasons = px.bar(
                reason_counts, x="Reason", y="Count",
                color_discrete_sequence=["#f6546a"],
            )
            fig_reasons.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_reasons, use_container_width=True)
    finally:
        db.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: LIVE MONITOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Live Monitor":
    st.markdown("# 🔴 Live Call Monitor")

    col_a, col_b, col_c = st.columns(3)

    try:
        import httpx
        resp = httpx.get("http://localhost:8000/metrics", timeout=2)
        metrics = resp.json()
        col_a.metric("Active Calls", metrics.get("active_calls", 0))
        col_b.metric("FCR", f"{metrics.get('first_contact_resolution_pct', 0)}%")
        col_c.metric("Avg CSAT", metrics.get("avg_csat", 0))
    except Exception:
        col_a.metric("Active Calls", "— (server offline)")
        col_b.metric("FCR", f"{(df_sessions['resolution_status']=='resolved').mean()*100:.1f}%" if not df_sessions.empty else "—")
        col_c.metric("Avg CSAT", f"{df_sessions['csat_score'].dropna().mean():.2f}" if not df_sessions.empty else "—")

    st.markdown("---")
    st.markdown("### Recent Calls")

    if not df_sessions.empty:
        recent = df_sessions.sort_values("start_time", ascending=False).head(20).copy()
        recent["start_time"] = pd.to_datetime(recent["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
        recent["duration"] = recent["duration_seconds"].apply(lambda x: f"{int(x//60)}m {int(x%60)}s")
        recent["Intent"] = recent["primary_intent"].str.replace("_", " ").str.title()
        recent["Sentiment"] = recent["avg_sentiment"].apply(
            lambda x: f"😊 {x:.2f}" if x > 0.05 else (f"😞 {x:.2f}" if x < -0.05 else f"😐 {x:.2f}")
        )
        recent["Status"] = recent["resolution_status"].apply(
            lambda x: "✅ Resolved" if x == "resolved" else ("🚨 Escalated" if x == "escalated" else "⏳ Pending")
        )

        display = recent[["session_id", "start_time", "Intent", "duration", "Sentiment", "Status", "csat_score"]].rename(columns={
            "session_id": "Session", "start_time": "Time", "duration": "Duration", "csat_score": "CSAT"
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Auto-refresh every 30s · Values sourced from SQLite DB · API metrics require server running on :8000")
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()
