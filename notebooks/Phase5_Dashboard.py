# ============================================================
# Phase 5: Streamlit Dashboard + Docker on Google Colab
# Groww AI Voice Agent
# ============================================================

# %% [markdown]
"""
# 📊 Phase 5: Operations Dashboard via Streamlit (on Colab)

Runs the Streamlit dashboard on Colab using `pyngrok` to generate a public URL.
The dashboard connects to `data/voice_agent.db` (SQLite) seeded with 100 mock calls.
"""

# %% Step 1 — Install
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "streamlit", "plotly", "sqlalchemy", "pyngrok"], check=True)
print("✅ Installed")


# %% Step 2 — Seed the Database
# Run standalone: python app/database.py
# Or inline:
import sys, os
# Add project root to path (adjust if needed)
sys.path.insert(0, "/content")   # Colab default mount

try:
    from app.database import init_db, seed_mock_data
    init_db()
    seed_mock_data(100)
    print("✅ Database seeded with 100 mock calls")
except ImportError:
    # Fallback: create minimal DB inline
    from sqlalchemy import create_engine, text
    engine = create_engine("sqlite:///voice_agent.db")
    print("⚠️  app.database not found. Upload full project to /content/ first.")
    print("   Or run: !git clone <your-repo-url> /content/groww && cd /content/groww")


# %% Step 3 — Write Streamlit App to disk (if dashboard/app.py not present)
import os
if not os.path.exists("dashboard/app.py"):
    print("dashboard/app.py not found. Please upload the full project.")
    print("Required structure: /content/groww/")
else:
    print("✅ dashboard/app.py found")


# %% Step 4 — Set ngrok Auth Token
# Get your FREE token at: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"   # <-- replace this

from pyngrok import ngrok
ngrok.set_auth_token(NGROK_TOKEN)
print("✅ ngrok authenticated")


# %% Step 5 — Launch Streamlit + Expose Public URL
import threading, time

def run_streamlit():
    os.system("streamlit run dashboard/app.py --server.port 8501 --server.headless true")

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()
time.sleep(5)   # Wait for Streamlit to start

tunnel = ngrok.connect(8501, "http")
print(f"\n🌐 Dashboard URL: {tunnel.public_url}")
print("   Open the URL above in your browser!")
print("   Dashboard has 4 pages: Overview · Call Volume · Sentiment Trends · Live Monitor")


# %% Step 6 — Docker Reference (for local/cloud deploy)
print("""
════════════════════════════════════════════════════════
  Docker Deployment (local machine with NVIDIA GPU):
════════════════════════════════════════════════════════

  # Build and start both services:
  docker-compose up --build

  # FastAPI agent:   http://localhost:8000/docs
  # Dashboard:       http://localhost:8501
  # Health check:    http://localhost:8000/health
  # Live metrics:    http://localhost:8000/metrics

  # Test via WebSocket (text mode):
  wscat -c ws://localhost:8000/ws/test/demo01
  > {"type": "text", "data": "What is the minimum SIP amount?"}
  > {"type": "end"}

════════════════════════════════════════════════════════
""")
