# ============================================================
# VoiceAgent_Complete.py — Master Integration: Full Pipeline on Colab
# Groww Multilingual AI Voice Agent
#
# HOW TO USE:
#   1. Upload entire project folder to Google Drive
#   2. Mount Drive in Colab: drive.mount('/content/drive')
#   3. Set PROJECT_ROOT below to your project path
#   4. Runtime → T4 GPU → Run All
# ============================================================

# %% [markdown]
"""
# 🚀 Groww Multilingual AI Voice Agent — Complete Demo!

**Single notebook to run the entire pipeline:**
1. Install all dependencies
2. Load all models (faster-whisper, XTTS-v2, MiniLM-L12, Mistral-7B)
3. Start FastAPI WebSocket server
4. Expose via ngrok public URL
5. Run interactive text-mode conversation demo
6. Launch Streamlit dashboard
"""

# %% CONFIGURATION — Edit these values!
PROJECT_ROOT = "/content/groww"   # Change to your actual path
NGROK_TOKEN  = "YOUR_NGROK_TOKEN_HERE"   # Free at ngrok.com


# %% Cell 1 — Mount Drive + Setup Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
    print("✅ Google Drive mounted")
except:
    print("⚠️  Not in Colab — running in local mode")

import sys, os
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print(f"Working dir: {os.getcwd()}")


# %% Cell 2 — Install All Dependencies (~8 mins on first run)
import subprocess

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)

packages = [
    "faster-whisper==1.0.1",
    "TTS==0.22.0",
    "sentence-transformers==3.0.1",
    "faiss-cpu==1.8.0",
    "langchain==0.2.5",
    "langchain-community==0.2.5",
    "transformers==4.41.2",
    "bitsandbytes==0.43.1",
    "accelerate==0.30.0",
    "vaderSentiment==3.3.2",
    "fastapi==0.111.0",
    "uvicorn[standard]==0.30.0",
    "websockets==12.0",
    "sqlalchemy==2.0.30",
    "streamlit==1.35.0",
    "plotly==5.22.0",
    "pandas==2.2.2",
    "pyngrok==7.1.6",
    "httpx==0.27.0",
    "scikit-learn",
    "python-dotenv",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
print("✅ All packages installed")


# %% Cell 3 — Initialize Database with Mock Data
from app.database import init_db, seed_mock_data
init_db()
seed_mock_data(100)
print("✅ Database initialized with 100 mock call sessions")


# %% Cell 4 — Load STT Model (faster-whisper)
import time
from faster_whisper import WhisperModel

print("Loading faster-whisper large-v3 (STT)...")
t0 = time.time()
stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"✅ STT ready in {time.time()-t0:.1f}s")


# %% Cell 5 — Load TTS Model (Coqui XTTS-v2)
from TTS.api import TTS as CoquiTTS

print("Loading Coqui XTTS-v2 (TTS)...")
t0 = time.time()
tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print(f"✅ TTS ready in {time.time()-t0:.1f}s")


# %% Cell 6 — Load Intent Classifier + RAG Engine
from app.intent_classifier import get_classifier
from app.rag_engine import get_rag_engine

print("Loading intent classifier...")
classifier = get_classifier()

print("Loading RAG engine (builds FAISS index)...")
rag = get_rag_engine()
print("✅ Intent + RAG ready")


# %% Cell 7 — Start FastAPI WebSocket Server in Background
import threading

# Inject models into WebSocket handler
from app.websocket_handler import initialize_models
from app.stt_engine import STTEngine
from app.tts_engine import TTSEngine

# Wrap loaded models in our engine classes
class ColabSTT:
    def __init__(self, model): self.model = model
    def transcribe(self, audio_bytes):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes); tmp = f.name
        segs, info = self.model.transcribe(tmp, task="translate", beam_size=5, vad_filter=True)
        os.unlink(tmp)
        return {"text": " ".join(s.text for s in segs).strip(), "language": info.language}

class ColabTTS:
    def __init__(self, model): self.model = model
    def synthesize(self, text, target_language="en"):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: tmp = f.name
        self.model.tts_to_file(text=text, speaker_wav="data/reference_voice.wav",
                               language=target_language, file_path=tmp)
        with open(tmp, "rb") as f: data = f.read()
        os.unlink(tmp)
        return data

initialize_models(ColabSTT(stt_model), ColabTTS(tts_model))

def run_server():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

import httpx
try:
    r = httpx.get("http://localhost:8000/health")
    print(f"✅ FastAPI server online: {r.json()}")
except:
    print("⚠️  Server starting... wait 5 more seconds")


# %% Cell 8 — Expose via ngrok
from pyngrok import ngrok

ngrok.set_auth_token(NGROK_TOKEN)
tunnel = ngrok.connect(8000, "http")
ws_url = tunnel.public_url.replace("https://", "wss://")

print(f"\n🌐 REST API  : {tunnel.public_url}")
print(f"🔌 WebSocket : {ws_url}/ws/test/session001")
print(f"📋 API Docs  : {tunnel.public_url}/docs")
print(f"📊 Metrics   : {tunnel.public_url}/metrics")


# %% Cell 9 — Interactive Text-Mode Demo (no microphone needed)
import json, asyncio, websockets

async def text_demo():
    uri = f"ws://localhost:8000/ws/test/demo_{int(time.time())}"
    print(f"\n{'='*60}")
    print("🤖 Groww AI Voice Agent — Interactive Demo")
    print(f"{'='*60}")

    async with websockets.connect(uri) as ws:
        # Receive greeting
        greeting = json.loads(await ws.recv())
        print(f"\n🤖 Agent: {greeting['text']}")

        conversations = [
            ("What is a SIP and how much do I need to start?", "en"),
            ("Mera payment fail ho gaya lekin paisa kat gaya", "hi"),
            ("How do I complete KYC?", "en"),
            ("I am very frustrated with this service!!", "en"),    # negative sentiment
            ("This is totally unacceptable. I want to escalate.", "en"),
        ]

        for user_text, lang in conversations:
            print(f"\n👤 Customer [{lang.upper()}]: {user_text}")
            await ws.send(json.dumps({"type": "text", "data": user_text, "language": lang}))

            response = json.loads(await ws.recv())
            print(f"🤖 Agent    : {response['text']}")
            print(f"   Intent   : {response.get('intent')} ({response.get('intent_confidence',0):.0%})")
            sent = response.get('sentiment', {})
            print(f"   Sentiment: {sent.get('compound', 0):+.3f} ({sent.get('label', 'N/A')})")

            if response.get("type") == "escalation":
                print(f"\n🚨 ESCALATION TRIGGERED!")
                print(f"   Payload: {json.dumps(response.get('escalation_payload',{}), indent=6)}")
                break

        await ws.send(json.dumps({"type": "end"}))
        print("\n✅ Session ended.")

# Run the demo
asyncio.run(text_demo())


# %% Cell 10 — Launch Streamlit Dashboard
def run_dashboard():
    os.system(f"streamlit run {PROJECT_ROOT}/dashboard/app.py --server.port 8501 --server.headless true")

dash_thread = threading.Thread(target=run_dashboard, daemon=True)
dash_thread.start()
time.sleep(5)

dash_tunnel = ngrok.connect(8501, "http")
print(f"\n📊 Streamlit Dashboard: {dash_tunnel.public_url}")
print("   Pages: Overview | Call Volume | Sentiment Trends | Live Monitor")
