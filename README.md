# Groww: Multilingual AI Voice Agent for D2C Customer Support

> A production-grade, India-first AI voice agent that handles multilingual (Hindi/English) customer support calls in real time — powered entirely by open-source models on Google Colab T4 GPU.

---

## Architecture

```
📞 Caller (Hindi/English)
        │
        ▼ Raw Audio bytes (WebSocket)
┌──────────────────────────────┐
│  FastAPI WebSocket Server    │  ← tunneled via ngrok from Colab
└─────────────┬────────────────┘
              │
     ┌────────▼────────┐
     │ faster-whisper  │  STT: Hindi → English transcript
     │   large-v3      │  (Colab T4 GPU)
     └────────┬────────┘
              │ English transcript + language tag
     ┌────────▼────────────────────────────────────┐
     │               Agent Pipeline                │
     │                                             │
     │  ┌────────────┐   ┌──────────────────────┐  │
     │  │ MiniLM-L12 │   │    VADER Sentiment    │  │
     │  │  Intent    │   │  (instant, CPU-only)  │  │
     │  │ Classifier │   └──────────┬───────────┘  │
     │  └─────┬──────┘              │ escalate?     │
     │        │                     ▼               │
     │    product_query    🚨 Escalation Payload    │
     │        ▼                                     │
     │  ┌────────────┐   other intents              │
     │  │ FAISS RAG  │──────────────────┐           │
     │  │ Mistral-7B │  Canned Response │           │
     │  └────────────┘                  │           │
     └──────────────────────────────────┼───────────┘
                                        │ English response
                               ┌────────▼────────┐
                               │  Coqui XTTS-v2  │  TTS → native audio
                               │   (T4 GPU)      │
                               └────────┬────────┘
                                        │ WAV bytes
                               returns to caller
                                        │
                               ┌────────▼────────┐
                               │   SQLite DB     │  KPI logging
                               └────────┬────────┘
                                        │
                               ┌────────▼────────┐
                               │ Streamlit Dash  │  localhost:8501
                               └─────────────────┘
```

---

## Tech Stack

| Component | Technology | Model/Library |
|---|---|---|
| Speech-to-Text | `faster-whisper` | `large-v3` (best Hindi accuracy) |
| Text-to-Speech | `Coqui TTS` | `xtts_v2` (multilingual) |
| LLM (RAG) | `transformers` | `Mistral-7B-Instruct-v0.2` (4-bit) |
| Embeddings | `sentence-transformers` | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | `faiss-cpu` | `IndexFlatIP` (cosine similarity) |
| Sentiment | `vaderSentiment` | lexicon-based, instant |
| Orchestration | `langchain` | `ConversationalRetrievalChain` |
| API | `FastAPI` | WebSocket + REST |
| Dashboard | `Streamlit` + `Plotly` | 4-page KPI dashboard |
| Infra | `Docker` + `ngrok` | containerized + Colab tunneling |

---

## Project Structure

```
Updated Groww/
├── app/
│   ├── main.py              # FastAPI entry point + WebSocket routes
│   ├── websocket_handler.py # Full-duplex audio session manager
│   ├── stt_engine.py        # faster-whisper STT wrapper
│   ├── tts_engine.py        # Coqui XTTS-v2 TTS wrapper
│   ├── intent_classifier.py # MiniLM-L12 centroid intent classifier
│   ├── rag_engine.py        # FAISS + Mistral-7B RAG pipeline
│   ├── sentiment_engine.py  # VADER sentiment + escalation logic
│   ├── agent_pipeline.py    # Central orchestrator (per-session)
│   └── database.py          # SQLAlchemy models + 100-call seed
├── dashboard/
│   ├── app.py               # Streamlit KPI dashboard (4 pages)
│   └── Dockerfile
├── notebooks/
│   ├── Phase1_STT_TTS.py              # STT + TTS demo
│   ├── Phase2_Intent_Classification.py # Embedding + centroid classifier
│   ├── Phase3_RAG_Pipeline.py          # FAISS + Mistral-7B RAG
│   ├── Phase4_Sentiment_Escalation.py  # VADER + escalation demo
│   ├── Phase5_Dashboard.py             # Streamlit via ngrok
│   └── VoiceAgent_Complete.py          # ⭐ Master integration notebook
├── data/
│   ├── intent_training_data.json  # 6 intents × 15 bilingual utterances
│   └── faq_documents.json         # 30+ Groww support FAQs
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start — Google Colab (Recommended)

### Prerequisites
- Google account (for Colab + Drive)
- Free [ngrok](https://ngrok.com) account for tunneling

### Steps

**1. Upload project to Google Drive**
```
Upload the entire "Updated Groww" folder to My Drive/groww/
```

**2. Open `VoiceAgent_Complete.py` in Colab**
```
File → Upload notebook → select VoiceAgent_Complete.py
Runtime → Change runtime type → T4 GPU
```

**3. Set your ngrok token**
```python
# In Cell 1 of the notebook:
NGROK_TOKEN = "your_token_here"   # from ngrok.com dashboard
PROJECT_ROOT = "/content/drive/MyDrive/groww"
```

**4. Run All cells** (`Runtime → Run all`)

The notebook will:
- Install all dependencies (~8 mins first time)
- Load all 4 models (STT, TTS, LLM, Embeddings)
- Start FastAPI WebSocket server
- Print public ngrok URL for the API
- Run a 5-turn interactive demo conversation
- Launch Streamlit dashboard with a second ngrok URL

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Live KPI snapshot (FCR, CSAT, escalation rate) |
| `GET` | `/sessions/active` | Currently active WebSocket sessions |
| `POST` | `/sessions/new` | Generate new session ID |

### WebSocket Protocol

**Connect**: `ws://localhost:8000/ws/{session_id}`

**Send (text-mode for testing)**:
```json
{"type": "text", "data": "What is the minimum SIP amount?", "language": "en"}
```

**Send (audio mode for production)**:
```json
{"type": "audio", "data": "<base64-encoded WAV bytes>"}
```

**End session**:
```json
{"type": "end"}
```

**Receive**:
```json
{
  "type": "response",
  "text": "The minimum SIP amount on Groww is ₹100 per month.",
  "audio": "<base64 WAV>",
  "intent": "product_query",
  "intent_confidence": 0.872,
  "sentiment": {"compound": 0.12, "label": "positive"},
  "escalate": false,
  "turn": 1
}
```

---

## How Intent Classification Works

1. **Training**: 6 intents × 15 bilingual utterances (Hindi + English) embedded using `paraphrase-multilingual-MiniLM-L12-v2`
2. **Centroid**: Per-intent embeddings averaged → single representative vector
3. **Inference**: Query vector → cosine similarity to all 6 centroids → highest = predicted intent
4. **Speed**: < 5ms per classification (CPU, no LLM call)

**Intents**: `order_status` · `refund_request` · `product_query` · `account_issue` · `payment_failure` · `general_complaint`

---

## Escalation Logic

VADER compound scores range from **-1.0** (extremely negative) to **+1.0** (extremely positive).

| Rule | Condition | Action |
|---|---|---|
| Hard threshold | `compound < -0.6` on any single turn | Immediate escalation |
| Trend escalation | Last 3 consecutive turns all `< -0.2` | Soft escalation |

On escalation: agent sends apology, generates JSON payload (session ID, reason, full sentiment history, call summary) for supervisor handoff, logs `EscalationEvent` to DB.

---

## Streamlit Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | Total calls, FCR%, CSAT score, escalation rate, resolution donut, language distribution |
| **Call Volume** | Intent bar chart, hourly heatmap, daily area trend, RAG vs. canned breakdown |
| **Sentiment Trends** | Sentiment distribution, avg sentiment by intent, timeline with escalation markers |
| **Live Monitor** | Recent calls table, live API metrics (requires server running) |

---

## Docker Deployment (Local GPU)

```bash
# Start both services
docker-compose up --build

# FastAPI:   http://localhost:8000/docs
# Dashboard: http://localhost:8501

# Test WebSocket via wscat:
npm install -g wscat
wscat -c ws://localhost:8000/ws/test/demo01
> {"type": "text", "data": "How do I complete KYC?"}
```

---

## References

| Resource | Relevance |
|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2-based Whisper for 4× faster GPU inference |
| [Coqui TTS / XTTS-v2](https://github.com/coqui-ai/TTS) | Multilingual zero-shot neural TTS |
| [sentence-transformers](https://www.sbert.net/) | Efficient sentence embeddings for intent and RAG |
| [FAISS](https://faiss.ai) | Facebook AI similarity search for vector retrieval |
| [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | Best-in-class open-source LLM for instruction following |
| [vaderSentiment](https://github.com/cjhutto/vaderSentiment) | Hutto & Gilbert (2014) — lexicon-based sentiment analysis |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM orchestration and conversational memory |
| [Streamlit](https://streamlit.io/) | Rapid Python data dashboard development |
| GC Open Soft 2026 — Problem Statement | Architecture blueprint for this project |

---

*Built for Groww Internship · March 2026 · Powered by open-source AI*
