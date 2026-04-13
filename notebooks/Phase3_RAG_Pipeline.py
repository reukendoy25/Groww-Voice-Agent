# ============================================================
# Phase 3: Semantic FAQ Retrieval (RAG)
# Groww AI Voice Agent · Google Colab T4 GPU
#
# HOW TO USE:
#   1. Upload to Google Colab with T4 GPU runtime
#   2. Upload data/faq_documents.json to Colab session
#   3. Run cells top-to-bottom (Mistral-7B loads in ~3 mins)
# ============================================================

# %% [markdown]
"""
# 🔍 Phase 3: Semantic FAQ Retrieval (RAG)

**Embedding model**: `paraphrase-multilingual-MiniLM-L12-v2`
**Vector DB**: `FAISS` (IndexFlatIP — cosine similarity)
**LLM**: `Mistral-7B-Instruct-v0.2` (4-bit quantized, fits on Colab T4)
**Framework**: LangChain `ConversationalRetrievalChain`
"""

# %% Step 1 — Install
import subprocess, sys
pkgs = [
    "sentence-transformers==3.0.1",
    "faiss-cpu==1.8.0",
    "langchain==0.2.5",
    "langchain-community==0.2.5",
    "transformers==4.41.2",
    "bitsandbytes==0.43.1",
    "accelerate==0.30.0",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("✅ Installed")


# %% Step 2 — Load FAQ Documents + Build FAISS Index
import json, time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Load FAQ data (upload faq_documents.json or use inline sample)
try:
    with open("faq_documents.json") as f:
        faq_data = json.load(f)["faqs"]
    print(f"✅ Loaded {len(faq_data)} FAQ documents from file")
except FileNotFoundError:
    faq_data = [
        {"id":"faq_001","question":"What is SIP?","answer":"A SIP (Systematic Investment Plan) lets you invest a fixed amount regularly in a mutual fund. Minimum SIP on Groww is ₹100/month."},
        {"id":"faq_002","question":"How to withdraw from mutual fund?","answer":"Open Groww app → My Investments → Select fund → Withdraw → Enter amount → Confirm. Funds credited in 1–3 business days."},
        {"id":"faq_003","question":"What is NAV?","answer":"NAV is the per-unit price of a mutual fund, calculated daily by dividing total fund assets minus liabilities by total units outstanding."},
        {"id":"faq_004","question":"How to complete KYC on Groww?","answer":"Go to Groww app → Complete KYC → Enter PAN → Upload Aadhaar → Take selfie → Enter bank details. Takes 2–5 minutes."},
        {"id":"faq_005","question":"Is my money safe on Groww?","answer":"Yes. Mutual fund assets are held by SEBI-registered entities, not Groww. Stock investments are held in your CDSL Demat account. 256-bit SSL encryption."},
    ]
    print("⚠️  Using inline FAQ sample. Upload faq_documents.json for full 30-document index.")

# Embed all FAQ Q+A pairs
texts = [f"{doc['question']} {doc['answer']}" for doc in faq_data]
print(f"\nEmbedding {len(texts)} FAQ documents...")
t0 = time.time()
embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize
print(f"✅ Embedded in {time.time()-t0:.2f}s | shape: {embeddings.shape}")

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity (after normalization)
index.add(embeddings.astype(np.float32))
print(f"✅ FAISS index: {index.ntotal} vectors, dim={dim}")


# %% Step 3 — Retrieval Test (no LLM)
def retrieve(query: str, top_k: int = 3):
    vec = embedder.encode([query], convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec)
    scores, indices = index.search(vec.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            doc = faq_data[idx].copy()
            doc["score"] = round(float(score), 4)
            results.append(doc)
    return results

test_queries = [
    "What is the minimum amount for SIP?",
    "Mutual fund se paise kaise nikale?",
    "How safe is my money?",
]

for q in test_queries:
    print(f"\n❓ Query: {q}")
    for r in retrieve(q, top_k=2):
        print(f"   [{r['score']:.3f}] {r['question']}")


# %% Step 4 — Load Mistral-7B-Instruct (4-bit quantized)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
print(f"\nLoading {LLM_MODEL} in 4-bit (this takes ~3 mins on first run)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=250, temperature=0.3, do_sample=True,
    repetition_penalty=1.1, return_full_text=False
)
print(f"✅ Mistral-7B loaded in {time.time()-t0:.1f}s")
print(f"   Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# %% Step 5 — Full RAG Pipeline
SYSTEM_PROMPT = """You are a helpful Groww customer support agent. Answer using ONLY the provided FAQ context.
If unsure, say "I'll connect you with a specialist."

FAQ CONTEXT:
{context}
"""

def rag_answer(query: str) -> str:
    chunks = retrieve(query, top_k=3)
    context = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in chunks])
    prompt = f"<s>[INST] {SYSTEM_PROMPT.format(context=context)}\n\nCustomer: {query} [/INST]"
    t0 = time.time()
    out = pipe(prompt)[0]["generated_text"].strip()
    print(f"   ⚡ Generation: {time.time()-t0:.2f}s")
    return out

# Multi-turn conversation demo
demo_questions = [
    "What is a SIP and how do I start one?",
    "What is the minimum amount I need?",
    "How do I cancel it if I want to?",
]

history = ""
print("\n" + "="*60)
print("🎯 Multi-turn RAG Conversation Demo")
print("="*60)
for q in demo_questions:
    print(f"\n👤 Customer: {q}")
    answer = rag_answer(q)
    print(f"🤖 Agent   : {answer}")
    history += f"Customer: {q}\nAgent: {answer}\n"
