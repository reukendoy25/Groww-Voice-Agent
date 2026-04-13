"""
rag_engine.py — FAISS-powered Retrieval-Augmented Generation (RAG)
Uses: paraphrase-multilingual-MiniLM-L12-v2 for embeddings
      FAISS for vector search
      Mistral-7B-Instruct-v0.2 (4-bit via bitsandbytes) for generation
      LangChain for orchestration + conversation memory
"""

import json
import os
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

# Lazy imports for heavy models (loaded only when needed)
_llm = None
_tokenizer = None

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FAQ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faq_documents.json")
TOP_K = 4  # Number of FAQ chunks to retrieve per query


SYSTEM_PROMPT = """You are a helpful, empathetic customer support agent for Groww — India's leading investment platform.
You assist customers with questions about mutual funds, SIPs, stocks, KYC, payments, and account management.

GUIDELINES:
- Be concise and friendly. Speak in simple language.
- Base your answers ONLY on the provided FAQ context.
- If the answer is not in the context, say "I don't have that specific information right now, but I can connect you with a Groww specialist."
- Never make up financial advice or invented policies.
- If the customer is frustrated, acknowledge their feeling before answering.

FAQ CONTEXT:
{context}
"""


def _load_llm():
    """Lazy-load Mistral-7B with 4-bit quantization (fits on Colab T4)."""
    global _llm, _tokenizer
    if _llm is not None:
        return _llm, _tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    print(f"[RAG] Loading {LLM_MODEL} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    _llm = HuggingFacePipeline(pipeline=pipe)
    print("[RAG] Mistral-7B loaded.")
    return _llm, _tokenizer


class RAGEngine:
    """
    FAISS-backed RAG engine for Groww product queries.

    Flow:
        query → embed → FAISS top-K search → inject FAQ chunks → Mistral-7B → answer
    """

    def __init__(self):
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.faq_docs: list[dict] = []
        self.index: Optional[faiss.Index] = None
        self.doc_embeddings: Optional[np.ndarray] = None

        self._build_index()

        # Per-session memories: {session_id: ConversationBufferWindowMemory}
        self._memories: dict[str, ConversationBufferWindowMemory] = {}

    def _build_index(self):
        """Load FAQ JSON, embed all Q+A pairs, build FAISS flat index."""
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.faq_docs = data["faqs"]

        # Embed concatenation of Q + A for richer retrieval
        texts = [f"{doc['question']} {doc['answer']}" for doc in self.faq_docs]
        print(f"[RAG] Embedding {len(texts)} FAQ documents...")
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner product == cosine after normalization
        self.index.add(embeddings.astype(np.float32))
        self.doc_embeddings = embeddings

        print(f"[RAG] FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve the most relevant FAQ chunks for the given query.

        Args:
            query: Customer's question (in English after STT translation).
            top_k: Number of chunks to return.

        Returns:
            List of FAQ dicts with 'question', 'answer', 'score'.
        """
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec)

        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc = self.faq_docs[idx].copy()
                doc["score"] = round(float(score), 4)
                results.append(doc)
        return results

    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Return or create per-session conversation memory (last 5 turns)."""
        if session_id not in self._memories:
            self._memories[session_id] = ConversationBufferWindowMemory(k=5)
        return self._memories[session_id]

    def answer(self, query: str, session_id: str) -> str:
        """
        Full RAG pipeline: retrieve → build prompt → generate → return answer.

        Args:
            query: Customer's question in English.
            session_id: Unique session identifier for memory tracking.

        Returns:
            Agent's answer string.
        """
        llm, tokenizer = _load_llm()
        memory = self._get_memory(session_id)

        # 1. Retrieve relevant FAQ chunks
        chunks = self.retrieve(query)
        context = "\n\n".join(
            [f"Q: {c['question']}\nA: {c['answer']}" for c in chunks]
        )

        # 2. Build conversation history
        history = memory.load_memory_variables({}).get("history", "")

        # 3. Build Mistral-format prompt
        prompt = (
            f"<s>[INST] {SYSTEM_PROMPT.format(context=context)}\n\n"
            f"Conversation so far:\n{history}\n\n"
            f"Customer: {query} [/INST]"
        )

        # 4. Generate answer
        response = llm.invoke(prompt)
        answer_text = response.strip()

        # 5. Save turn to memory
        memory.save_context({"input": query}, {"output": answer_text})

        return answer_text

    def clear_session(self, session_id: str):
        """Clear conversation memory for a session (call ended)."""
        self._memories.pop(session_id, None)


# ── Singleton ────────────────────────────────────────────────────────────────
_rag_instance = None

def get_rag_engine() -> RAGEngine:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = RAGEngine()

    # Test 1: Retrieval without LLM
    print("\n=== Retrieval Test ===")
    query = "What is the minimum SIP amount?"
    chunks = engine.retrieve(query, top_k=2)
    for c in chunks:
        print(f"  [{c['score']:.3f}] {c['question']}")

    # Test 2: Full RAG (requires Mistral-7B on GPU)
    print("\n=== Full RAG Test ===")
    answer = engine.answer("How do I stop my SIP?", session_id="test_001")
    print(f"Answer: {answer}")
