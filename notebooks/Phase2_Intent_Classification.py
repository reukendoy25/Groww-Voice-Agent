# ============================================================
# Phase 2: Embedding-Based Intent Classification
# Groww AI Voice Agent · Google Colab (CPU sufficient)
#
# HOW TO USE:
#   1. Upload to Google Colab
#   2. Upload data/intent_training_data.json to Colab session
#   3. Run cells top-to-bottom
# ============================================================

# %% [markdown]
"""
# 🧠 Phase 2: Intent Classification with Multilingual Embeddings

**Model**: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
**Method**: Centroid-based cosine similarity — no LLM, sub-millisecond inference

How it works:
1. Each training example → 384-dim dense embedding vector
2. Embeddings per intent → averaged to a single `centroid`
3. New query → embed → cosine similarity to all centroids
4. Highest similarity = predicted intent
"""

# %% Step 1 — Install
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers==3.0.1", "scikit-learn", "plotly"], check=True)
print("✅ Installed")


# %% Step 2 — Load Model + Build Centroids
import json, time, os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("✅ Model loaded")

# Load training data (upload intent_training_data.json to Colab or use inline)
# If running without the file, we embed the inline sample below
TRAINING_DATA = {
  "intents": [
    {"label": "order_status",     "examples": ["Where is my order?","Mera order kab aayega?","Track my shipment","Order status check karna hai","When will my order arrive?"]},
    {"label": "refund_request",   "examples": ["I want a refund","Paise wapas chahiye","Please process my refund","Refund nahi aaya","Money return karo"]},
    {"label": "product_query",    "examples": ["What is SIP?","SIP kaise kaam karta hai?","Minimum investment amount?","How to withdraw mutual fund?","KYC kaise karein?"]},
    {"label": "account_issue",    "examples": ["Login nahi ho raha","I forgot my password","Account locked hai","Mobile number change karna hai","MPIN reset karo"]},
    {"label": "payment_failure",  "examples": ["Payment fail ho gaya","UPI failed","Paise kat gaye order nahi hua","Double charge ho gaya","Transaction pending hai"]},
    {"label": "general_complaint", "examples": ["This is unacceptable","Bahut bura service hai","I am very frustrated","Koi help nahi kar raha","Main bahut pareshan hoon"]},
  ]
}

try:
    with open("intent_training_data.json") as f:
        TRAINING_DATA = json.load(f)
    print("✅ Loaded training data from file")
except FileNotFoundError:
    print("⚠️  Using inline training data. Upload intent_training_data.json for full 15-example training.")

# Build centroids
centroids = {}
labels = []
t0 = time.time()

for intent_block in TRAINING_DATA["intents"]:
    label = intent_block["label"]
    examples = intent_block["examples"]
    embs = model.encode(examples, convert_to_numpy=True, show_progress_bar=False)
    centroid = np.mean(embs, axis=0)
    centroid /= np.linalg.norm(centroid)   # normalize
    centroids[label] = centroid
    labels.append(label)

print(f"✅ Centroids built for {len(labels)} intents in {time.time()-t0:.2f}s")
print(f"   Embedding dim: {list(centroids.values())[0].shape[0]}")


# %% Step 3 — Classify Function
def classify(text: str, threshold: float = 0.35):
    """Classify text into an intent using cosine similarity to centroids."""
    vec = model.encode([text], convert_to_numpy=True)[0]
    vec /= np.linalg.norm(vec)

    matrix = np.stack([centroids[l] for l in labels])
    sims = cosine_similarity(vec.reshape(1,-1), matrix)[0]

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_label = labels[best_idx]

    if best_score < threshold:
        return "general_complaint", best_score
    return best_label, round(best_score, 4)


# %% Step 4 — Test on Sample Utterances
test_cases = [
    ("Where is my order? It's been 5 days!",       "order_status"),
    ("Mujhe refund chahiye mere order ka",          "refund_request"),
    ("SIP kaise start karein?",                     "product_query"),
    ("Login nahi ho raha mera account mein",        "account_issue"),
    ("Payment fail ho gaya lekin paisa kat gaya",   "payment_failure"),
    ("This service is absolutely terrible!",        "general_complaint"),
    ("Mera order track karo please",                "order_status"),
    ("What is NAV in mutual fund?",                 "product_query"),
    ("I want to change my bank account",            "account_issue"),
    ("UPI transaction failed twice",                "payment_failure"),
]

print(f"\n{'Input':<45} {'Expected':<22} {'Predicted':<22} {'Score'}")
print("─" * 100)
correct = 0
for text, expected in test_cases:
    t0 = time.time()
    predicted, score = classify(text)
    ms = (time.time()-t0)*1000
    ok = "✅" if predicted == expected else "❌"
    if predicted == expected:
        correct += 1
    print(f"{ok} {text[:43]:<43} {expected:<22} {predicted:<22} {score:.3f}  ({ms:.1f}ms)")

print(f"\n🎯 Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")


# %% Step 5 — Visualize Centroid Similarity Heatmap
try:
    import plotly.express as px
    import pandas as pd

    centroid_matrix = np.stack([centroids[l] for l in labels])
    sim_matrix = cosine_similarity(centroid_matrix, centroid_matrix)

    df_heat = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    fig = px.imshow(
        df_heat,
        text_auto=".2f",
        title="Inter-Intent Centroid Cosine Similarity",
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=1,
    )
    fig.show()
    print("✅ Heatmap displayed — lower off-diagonal values = better class separation")
except Exception as e:
    print(f"Plotly not available: {e}")
