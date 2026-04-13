"""
intent_classifier.py — Centroid-Based Intent Classifier
Uses paraphrase-multilingual-MiniLM-L12-v2 embeddings (handles Hindi + English natively).
Classification is sub-millisecond cos-similarity lookup — no LLM needed.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intent_training_data.json")


class IntentClassifier:
    """
    Classifies customer support utterances into predefined intents using
    dense vector centroid matching. Works for Hindi and English text.

    How it works:
      1. Each training example → embedding via MiniLM-L12
      2. Embeddings for same intent → averaged to form centroid
      3. New query → embed → cosine similarity to all centroids
      4. Highest similarity = predicted intent
    """

    FALLBACK_INTENT = "general_complaint"
    CONFIDENCE_THRESHOLD = 0.35  # Below this → treat as fallback

    def __init__(self):
        print(f"[Intent] Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.intent_centroids: dict[str, np.ndarray] = {}
        self.intent_labels: list[str] = []
        self._build_centroids()
        print(f"[Intent] Ready. Intents: {self.intent_labels}")

    def _build_centroids(self):
        """Load training data and compute per-intent embedding centroids."""
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for intent_block in data["intents"]:
            label = intent_block["label"]
            examples = intent_block["examples"]

            # Batch encode all examples at once (faster)
            embeddings = self.model.encode(examples, convert_to_numpy=True, show_progress_bar=False)

            # Centroid = mean of all example embeddings
            centroid = np.mean(embeddings, axis=0)
            # Normalize to unit vector for cleaner cosine similarity
            centroid = centroid / np.linalg.norm(centroid)

            self.intent_centroids[label] = centroid
            self.intent_labels.append(label)

        print(f"[Intent] Built centroids for {len(self.intent_labels)} intents.")

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify input text into an intent.

        Args:
            text: Customer utterance (Hindi, English, or mixed).

        Returns:
            Tuple of (intent_label, confidence_score)
            confidence_score is cosine similarity [0, 1]
        """
        if not text or not text.strip():
            return self.FALLBACK_INTENT, 0.0

        # Embed the query
        query_vec = self.model.encode([text], convert_to_numpy=True)[0]
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Stack centroids into matrix for batch similarity
        centroid_matrix = np.stack(
            [self.intent_centroids[label] for label in self.intent_labels], axis=0
        )  # shape: (num_intents, embedding_dim)

        # Cosine similarity against all centroids
        similarities = cosine_similarity(query_vec.reshape(1, -1), centroid_matrix)[0]

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_label = self.intent_labels[best_idx]

        # Apply confidence threshold
        if best_score < self.CONFIDENCE_THRESHOLD:
            return self.FALLBACK_INTENT, best_score

        return best_label, round(best_score, 4)

    def classify_with_scores(self, text: str) -> dict:
        """
        Returns full similarity scores for all intents — useful for debugging.
        """
        query_vec = self.model.encode([text], convert_to_numpy=True)[0]
        query_vec = query_vec / np.linalg.norm(query_vec)

        centroid_matrix = np.stack(
            [self.intent_centroids[label] for label in self.intent_labels], axis=0
        )
        similarities = cosine_similarity(query_vec.reshape(1, -1), centroid_matrix)[0]

        scores = {
            label: round(float(sim), 4)
            for label, sim in zip(self.intent_labels, similarities)
        }
        predicted, confidence = self.classify(text)
        scores["_predicted"] = predicted
        scores["_confidence"] = confidence
        return scores


# ── Singleton for use across the app ────────────────────────────────────────
_classifier_instance = None

def get_classifier() -> IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = IntentClassifier()

    test_cases = [
        ("Where is my order? It's been 5 days!", "order_status"),
        ("Mujhe refund chahiye mere order ka", "refund_request"),
        ("SIP kaise start karein?", "product_query"),
        ("Login nahi ho raha mera account mein", "account_issue"),
        ("Payment fail ho gaya lekin paisa kat gaya", "payment_failure"),
        ("This service is absolutely terrible!", "general_complaint"),
        # Multilingual test
        ("Mera order track karo please", "order_status"),
        ("What is NAV in mutual fund?", "product_query"),
    ]

    print("\n{:<45} {:<22} {:<10} {}".format("Input", "Expected", "Predicted", "Score"))
    print("-" * 90)
    correct = 0
    for text, expected in test_cases:
        predicted, score = clf.classify(text)
        ok = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        print(f"{ok} {text[:43]:<43} {expected:<22} {predicted:<22} {score:.3f}")

    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")
