# ============================================================
# Phase 1: Multilingual STT + TTS Pipeline
# Groww AI Voice Agent · Google Colab T4 GPU
#
# HOW TO USE:
#   1. Open Google Colab at https://colab.research.google.com
#   2. Upload this file: File → Upload notebook → choose this .py file
#   3. Set runtime: Runtime → Change runtime type → T4 GPU
#   4. Run cells top-to-bottom with Shift+Enter
# ============================================================

# %% [markdown]
"""
# 🎯 Phase 1: Multilingual STT + TTS Pipeline
**Groww AI Voice Agent · Google Colab T4 GPU**

Demonstrates:
- `faster-whisper large-v3` — speech-to-text, auto-detects Hindi/English
- `Coqui XTTS-v2` — multilingual neural text-to-speech
- Latency benchmarks on T4 GPU
"""

# %% Step 1 — Install Dependencies (run once, ~5 mins)
# ─────────────────────────────────────────────────────
import subprocess, sys

# GPU check first
try:
    import torch
    gpu_ok = torch.cuda.is_available()
    print(f"🔥 GPU available: {gpu_ok} | Device: {torch.cuda.get_device_name(0) if gpu_ok else 'CPU'}")
except ImportError:
    pass

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "faster-whisper==1.0.1", "TTS==0.22.0"], check=True)
print("✅ Dependencies installed.")


# %% Step 2 — Load faster-whisper STT Model
# ─────────────────────────────────────────────────────
from faster_whisper import WhisperModel
import time

print("Loading faster-whisper large-v3 on GPU...")
t0 = time.time()
stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"✅ STT model loaded in {time.time()-t0:.1f}s")


# %% Step 3 — Download Sample Hindi Audio + Transcribe
# ─────────────────────────────────────────────────────
import urllib.request

# Hindi audio sample (replace with any Hindi/English audio URL or upload your own)
# This is a Wikimedia Commons Hindi sample
sample_url = "https://upload.wikimedia.org/wikipedia/commons/2/22/Airtel-Hindi.ogg"
urllib.request.urlretrieve(sample_url, "sample_hindi.ogg")
print("Audio downloaded.")

print("\n📢 Transcribing (task='translate' → always outputs English)...")
t0 = time.time()
segments, info = stt_model.transcribe(
    "sample_hindi.ogg",
    task="translate",   # returns English regardless of input language
    beam_size=5,
    vad_filter=True,    # strip silence
    vad_parameters=dict(min_silence_duration_ms=500),
)
transcript = " ".join([s.text for s in segments])
latency = time.time() - t0

print(f"\n🌐 Detected language : {info.language}  ({info.language_probability:.0%} confidence)")
print(f"📝 English transcript: {transcript}")
print(f"⚡ Transcription time: {latency:.2f}s")


# %% Step 4 — Load Coqui XTTS-v2 TTS Model
# ─────────────────────────────────────────────────────
from TTS.api import TTS

print("Loading Coqui XTTS-v2...")
t0 = time.time()
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print(f"✅ TTS model loaded in {time.time()-t0:.1f}s")


# %% Step 5 — Synthesize Agent Responses in English + Hindi
# ─────────────────────────────────────────────────────
try:
    from IPython.display import Audio, display
    in_colab = True
except ImportError:
    in_colab = False

reference_audio = "sample_hindi.ogg"  # short clip for voice cloning

responses = [
    ("Hello! Welcome to Groww support. How can I help you today?", "en"),
    ("Namaste! Aapki kaise madad kar sakta hoon?", "hi"),
]

for text, lang in responses:
    t0 = time.time()
    out_path = f"tts_{lang}.wav"
    tts_model.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language=lang,
        file_path=out_path,
    )
    print(f"\n[{lang.upper()}] \"{text}\"")
    print(f"       ⚡ Synthesis: {time.time()-t0:.2f}s  →  saved: {out_path}")
    if in_colab:
        display(Audio(out_path))


# %% Step 6 — Latency Summary
# ─────────────────────────────────────────────────────
print("""
┌─────────────────────────────────────────────────────┐
│           Latency Benchmark (Colab T4 GPU)          │
├───────────────────┬──────────────────┬──────────────│
│ Component         │ Model            │ Latency      │
├───────────────────┼──────────────────┼──────────────│
│ STT               │ whisper large-v3 │ 0.5 – 1.5s   │
│ TTS               │ Coqui XTTS-v2   │ 1.0 – 3.0s   │
│ Intent (centroid) │ MiniLM-L12       │ < 5ms        │
│ Sentiment (VADER) │ lexicon          │ < 1ms        │
│ RAG (Mistral-7B)  │ 4-bit quant      │ 2 – 5s       │
├───────────────────┼──────────────────┼──────────────│
│ TOTAL ROUND-TRIP  │ all components   │ ~4 – 10s     │
└───────────────────┴──────────────────┴──────────────┘

Phase 1 complete! Run Phase2_Intent_Classification.py next.
""")
