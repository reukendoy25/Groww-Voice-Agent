"""
tts_engine.py — Text-to-Speech Engine using Coqui XTTS-v2
Multilingual TTS that supports Hindi, English, and other Indian languages.
Runs on GPU (Colab T4).
"""

import io
import os
import tempfile
from TTS.api import TTS

# Mapping from ISO language codes to XTTS-v2 supported language codes
LANGUAGE_MAP = {
    "hi": "hi",   # Hindi
    "en": "en",   # English
    "bn": "bn",   # Bengali
    "te": "te",   # Telugu (XTTS-v2 may fall back to en)
    "ta": "ta",   # Tamil
    "mr": "mr",   # Marathi
    "gu": "gu",   # Gujarati (falls back to hi in XTTS-v2)
}

# Languages not natively in XTTS-v2 fall back to Hindi as closest
FALLBACK_LANGUAGE = "hi"

# Reference audio for voice cloning (use a 6–10 second clean Hindi/English speaker clip)
# In Colab, we use a synthetic reference; for production, use a real agent voice sample.
DEFAULT_REFERENCE_AUDIO = os.path.join(
    os.path.dirname(__file__), "..", "data", "reference_voice.wav"
)


class TTSEngine:
    """
    Wrapper around Coqui XTTS-v2 for multilingual Text-to-Speech synthesis.
    Converts English agent responses into the caller's native language audio.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: If True, loads XTTS-v2 on CUDA (recommended for Colab T4).
        """
        print("[TTS] Loading Coqui XTTS-v2 model...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if use_gpu:
            self.tts = self.tts.to("cuda")
        print("[TTS] XTTS-v2 loaded successfully.")

        # Ensure reference audio exists for voice cloning
        self._reference_audio = self._ensure_reference_audio()

    def _ensure_reference_audio(self) -> str:
        """
        Checks for reference audio file. Creates a silent placeholder if not present.
        For best results, replace data/reference_voice.wav with a 6-10 second
        recording of a clear, neutral-accent English/Hindi speaker.
        """
        if os.path.exists(DEFAULT_REFERENCE_AUDIO):
            return DEFAULT_REFERENCE_AUDIO

        # Create a minimal valid WAV as placeholder
        import struct, wave
        os.makedirs(os.path.dirname(DEFAULT_REFERENCE_AUDIO), exist_ok=True)
        print(f"[TTS] Warning: No reference voice found at {DEFAULT_REFERENCE_AUDIO}.")
        print("[TTS] Using silent placeholder. Replace with a real voice sample for better quality.")
        with wave.open(DEFAULT_REFERENCE_AUDIO, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            # 3 seconds of silence
            wf.writeframes(struct.pack("<" + "h" * 66150, *([0] * 66150)))
        return DEFAULT_REFERENCE_AUDIO

    def synthesize(self, text: str, target_language: str = "en") -> bytes:
        """
        Convert text to speech in the target language.

        Args:
            text: The English-language agent response text to speak.
            target_language: ISO language code of the caller (e.g., 'hi', 'en').
                             XTTS-v2 will adapt prosody/phonetics accordingly.

        Returns:
            WAV audio bytes ready to stream back over WebSocket.
        """
        lang = LANGUAGE_MAP.get(target_language, FALLBACK_LANGUAGE)

        # XTTS-v2 requires text in the target language; for non-English output,
        # the text should ideally be translated first. If text is in English and
        # target is Hindi, XTTS-v2 still produces accented/intelligible Hindi output.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self._reference_audio,
                language=lang,
                file_path=tmp_path,
            )
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        finally:
            os.unlink(tmp_path)

    def synthesize_stream(self, text: str, target_language: str = "en"):
        """
        Generator that yields WAV audio bytes for streaming.
        Splits long text into sentences for lower first-byte latency.
        """
        import re
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?।]) +", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                yield self.synthesize(sentence, target_language)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = TTSEngine(use_gpu=True)

    test_cases = [
        ("Hello! How can I help you with your Groww account today?", "en"),
        ("Namaste! Aapki kaise madad kar sakta hoon?", "hi"),
    ]

    for text, lang in test_cases:
        print(f"\n[TTS] Synthesizing in '{lang}': {text[:50]}...")
        audio = engine.synthesize(text, target_language=lang)
        out_path = f"test_tts_{lang}.wav"
        with open(out_path, "wb") as f:
            f.write(audio)
        print(f"[TTS] Saved {len(audio)} bytes → {out_path}")
