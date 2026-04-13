"""
stt_engine.py — Speech-to-Text Engine using faster-whisper
Runs on GPU (Colab T4). Supports Hindi, English, and auto-detection.
"""

import io
import tempfile
import os
from faster_whisper import WhisperModel

# Language code mapping for downstream TTS
LANGUAGE_MAP = {
    "hi": "Hindi",
    "en": "English",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "pa": "Punjabi",
}


class STTEngine:
    """
    Wrapper around faster-whisper for real-time multilingual STT.
    Auto-detects language and returns English translation of transcript.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        """
        Args:
            model_size: Whisper model variant. 'large-v3' is best for Indian languages.
            device: 'cuda' for GPU (Colab T4), 'cpu' for local fallback.
            compute_type: 'float16' on GPU, 'int8' on CPU.
        """
        print(f"[STT] Loading faster-whisper model: {model_size} on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size
        print("[STT] Model loaded successfully.")

    def transcribe(self, audio_bytes: bytes, task: str = "translate") -> dict:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw WAV/MP3 audio bytes from WebSocket or microphone.
            task: 'translate' returns English regardless of input language.
                  'transcribe' returns text in the original language.

        Returns:
            {
                "text": str,          # English transcript
                "language": str,      # Detected ISO language code (e.g. 'hi', 'en')
                "language_name": str, # Human-readable language name
                "confidence": float,  # Language detection confidence
                "segments": list      # Word-level segments with timestamps
            }
        """
        # Write bytes to a temp file (faster-whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            segments, info = self.model.transcribe(
                tmp_path,
                task=task,
                beam_size=5,
                vad_filter=True,        # Suppress silence
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Collect all segments
            segment_list = []
            full_text = ""
            for seg in segments:
                full_text += seg.text + " "
                segment_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                })

            detected_lang = info.language
            return {
                "text": full_text.strip(),
                "language": detected_lang,
                "language_name": LANGUAGE_MAP.get(detected_lang, detected_lang.upper()),
                "confidence": round(info.language_probability, 3),
                "segments": segment_list,
            }

        finally:
            os.unlink(tmp_path)  # Clean up temp file

    def transcribe_file(self, file_path: str, task: str = "translate") -> dict:
        """Convenience method to transcribe directly from a file path."""
        with open(file_path, "rb") as f:
            return self.transcribe(f.read(), task=task)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    engine = STTEngine(model_size="large-v3", device="cuda", compute_type="float16")

    if len(sys.argv) > 1:
        result = engine.transcribe_file(sys.argv[1])
    else:
        # Generate a 1-second silent WAV for basic smoke test
        import struct, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))
        result = engine.transcribe(buf.getvalue())

    print("\n=== STT Result ===")
    print(f"Text       : {result['text']}")
    print(f"Language   : {result['language_name']} ({result['language']}) — confidence: {result['confidence']}")
    print(f"Segments   : {len(result['segments'])}")
