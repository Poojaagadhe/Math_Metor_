"""Audio input processor - uses Groq Whisper for transcription (free, no OpenAI key needed)"""
import os
from typing import Dict, Any, Optional
from pathlib import Path

from utils.config import Config
from utils.logger import setup_logger
from utils.hitl import HITLManager, HITLTrigger

logger = setup_logger(__name__)


class AudioProcessor:
    """Processes audio inputs and converts to text using Groq's Whisper endpoint"""

    def __init__(self):
        """Initialize AudioProcessor"""
        self._client = None
        self.provider = None
        self.hitl_manager = HITLManager()
        logger.info("AudioProcessor initialized (transcription client will be lazy-loaded)")

    @property
    def client(self):
        """Lazy-load the transcription client."""
        if self._client is None and self.provider is None:
            self._initialize_client()
        return self._client

    def _initialize_client(self):
        """Initialize Groq or OpenAI client."""
        # Try Groq first
        groq_key = getattr(Config, "GROQ_API_KEY", None)
        if groq_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=groq_key)
                self.provider = "groq"
                logger.info("AudioProcessor client initialized with Groq Whisper")
                return
            except ImportError:
                logger.warning("groq package not installed")

        # Fallback: OpenAI Whisper
        openai_key = getattr(Config, "OPENAI_API_KEY", None)
        if openai_key:
            try:
                import openai
                # Local import to avoid top-level slow import
                self._client = openai.OpenAI(api_key=openai_key)
                self.provider = "openai"
                logger.info("AudioProcessor client initialized with OpenAI Whisper")
                return
            except (ImportError, AttributeError):
                pass

        logger.warning("AudioProcessor: No API key found for Groq or OpenAI.")
        self.provider = "unavailable"

    # ------------------------------------------------------------------

    def process_audio(
        self,
        audio_input: Any,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_input: File path (str | Path), bytes, or file-like object.
            save_path: Optional path to save audio before processing.

        Returns:
            {transcript, confidence, audio_path, hitl_required, language}
        """
        logger.info("Processing audio input...")

        if self.provider == "unavailable" or (self.provider is None and self.client is None and self.provider == "unavailable"):
            return self._unavailable_result()

        # Resolve to a file path
        audio_path = self._resolve_audio_path(audio_input, save_path)
        if not audio_path:
            return self._unavailable_result("Could not resolve audio file path.")

        try:
            transcript = ""
            language = "en"

            if self.provider == "groq":
                transcript, language = self._transcribe_groq(audio_path)
            elif self.provider == "openai":
                transcript, language = self._transcribe_openai(audio_path)

            if not transcript:
                return self._unavailable_result("Transcription returned empty result.")

            # Post-process spoken math terms → symbols
            transcript = self._postprocess_math_terms(transcript)
            confidence = self._estimate_confidence(transcript)

            logger.info(
                f"Transcription done | provider={self.provider} "
                f"| language={language} | confidence={confidence:.2f} "
                f"| text={transcript[:80]}"
            )

            # HITL check
            hitl_required = False
            hitl_intervention = None
            if len(transcript.strip()) < 10 or confidence < 0.6:
                hitl_required = True
                hitl_intervention = self.hitl_manager.create_intervention(
                    trigger_type=HITLTrigger.LOW_ASR_CONFIDENCE,
                    message=f"Low-confidence transcription ({confidence:.0%}). Please verify.",
                    data={"transcript": transcript, "confidence": confidence},
                    suggestions=["Review the text", "Correct any errors", "Re-record if needed"]
                )

            return {
                "transcript": transcript,
                "confidence": float(confidence),
                "audio_path": audio_path,
                "hitl_required": hitl_required,
                "hitl_intervention": hitl_intervention,
                "language": language
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return self._unavailable_result(str(e))

    # ------------------------------------------------------------------
    # Provider-specific transcription
    # ------------------------------------------------------------------

    def _transcribe_groq(self, audio_path: str):
        """Transcribe using Groq's Whisper endpoint (whisper-large-v3-turbo)."""
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="verbose_json"
            )
        transcript = getattr(response, "text", "") or ""
        language = getattr(response, "language", "en") or "en"
        return transcript.strip(), language

    def _transcribe_openai(self, audio_path: str):
        """Transcribe using OpenAI Whisper API."""
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )
        transcript = getattr(response, "text", "") or ""
        language = getattr(response, "language", "en") or "en"
        return transcript.strip(), language

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_audio_path(self, audio_input: Any, save_path: Optional[Path]) -> Optional[str]:
        """Convert any audio input type to a file path string."""
        import tempfile

        if isinstance(audio_input, (str, Path)):
            return str(audio_input)

        if isinstance(audio_input, bytes):
            target = save_path or Path(tempfile.mktemp(suffix=".wav"))
            with open(target, "wb") as f:
                f.write(audio_input)
            return str(target)

        # Streamlit UploadedFile or other file-like objects
        if hasattr(audio_input, "read"):
            data = audio_input.read()
            target = save_path or Path(tempfile.mktemp(suffix=".wav"))
            with open(target, "wb") as f:
                f.write(data)
            return str(target)

        return None

    def _estimate_confidence(self, transcript: str) -> float:
        """Heuristic confidence based on transcript quality."""
        if not transcript or len(transcript.strip()) < 5:
            return 0.3
        issues = 0
        if len(transcript) < 20:
            issues += 1
        if " " not in transcript:
            issues += 1
        if any(c * 3 in transcript for c in "abcdefghijklmnopqrstuvwxyz"):
            issues += 1
        return max(0.0, min(1.0, 0.9 - issues * 0.15))

    def _postprocess_math_terms(self, transcript: str) -> str:
        """Convert spoken math words to symbols."""
        replacements = [
            ("x squared",       "x^2"),
            ("x cubed",         "x^3"),
            ("y squared",       "y^2"),
            ("squared",         "^2"),
            ("cubed",           "^3"),
            ("to the power of", "^"),
            ("raised to",       "^"),
            ("square root of",  "sqrt("),
            ("multiplied by",   "*"),
            ("divided by",      "/"),
            ("plus",            "+"),
            ("minus",           "-"),
            ("equals",          "="),
            ("times",           "*"),
        ]
        result = transcript
        for spoken, symbol in replacements:
            result = result.replace(spoken, symbol)
        return result

    @staticmethod
    def _unavailable_result(reason: str = "No transcription provider available.") -> Dict[str, Any]:
        return {
            "transcript": "",
            "confidence": 0.0,
            "audio_path": None,
            "hitl_required": True,
            "hitl_intervention": None,
            "language": "en",
            "error": reason
        }
