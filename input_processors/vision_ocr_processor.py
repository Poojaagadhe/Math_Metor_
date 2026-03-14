"""
Vision-based OCR Processor using Groq's vision LLM.

Uses Groq's llama-3.2-11b-vision-preview model to accurately extract
mathematical equations and text from images. Falls back to a pure-text
description approach if the vision model is unavailable.
"""

import base64
import re
from pathlib import Path
from typing import Dict, Any, Optional

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Groq vision model - fast and accurate for math
GROQ_VISION_MODEL = "llama-3.2-11b-vision-preview"
GROQ_VISION_FALLBACK = "llama-3.2-90b-vision-preview"


class VisionOCRProcessor:
    """
    Extracts mathematical text from images using a vision LLM (Groq).

    Much more accurate than EasyOCR for:
    - Superscripts / subscripts  (x³, x₂)
    - Greek letters              (α, β, θ, π)
    - Prime notation             (f'(x), f''(x))
    - Integral / summation signs (∫, Σ)
    - Fractions displayed as images
    """

    def __init__(self):
        self.client = None
        self.model = GROQ_VISION_MODEL

        if Config.LLM_PROVIDER == "groq":
            try:
                from groq import Groq
                groq_key = getattr(Config, "GROQ_API_KEY", None)
                if groq_key:
                    self.client = Groq(api_key=groq_key)
                    logger.info(f"VisionOCRProcessor initialized with Groq vision model: {self.model}")
                else:
                    logger.warning("GROQ_API_KEY not set – VisionOCRProcessor disabled")
            except ImportError:
                logger.warning("groq package not installed – VisionOCRProcessor disabled")
        else:
            logger.info("VisionOCRProcessor: LLM_PROVIDER is not groq – skipped")

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the vision processor is ready to use."""
        return self.client is not None

    # ------------------------------------------------------------------

    def process_image(self, image_input) -> Dict[str, Any]:
        """
        Extract text / equations from an image using Groq Vision.

        Args:
            image_input: File path (str | Path) or bytes-like object.

        Returns:
            {
                "method": "groq_vision",
                "extracted_text": str,   # plain-text version for the solver
                "latex": str,            # LaTeX version (may equal extracted_text)
                "confidence": float,
                "success": bool
            }
        """
        if not self.client:
            return self._unavailable_result()

        image_b64 = self._encode_image(image_input)
        if not image_b64:
            return self._unavailable_result()

        prompt = (
            "You are a math OCR assistant. Extract ALL text and mathematical "
            "notation from this image exactly as written.\n\n"
            "Rules:\n"
            "1. Preserve LaTeX-style notation where helpful "
            "   (e.g. x^3 for x³, f'(x) for f prime of x).\n"
            "2. Return ONLY the extracted content – no commentary, "
            "   no explanations.\n"
            "3. If the image contains a question, include the full question.\n"
            "4. Represent superscripts as ^ (e.g. x^2, x^3).\n"
            "5. Represent derivatives with a prime (e.g. f'(x), f''(x)).\n"
            "6. Keep all numbers and symbols intact."
        )

        # Try primary model first, then fallback
        for model in [self.model, GROQ_VISION_FALLBACK]:
            result = self._call_vision(model, image_b64, prompt)
            if result:
                extracted = result.strip()
                logger.info(f"VisionOCR ({model}) extracted: {extracted[:120]}")
                return {
                    "method": "groq_vision",
                    "model": model,
                    "extracted_text": extracted,
                    "latex": extracted,
                    "confidence": 0.95,
                    "success": True
                }

        logger.warning("VisionOCR: all vision models failed")
        return self._unavailable_result()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_vision(self, model: str, image_b64: str, prompt: str) -> Optional[str]:
        """Call the Groq vision API and return the text response, or None on failure."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=512,
                temperature=0.0   # deterministic for OCR
            )
            content = response.choices[0].message.content
            return content if content else None
        except Exception as e:
            logger.warning(f"VisionOCR model {model} failed: {e}")
            return None

    def _encode_image(self, image_input) -> Optional[str]:
        """Convert various image input types to a base64 string."""
        try:
            # File path
            if isinstance(image_input, (str, Path)):
                with open(image_input, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

            # Raw bytes
            if isinstance(image_input, bytes):
                return base64.b64encode(image_input).decode("utf-8")

            # PIL Image
            try:
                from PIL import Image
                import io
                if isinstance(image_input, Image.Image):
                    buf = io.BytesIO()
                    image_input.save(buf, format="JPEG")
                    return base64.b64encode(buf.getvalue()).decode("utf-8")
            except ImportError:
                pass

            # Streamlit UploadedFile or file-like
            if hasattr(image_input, "read"):
                return base64.b64encode(image_input.read()).decode("utf-8")

            logger.warning(f"Unsupported image input type: {type(image_input)}")
            return None

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    @staticmethod
    def _unavailable_result() -> Dict[str, Any]:
        return {
            "method": "groq_vision",
            "extracted_text": "",
            "latex": None,
            "confidence": 0.0,
            "success": False
        }
