"""
Math OCR Processor
Uses Pix2Tex for equations and falls back to EasyOCR if unavailable.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import easyocr

# Try importing Pix2Tex
try:
    from pix2tex.cli import LatexOCR
    PIX2TEX_AVAILABLE = True
except ImportError:
    PIX2TEX_AVAILABLE = False


class MathOCRProcessor:

    def __init__(self):

        self.pix2tex_model = None

        if PIX2TEX_AVAILABLE:
            try:
                self.pix2tex_model = LatexOCR()
            except Exception:
                self.pix2tex_model = None

        # EasyOCR fallback
        self.easyocr_reader = easyocr.Reader(["en"], gpu=False)

    # --------------------------------------------------

    def process_image(self, image_input) -> Dict[str, Any]:

        image = self._load_image(image_input)

        # -----------------------------
        # Try Pix2Tex (math equations)
        # -----------------------------

        if self.pix2tex_model:

            try:

                latex = self.pix2tex_model(image)

                return {
                    "method": "pix2tex",
                    "latex": latex,
                    "extracted_text": latex,
                    "confidence": 1.0
                }

            except Exception:
                pass

        # -----------------------------
        # Fallback: EasyOCR
        # -----------------------------

        image_np = np.array(image)

        results = self.easyocr_reader.readtext(image_np)

        texts = []
        confidences = []

        for _, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        extracted_text = " ".join(texts)

        avg_conf = np.mean(confidences) if confidences else 0

        return {
            "method": "easyocr",
            "latex": None,
            "extracted_text": extracted_text,
            "confidence": float(avg_conf),
            "segments": results
        }

    # --------------------------------------------------

    def _load_image(self, image_input):

        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, (str, Path)):
            return Image.open(image_input)

        # Streamlit UploadedFile
        if hasattr(image_input, "read"):
            return Image.open(image_input)

        raise ValueError("Unsupported image input type")