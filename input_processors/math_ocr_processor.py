"""
Math OCR Processor
Uses Pix2Tex for equations and falls back to EasyOCR if unavailable.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import easyocr

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Removed heavy Pix2Tex imports from top level
PIX2TEX_AVAILABLE = True # Assume available, check on load


class MathOCRProcessor:

    def __init__(self):
        self._pix2tex_model = None
        self._easyocr_reader = None
        logger.info("MathOCRProcessor initialized (models will be lazy-loaded)")

    @property
    def pix2tex_model(self):
        """Lazy-load and cache the Pix2Tex model."""
        if self._pix2tex_model is None:
            self._pix2tex_model = self._get_cached_pix2tex()
        return self._pix2tex_model

    @property
    def easyocr_reader(self):
        """Lazy-load and cache the EasyOCR reader."""
        if self._easyocr_reader is None:
            self._easyocr_reader = self._get_cached_easyocr()
        return self._easyocr_reader

    @staticmethod
    def _get_cached_pix2tex():
        """Get or create a cached Pix2Tex model."""
        try:
            import streamlit as st
            @st.cache_resource(show_spinner="Initializing Pix2Tex...")
            def load_pix2tex():
                from pix2tex.cli import LatexOCR
                logger.info("Initializing Pix2Tex model (cached)...")
                return LatexOCR()
            return load_pix2tex()
        except (ImportError, Exception) as e:
            try:
                from pix2tex.cli import LatexOCR
                logger.info(f"Initializing Pix2Tex (non-cached): {e}")
                return LatexOCR()
            except ImportError:
                logger.warning("Pix2Tex not installed")
                return None

    @staticmethod
    def _get_cached_easyocr():
        """Get or create a cached EasyOCR reader."""
        try:
            import streamlit as st
            @st.cache_resource(show_spinner="Initializing OCR engine...")
            def load_reader():
                import easyocr
                logger.info("Initializing EasyOCR reader (cached)...")
                return easyocr.Reader(["en"], gpu=False)
            return load_reader()
        except (ImportError, Exception) as e:
            import easyocr
            logger.info(f"Initializing EasyOCR reader (non-cached): {e}")
            return easyocr.Reader(["en"], gpu=False)

    # --------------------------------------------------

    def process_image(self, image_input) -> Dict[str, Any]:

        image = self._load_image(image_input)

        # -----------------------------
        # Try Pix2Tex (math equations)
        # -----------------------------

        if self.pix2tex_model:

            try:
                # Check if image is completely uniform (empty/pure white)
                # Pix2Tex's pad() function crashes with OpenCV error on empty images
                gray = image.convert('L')
                extrema = gray.getextrema()
                
                if extrema[0] != extrema[1]: # Not empty
                    latex = self.pix2tex_model(image)

                    if latex:
                        return {
                            "method": "pix2tex",
                            "latex": latex,
                            "extracted_text": latex,
                            "confidence": 1.0
                        }
                else:
                    logger.warning("MathOCRProcessor: Image is completely uniform, skipping Pix2Tex to avoid OpenCV crash.")

            except Exception as e:
                logger.warning(f"MathOCRProcessor: Pix2Tex extraction failed: {e}")
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