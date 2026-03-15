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

    @classmethod
    def _ensure_pix2tex_weights(cls):
        """Downloads Pix2Tex weights to local data directory to avoid site-packages permission errors."""
        import os
        import urllib.request
        
        weights_dir = Path("data/weights/pix2tex").absolute()
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        weights_path = weights_dir / "weights.pth"
        resizer_path = weights_dir / "image_resizer.pth"
        
        base_url = "https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/"
        
        if not weights_path.exists():
            logger.info(f"Downloading Pix2Tex weights to {weights_path}...")
            try:
                urllib.request.urlretrieve(base_url + "weights.pth", str(weights_path))
            except Exception as e:
                logger.error(f"Failed to download weights.pth: {e}")
                
        if not resizer_path.exists():
            logger.info(f"Downloading Pix2Tex image_resizer to {resizer_path}...")
            try:
                urllib.request.urlretrieve(base_url + "image_resizer.pth", str(resizer_path))
            except Exception as e:
                logger.error(f"Failed to download image_resizer.pth: {e}")
        
        return weights_dir

    @classmethod
    def _get_cached_pix2tex(cls):
        """Get or create a cached Pix2Tex model."""
        try:
            import streamlit as st
            from munch import Munch
            @st.cache_resource(show_spinner="Initializing Pix2Tex...")
            def load_pix2tex():
                from pix2tex.cli import LatexOCR
                
                weights_dir = cls._ensure_pix2tex_weights()
                args = Munch({
                    'config': 'settings/config.yaml',
                    'checkpoint': str(weights_dir / 'weights.pth'),
                    'no_cuda': True,
                    'no_resize': False
                })
                
                logger.info("Initializing Pix2Tex model (cached) with local weights...")
                return LatexOCR(arguments=args)
            return load_pix2tex()
        except (ImportError, Exception) as e:
            try:
                from pix2tex.cli import LatexOCR
                from munch import Munch
                
                weights_dir = cls._ensure_pix2tex_weights()
                args = Munch({
                    'config': 'settings/config.yaml',
                    'checkpoint': str(weights_dir / 'weights.pth'),
                    'no_cuda': True,
                    'no_resize': False
                })
                
                logger.info(f"Initializing Pix2Tex (non-cached) with local weights: {e}")
                return LatexOCR(arguments=args)
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