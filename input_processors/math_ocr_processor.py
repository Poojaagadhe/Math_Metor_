"""
Math OCR Processor
Uses Pix2Tex for equations and falls back to EasyOCR if unavailable.
Includes LaTeX cleaning for solver compatibility.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import re

from utils.logger import setup_logger

logger = setup_logger(__name__)

PIX2TEX_AVAILABLE = True


class MathOCRProcessor:

    def __init__(self):
        self._pix2tex_model = None
        self._easyocr_reader = None
        logger.info("MathOCRProcessor initialized (models will be lazy-loaded)")

    # --------------------------------------------------
    # Lazy Loaders
    # --------------------------------------------------

    @property
    def pix2tex_model(self):
        if self._pix2tex_model is None:
            self._pix2tex_model = self._get_cached_pix2tex()
        return self._pix2tex_model

    @property
    def easyocr_reader(self):
        if self._easyocr_reader is None:
            self._easyocr_reader = self._get_cached_easyocr()
        return self._easyocr_reader

    # --------------------------------------------------
    # Pix2Tex Weight Handling
    # --------------------------------------------------

    @classmethod
    def _ensure_pix2tex_weights(cls):

        import urllib.request

        weights_dir = Path("weights/pix2tex").absolute()
        weights_dir.mkdir(parents=True, exist_ok=True)

        weights_path = weights_dir / "weights.pth"
        resizer_path = weights_dir / "image_resizer.pth"

        base_url = "https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/"

        if not weights_path.exists():
            logger.info(f"Downloading Pix2Tex weights → {weights_path}")
            urllib.request.urlretrieve(base_url + "weights.pth", str(weights_path))

        if not resizer_path.exists():
            logger.info(f"Downloading Pix2Tex image resizer → {resizer_path}")
            urllib.request.urlretrieve(base_url + "image_resizer.pth", str(resizer_path))

        return weights_dir

    # --------------------------------------------------

    @classmethod
    def _get_cached_pix2tex(cls):

        try:
            import streamlit as st
            from munch import Munch

            @st.cache_resource(show_spinner="Initializing Pix2Tex OCR...")
            def load_pix2tex():

                from pix2tex.cli import LatexOCR

                weights_dir = cls._ensure_pix2tex_weights()

                args = Munch({
                    "config": "settings/config.yaml",
                    "checkpoint": str(weights_dir / "weights.pth"),
                    "no_cuda": True,
                    "no_resize": False
                })

                logger.info("Pix2Tex initialized with local weights")

                return LatexOCR(arguments=args)

            return load_pix2tex()

        except Exception as e:

            logger.warning(f"Pix2Tex cache loading failed: {e}")

            try:
                from pix2tex.cli import LatexOCR
                from munch import Munch

                weights_dir = cls._ensure_pix2tex_weights()

                args = Munch({
                    "config": "settings/config.yaml",
                    "checkpoint": str(weights_dir / "weights.pth"),
                    "no_cuda": True,
                    "no_resize": False
                })

                return LatexOCR(arguments=args)

            except ImportError:
                logger.warning("Pix2Tex not installed")
                return None

    # --------------------------------------------------

    @staticmethod
    def _get_cached_easyocr():

        try:
            import streamlit as st

            @st.cache_resource(show_spinner="Initializing EasyOCR...")
            def load_reader():
                import easyocr
                return easyocr.Reader(["en"], gpu=False)

            return load_reader()

        except Exception as e:
            import easyocr
            logger.warning(f"EasyOCR cache failed: {e}")
            return easyocr.Reader(["en"], gpu=False)

    # --------------------------------------------------
    # LaTeX Cleaning
    # --------------------------------------------------

    def _clean_latex(self, latex: str) -> str:
        """
        Remove formatting tokens and normalize LaTeX
        so it can be parsed by the solver.
        """

        if not latex:
            return latex

        # remove formatting tokens
        latex = re.sub(r'\\left|\\right', '', latex)
        latex = re.sub(r'\\bf', '', latex)
        latex = re.sub(r'\\mathrm\{.*?\}', '', latex)
        latex = re.sub(r'\\bar', '', latex)

        # remove stray commands
        latex = re.sub(r'\\[a-zA-Z]+', '', latex)

        # normalize exponent
        latex = latex.replace("^", "**")

        # remove extra whitespace
        latex = latex.strip()

        return latex

    # --------------------------------------------------
    # OCR Pipeline
    # --------------------------------------------------

    def process_image(self, image_input) -> Dict[str, Any]:

        image = self._load_image(image_input)

        # -----------------------------
        # Pix2Tex Extraction
        # -----------------------------

        if self.pix2tex_model:

            try:

                gray = image.convert("L")
                extrema = gray.getextrema()

                if extrema[0] != extrema[1]:

                    latex = self.pix2tex_model(image)

                    if latex:

                        clean_latex = self._clean_latex(latex)

                        return {
                            "method": "pix2tex",
                            "latex": clean_latex,
                            "extracted_text": clean_latex,
                            "raw_latex": latex,
                            "confidence": 1.0
                        }

                else:
                    logger.warning("Image appears empty, skipping Pix2Tex")

            except Exception as e:
                logger.warning(f"Pix2Tex failed: {e}")

        # -----------------------------
        # EasyOCR Fallback
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

        if hasattr(image_input, "read"):
            return Image.open(image_input)

        raise ValueError("Unsupported image input type")