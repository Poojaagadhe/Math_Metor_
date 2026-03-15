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

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image for better OCR: grayscale, contrast, and thresholding.
        """
        from PIL import ImageEnhance, ImageOps

        # 1. Grayscale
        processed = image.convert("L")

        # 2. Contrast Enhancement
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(2.0)

        # 3. Thresholding (Binary)
        # Using a simple adaptive-like thresholding via point lookup
        threshold = 128
        processed = processed.point(lambda p: 255 if p > threshold else 0)
        
        # 4. Optional: Invert if background is dark
        # (Assuming light background for most math problems)
        
        return processed

    def _clean_latex(self, latex: str) -> str:
        """
        Original internal cleaner for solver compatibility.
        """
        if not latex:
            return latex

        # remove common font styles/formatting
        latex = re.sub(r'\\left|\\right', '', latex)
        latex = re.sub(r'\\bf', '', latex)
        latex = re.sub(r'\\mathrm\{.*?\}', '', latex)
        latex = re.sub(r'\\mathit\{.*?\}', '', latex)
        latex = re.sub(r'\\mathbf\{.*?\}', '', latex)
        latex = re.sub(r'\\text\{.*?\}', '', latex)
        latex = re.sub(r'\\bar', '', latex)
        latex = re.sub(r'\\,', ' ', latex)
        latex = re.sub(r'\\:', ' ', latex)
        latex = re.sub(r'\\;', ' ', latex)
        latex = re.sub(r'\\quad', ' ', latex)
        
        # normalize exponents to ^
        latex = latex.replace('**', '^')

        # normalize spaces
        latex = latex.replace('\n', ' ')
        latex = re.sub(r'\s+', ' ', latex)

        return latex.strip()

    def clean_latex_output(self, text: str) -> str:
        """
        Converts LaTeX into human-readable text and normalizes for SymPy.
        Also removes question numbers and preserves basic math structure.
        """
        if not text:
            return text

        # 1. Remove Question Numbers (e.g., "1.", "Q1.", "Question 1.")
        text = re.sub(r"^\d+\.", "", text)
        text = re.sub(r"^Q\d+\.", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(Question|q)\s*\d+[:\.]?", "", text, flags=re.IGNORECASE)

        # 2. Handle common LaTeX math structures before removing backslashes
        # Handle \frac{a}{b} -> a/b
        text = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'\1/\2', text)
        
        # Preserve common functions (sin, cos, tan, log, ln, sqrt, lim, int) by removing backslash only
        functions = ["sin", "cos", "tan", "log", "ln", "sqrt", "lim", "int"]
        for func in functions:
            text = text.replace(f"\\{func}", func)

        # 3. Remove LaTeX environments
        text = re.sub(r'\\begin\{.*?\}', '', text)
        text = re.sub(r'\\end\{.*?\}', '', text)

        # 4. Remove remaining LaTeX commands (backslashed words)
        text = re.sub(r'\\[a-zA-Z]+', '', text)

        # 5. Remove curly braces
        text = text.replace("{", "")
        text = text.replace("}", "")

        # 6. SymPy Normalization: replace ^ with **
        text = text.replace("^", "**")

        # Clean spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def contains_math(self, text: str) -> bool:
        """
        Detects if the text contains mathematical patterns.
        """
        patterns = [
            r"x\^",
            r"\d+\^",
            r"x\*\*",
            r"\d+\*\*",
            r"[+\-*/=]", # Operators
            r"[a-z]\([a-z]\)", # Function notation
            r"∫",
            r"√",
            r"\\frac", # LaTeX remnants in context
            r"sum|log|sin|cos|tan" # Keywords
        ]

        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True

        return False

    # --------------------------------------------------
    # OCR Pipeline
    # --------------------------------------------------

    def process_image(self, image_input) -> Dict[str, Any]:
        """
        Two-Pass OCR Strategy:
        1. Preprocess + EasyOCR for context.
        2. If math detected: Pix2Tex for precision.
        3. Merge results.
        """
        image = self._load_image(image_input)
        preprocessed_image = self.preprocess_image(image)

        # -----------------------------
        # PASS 1: EasyOCR for context
        # -----------------------------
        logger.info("Pass 1: Running EasyOCR for context...")
        image_np = np.array(preprocessed_image)
        easy_results = self.easyocr_reader.readtext(image_np)
        
        easy_text = " ".join([r[1] for r in easy_results])
        avg_conf = np.mean([r[2] for r in easy_results]) if easy_results else 0.0

        # -----------------------------
        # Detect Math Presence
        # -----------------------------
        # We use a broad check to see if it's worth running Pix2Tex
        is_math = self.contains_math(easy_text)
        
        if is_math and self.pix2tex_model:
            # -----------------------------
            # PASS 2: Pix2Tex for precision
            # -----------------------------
            logger.info("Pass 2: Math detected. Running Pix2Tex for precision...")
            try:
                # Run Pix2Tex on the original image (it handles its own normalization)
                raw_latex = self.pix2tex_model(image)
                
                if raw_latex:
                    cleaned_expression = self.clean_latex_output(raw_latex)
                    
                    # Merge Strategy: 
                    # If EasyOCR detected instructions (Differentiate, Solve, etc.), 
                    # we try to keep them if they are at the start.
                    instruction_match = re.search(r"^(Differentiate|Solve|Integrate|Evaluate|Simplify|Find)\b", easy_text, re.IGNORECASE)
                    
                    if instruction_match:
                        instruction = instruction_match.group(0)
                        final_text = f"{instruction} {cleaned_expression}"
                    else:
                        final_text = cleaned_expression

                    return {
                        "method": "pix2tex_two_pass",
                        "latex": raw_latex,
                        "extracted_text": final_text,
                        "instruction": instruction_match.group(0) if instruction_match else None,
                        "expression": cleaned_expression,
                        "confidence": 1.0
                    }
            except Exception as e:
                logger.warning(f"Pix2Tex failed in two-pass: {e}")

        # -----------------------------
        # FALLBACK: Return EasyOCR text
        # -----------------------------
        logger.info("Returning EasyOCR fallback result.")
        return {
            "method": "easyocr_fallback",
            "latex": None,
            "extracted_text": easy_text,
            "confidence": float(avg_conf)
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