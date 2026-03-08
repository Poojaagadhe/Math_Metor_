"""Image input processor with OCR capabilities"""

import io
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import easyocr

from utils.config import Config
from utils.logger import setup_logger
from utils.hitl import HITLManager, HITLTrigger

logger = setup_logger(__name__)


class ImageProcessor:
    """Handles image uploads and performs OCR extraction"""

    def __init__(self):
        """Initialize EasyOCR reader"""
        logger.info("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(Config.OCR_LANGUAGES, gpu=False)
        self.hitl_manager = HITLManager()
        logger.info("EasyOCR reader initialized")

    def process_image(
        self,
        image_input: Any,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Process an image and extract text"""

        logger.info("Processing image input...")
        image_path = None

        # Handle file path input
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
            image_path = str(image_input)

        # Handle raw bytes
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))

        # Handle Streamlit UploadedFile
        elif hasattr(image_input, "read"):
            image_bytes = image_input.read()
            image = Image.open(io.BytesIO(image_bytes))

        # Handle PIL Image
        elif isinstance(image_input, Image.Image):
            image = image_input

        else:
            raise ValueError("Unsupported image input type")

        # Improve image quality before OCR
        image = self.preprocess_image(image)

        # Save uploaded image if path provided
        if save_path:
            image.save(save_path)
            image_path = str(save_path)
            logger.info(f"Image saved to {save_path}")

        # Convert image to numpy format for OCR
        image_np = np.array(image)

        logger.info("Running OCR...")
        results = self.reader.readtext(image_np)

        extracted_lines = []
        confidences = []

        # Extract detected text and confidence
        for bbox, text, confidence in results:
            extracted_lines.append(text)
            confidences.append(confidence)

        extracted_text = " ".join(extracted_lines)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        logger.info(
            f"OCR finished. Segments: {len(extracted_lines)}, "
            f"Confidence: {avg_confidence:.2f}"
        )

        # Determine if human review is needed
        hitl_required = self.hitl_manager.should_trigger(
            trigger_type=HITLTrigger.LOW_OCR_CONFIDENCE,
            confidence=avg_confidence,
            threshold=Config.OCR_CONFIDENCE_THRESHOLD
        )

        hitl_intervention = None

        if hitl_required:
            hitl_intervention = self.hitl_manager.create_intervention(
                trigger_type=HITLTrigger.LOW_OCR_CONFIDENCE,
                message=f"OCR confidence is low ({avg_confidence:.2f})",
                data={
                    "extracted_text": extracted_text,
                    "confidence": avg_confidence,
                    "image_path": image_path
                },
                suggestions=[
                    "Review extracted text",
                    "Correct OCR errors",
                    "Check mathematical symbols"
                ]
            )

        # Return OCR results
        return {
            "extracted_text": extracted_text,
            "confidence": float(avg_confidence),
            "raw_results": results,
            "image_path": image_path,
            "hitl_required": hitl_required,
            "hitl_intervention": hitl_intervention,
            "num_segments": len(extracted_lines)
        }

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Basic preprocessing to improve OCR accuracy"""

        # Convert to grayscale
        image = image.convert("L")

        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        return image
