
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor

def test_two_pass_flow_logic():
    print("Testing Two-Pass OCR Flow Logic (Mocked)...")
    processor = MathOCRProcessor()
    
    # Mock models
    mock_easy = MagicMock()
    mock_pix = MagicMock()
    
    processor._easyocr_reader = mock_easy
    processor._pix2tex_model = mock_pix
    
    # CASE 1: Math detected in EasyOCR
    mock_easy.readtext.return_value = [
        ([], "Differentiate f(x) = sin(x)", 0.9)
    ]
    mock_pix.return_value = r"f(x) = \sin(x)"
    
    # Create dummy image
    dummy_img = Image.new('RGB', (100, 100))
    
    with patch.object(processor, 'preprocess_image', return_value=dummy_img):
        result = processor.process_image(dummy_img)
        print(f"CASE 1 (Math): {result['extracted_text']}")
        assert "Differentiate" in result['extracted_text']
        assert "sin(x)" in result['extracted_text']
        assert result['method'] == "pix2tex_two_pass"

    # CASE 2: No math detected
    mock_easy.readtext.return_value = [
        ([], "This is just a paragraph about something.", 0.9)
    ]
    
    with patch.object(processor, 'preprocess_image', return_value=dummy_img):
        result = processor.process_image(dummy_img)
        print(f"CASE 2 (No Math): {result['extracted_text']}")
        assert "paragraph" in result['extracted_text']
        assert result['method'] == "easyocr_fallback"

    print("Two-Pass logic tests PASSED!")

if __name__ == "__main__":
    test_two_pass_flow_logic()
