
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor

def test_math_detection():
    print("Testing Math Detection algorithm...")
    processor = MathOCRProcessor()
    
    test_cases = [
        ("f(x) = x**2", True),
        ("x + 5 = 10", True),
        ("Integral of sin(x)", True), # matches / or sin patterns if added? wait.
        ("3 * 4 / 2", True),
        ("Just some random sentence about math.", False),
        ("The value of x is 5.", False), # Simple sentence, no operators
        ("x^2", True), # Normalized later, but check if raw matches
    ]
    
    for text, expected in test_cases:
        # contains_math is used on cleaned text where ^ is already **
        cleaned = processor.clean_latex_output(text)
        result = processor.contains_math(cleaned)
        if result == expected:
            print(f"PASS: '{text}' -> {result}")
        else:
            print(f"FAIL: '{text}' -> {result} (expected {expected})")

if __name__ == "__main__":
    test_math_detection()
