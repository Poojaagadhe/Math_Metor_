
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor

def test_latex_cleaner_v2():
    print("Testing Updated LaTeX Cleaner with Question Number Removal...")
    processor = MathOCRProcessor()
    
    test_cases = [
        (r"1. \begin{equation}x^3 + 1\end{equation}", "x**3 + 1"),
        (r"Q1. \frac{1}{2}x^2", "1/2x**2"),
        (r"Question 5: f(x) = x^{2}", "f(x) = x**2"),
        (r"q10. 2x + 3 = 7", "2x + 3 = 7"),
    ]
    
    for text, expected in test_cases:
        result = processor.clean_latex_output(text)
        if result == expected:
            print(f"PASS: '{text[:20]}...' -> '{result}'")
        else:
            print(f"FAIL: '{text[:20]}...' -> '{result}' (expected '{expected}')")

if __name__ == "__main__":
    test_latex_cleaner_v2()
