
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor
from agents.solver_agent import SolverAgent

def test_ocr_normalization():
    print("Testing OCR normalization...")
    processor = MathOCRProcessor()
    
    # Test cases: input -> expected output
    test_cases = [
        ("x**2 + 1", "x^2 + 1"),
        ("y**(1/2)", "y^(1/2)"),
        ("3**x", "3^x"),
        ("x^2", "x^2"), # Already correct
    ]
    
    for inp, expected in test_cases:
        result = processor._clean_latex(inp)
        if result == expected:
            print(f"PASS: '{inp}' -> '{result}'")
        else:
            print(f"FAIL: '{inp}' -> '{result}' (expected '{expected}')")

def test_solver_normalization():
    print("\nTesting SolverAgent symbolic normalization...")
    solver = SolverAgent()
    
    # Case 1: Derivative (Matches: if "f'(x)" in text or "derivative" in text.lower())
    # And: expr_match = re.search(r"f\(x\)\s*=\s*([^,\?]+)", text)
    problem_text = "f(x) = x**3. Find f'(x)"
    result = solver._solve_symbolically(problem_text, {})
    if result and ("**" not in result) and ("^" in result):
        print(f"PASS: Derivative result '{result}' uses '^' and not '**'")
    else:
        print(f"FAIL: Derivative result '{result}' is incorrect")

    # Case 2: Function evaluation (Matches: pattern = r"f\((\w)\)\s*=\s*(.+)")
    # And: call_match = re.search(r"f\((\d+)\)", text)
    # The (.+) might be problematic if there's trailing text. 
    # Let's try a very clean input.
    problem_text = "f(x) = x**2 f(5)"
    result = solver._solve_symbolically(problem_text, {})
    if result and ("**" not in result):
        print(f"PASS: Function eval result '{result}' does not contain '**'")
    else:
        print(f"FAIL: Function eval result '{result}' is None or contains '**'")

if __name__ == "__main__":
    test_ocr_normalization()
    test_solver_normalization()
