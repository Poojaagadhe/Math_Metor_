
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.router_agent import RouterAgent

def test_router_agent():
    print("Testing RouterAgent task detection...")
    router = RouterAgent()
    
    test_cases = [
        ("f(x) = x**3. Find f'(x)", "derivative"),
        ("What is the derivative of x**2?", "derivative"),
        ("Calculate the integral of sin(x)", "integral"),
        ("integrate x**2", "integral"),
        ("x + 5 = 10", "solve"),
        ("Evaluate f(5) if f(x)=x**2", "solve"), # contains =
        ("Simplify x**2 + 2x**2", "simplify"),
    ]
    
    for text, expected in test_cases:
        result = router.detect_task(text)
        if result == expected:
            print(f"PASS: '{text}' -> {result}")
        else:
            print(f"FAIL: '{text}' -> {result} (expected {expected})")

if __name__ == "__main__":
    test_router_agent()
