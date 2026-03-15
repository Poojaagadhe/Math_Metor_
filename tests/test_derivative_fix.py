
import sys
from pathlib import Path
import sympy as sp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.parser_agent import ParserAgent
from agents.solver_agent import SolverAgent

def test_derivative_parsing():
    print("Testing ParserAgent with derivative notation...")
    parser = ParserAgent()
    
    test_cases = [
        "If f(x) = x^3, what is f'(x)?",
        "differentiate f(x) = x3",
        "find the derivative of f(x) = sin(x)",
        "f'(x) where f(x) = x^2 + 5x + 6"
    ]
    
    for text in test_cases:
        result = parser.run({"text": text, "source": "text"})
        print(f"Input: {text}")
        print(f"Topic: {result.get('topic')}")
        print(f"Cleaned: {result.get('problem_text')}")
        print("-" * 20)
        assert result.get('topic') == 'calculus', f"Failed to detect calculus for: {text}"

def test_derivative_solving():
    print("\nTesting SolverAgent symbolic derivative logic...")
    solver = SolverAgent()
    parser = ParserAgent()
    
    test_cases = [
        "If f(x) = x^3, what is f'(x)?",
        "f(x) = x^2, find f'(x)",
        "derivative of f(x) = x^4"
    ]
    
    for text in test_cases:
        parsed = parser.run({"text": text, "source": "text"})
        result = solver.run({
            "problem_text": parsed.get("problem_text"),
            "topic": parsed.get("topic"),
            "parsed_data": parsed
        })
        print(f"Input: {text}")
        print(f"Solution: {result.get('solution')}")
        print("-" * 20)
        assert "3*x**2" in result.get('solution') or "2*x" in result.get('solution') or "4*x**3" in result.get('solution'), "Failed to solve derivative"

if __name__ == "__main__":
    try:
        test_derivative_parsing()
        test_derivative_solving()
        print("\n✅ All derivative logic tests passed!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
