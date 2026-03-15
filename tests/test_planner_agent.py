
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.planner_agent import PlannerAgent

def test_planner_agent():
    print("Testing PlannerAgent task-based routing...")
    planner = PlannerAgent()
    
    test_cases = [
        ("f(x) = x**3. Find f'(x)", "derivative", "calculus_tool"),
        ("integrate x**2", "integral", "calculus_tool"),
        ("x + 5 = 10", "solve", "sympy_tool"),
        ("Simplify x**2 + x", "simplify", "sympy_tool"),
        ("f(5) where f(x)=x**2", "solve", "function_tool"), # matches re.search(r"f\(\d+\)", text)
    ]
    
    for text, task, expected_tool in test_cases:
        result = planner._rule_based_plan(text, "unknown", task)
        strategy = result.get("strategy")
        if strategy == expected_tool:
            print(f"PASS: {task} with text '{text}' -> {strategy}")
        else:
            print(f"FAIL: {task} with text '{text}' -> {strategy} (expected {expected_tool})")

if __name__ == "__main__":
    test_planner_agent()
