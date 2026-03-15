"""
Test script to verify OCR self-correction using the PlannerAgent.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.planner_agent import PlannerAgent

def run_test():
    print("Initializing PlannerAgent for OCR Self-Correction Test...")
    
    planner = PlannerAgent()
    
    bad_ocr = "f(x) = x^2 t 3x - 4" # should be + instead of t
    
    feedback = "The problem text 'f(x) = x^2 t 3x - 4' has a syntax error. 't' is not a valid operator here, so it cannot be parsed or solved."
    
    print(f"Original Bad OCR: {bad_ocr}")
    print(f"Feedback: {feedback}")
    print("\nRunning PlannerAgent correction...")
    
    corrected = planner.plan_ocr_correction(bad_ocr, feedback)
    
    print(f"\nCorrected Output: {corrected}")
    
    if ("+" in corrected or "*" in corrected) and "t" not in corrected:
        print("\n[Passed]: Successfully corrected 't' to a valid operator!")
    else:
        print("\n[Failed]: Correction did not fix the operator.")

if __name__ == "__main__":
    run_test()
