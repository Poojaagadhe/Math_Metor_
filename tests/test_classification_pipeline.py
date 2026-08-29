"""
Regression tests for Math Mentor topic/subtopic classification pipeline.

These tests exercise the deterministic (rule-based) paths in ParserAgent and
RouterAgent -- NO LLM calls are required.

Run with:
    cd c:\\Users\\pooja\\Desktop\\math-mentor
    python -m pytest tests/test_classification_pipeline.py -v

Or directly:
    python tests/test_classification_pipeline.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.parser_agent import ParserAgent
from agents.router_agent import RouterAgent
from utils.taxonomy import canonical_topic, canonical_subtopic, validate_classification

# ---------------------------------------------------------------------------
# Test data
# Each entry: (input_text, expected_topic, expected_subtopic)
# ---------------------------------------------------------------------------

CLASSIFICATION_CASES = [
    # ── Arithmetic ─────────────────────────────────────────────────────────
    ("Calculate: 25 + 37",                          "Arithmetic",               "Addition"),
    ("100 - 43",                                    "Arithmetic",               "Subtraction"),
    ("7 * 8",                                       "Arithmetic",               "Multiplication"),
    ("144 / 12",                                    "Arithmetic",               "Division"),
    ("What is 15% of 200?",                         "Arithmetic",               "Percentages"),
    ("Simplify: (3 + 2) * 4 - 6 / 2",              "Arithmetic",               "Order of Operations"),

    # ── Algebra ────────────────────────────────────────────────────────────
    ("Solve: 2x + 3 = 7",                           "Algebra",                  "Linear Equations"),
    ("Solve: x^2 - 5x + 6 = 0",                    "Algebra",                  "Quadratic Equations"),
    ("Solve the system: 3x + 2y = 12, x - y = 1",  "Algebra",                  "Systems of Equations"),
    ("Simplify: x^2 + 2x^2",                        "Algebra",                  "Expressions"),
    ("Factor the polynomial: x^3 - 3x^2 + 2x",     "Algebra",                  "Polynomials"),

    # ── Calculus ───────────────────────────────────────────────────────────
    ("f(x) = x^3. Find f'(x)",                     "Calculus",                 "Derivatives"),
    ("What is the derivative of x^2?",             "Calculus",                 "Derivatives"),
    ("Calculate the integral of sin(x)",           "Calculus",                 "Integrals"),
    ("Find the limit as x approaches 0 of sin(x)/x", "Calculus",              "Limits"),

    # ── Statistics & Probability ───────────────────────────────────────────
    ("P(A) = 0.3. Find P(not A).",                  "Statistics & Probability", "Probability"),
    ("Find the mean of: 3, 7, 9, 12",              "Statistics & Probability", "Mean/Median/Mode"),

    # ── Geometry ───────────────────────────────────────────────────────────
    ("Find the area of a circle with radius 5",     "Geometry",                 "Circles"),
    ("Find the perimeter of a rectangle: 4 x 6",   "Geometry",                 "Perimeter"),

    # ── Linear Algebra ─────────────────────────────────────────────────────
    ("Find the determinant of the matrix [[1,2],[3,4]]", "Linear Algebra",     "Matrices"),
]

# ---------------------------------------------------------------------------
# Router task detection cases
# ---------------------------------------------------------------------------

TASK_CASES = [
    ("f(x) = x**3. Find f'(x)",    "derivative"),
    ("What is the derivative of x**2?", "derivative"),
    ("Calculate the integral of sin(x)", "integral"),
    ("integrate x**2",             "integral"),
    ("25 + 37",                    "arithmetic"),
    ("144 / 12",                   "arithmetic"),
    ("x + 5 = 10",                 "solve"),
    ("Evaluate f(5) if f(x)=x**2", "solve"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_classification_tests():
    parser = ParserAgent()
    passed = failed = 0
    failures = []

    print("\n" + "=" * 70)
    print("CLASSIFICATION TESTS (ParserAgent rule-based)")
    print("=" * 70)

    for text, exp_topic, exp_subtopic in CLASSIFICATION_CASES:
        result = parser.run({"text": text, "source": "text"})
        got_topic = result.get("topic", "")
        got_sub = result.get("subtopic", "")

        topic_ok = got_topic == exp_topic
        sub_ok = got_sub == exp_subtopic
        ok = topic_ok and sub_ok

        if ok:
            passed += 1
            print(f"  PASS  '{text[:55]:<55}'  ->  {got_topic} / {got_sub}")
        else:
            failed += 1
            msg = (
                f"  FAIL  '{text[:55]:<55}'\n"
                f"         Expected : {exp_topic} / {exp_subtopic}\n"
                f"         Got      : {got_topic} / {got_sub}"
            )
            print(msg)
            failures.append(msg)

    print(f"\nResult: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failures


def _run_task_detection_tests():
    router = RouterAgent()
    passed = failed = 0
    failures = []

    print("\n" + "=" * 70)
    print("TASK DETECTION TESTS (RouterAgent.detect_task)")
    print("=" * 70)

    for text, expected_task in TASK_CASES:
        got = router.detect_task(text)
        ok = got == expected_task

        if ok:
            passed += 1
            print(f"  PASS  '{text[:55]:<55}'  ->  {got}")
        else:
            failed += 1
            msg = (
                f"  FAIL  '{text[:55]:<55}'\n"
                f"         Expected : {expected_task}\n"
                f"         Got      : {got}"
            )
            print(msg)
            failures.append(msg)

    print(f"\nResult: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failures


def _run_taxonomy_tests():
    """Test the taxonomy validator."""
    cases = [
        ("arithmetic", "addition",           "Arithmetic", "Addition",    True),
        ("Arithmetic", "Addition",           "Arithmetic", "Addition",    True),
        ("algebra",    "linear equations",   "Algebra",    "Linear Equations", True),
        ("calculus",   "derivatives",        "Calculus",   "Derivatives", True),
        ("unknown",    "unknown",            "Algebra",    "General",     False),
        ("stats",      "probability",        "Statistics & Probability", "Probability", False),
    ]

    passed = failed = 0
    failures = []

    print("\n" + "=" * 70)
    print("TAXONOMY VALIDATOR TESTS")
    print("=" * 70)

    for raw_topic, raw_sub, exp_topic, exp_sub, exp_valid in cases:
        ct, cs, valid = validate_classification(raw_topic, raw_sub)
        ok = ct == exp_topic and cs == exp_sub

        if ok:
            passed += 1
            print(f"  PASS  ({raw_topic!r}, {raw_sub!r})  ->  {ct} / {cs}  valid={valid}")
        else:
            failed += 1
            msg = (
                f"  FAIL  ({raw_topic!r}, {raw_sub!r})\n"
                f"         Expected : {exp_topic} / {exp_sub}\n"
                f"         Got      : {ct} / {cs}"
            )
            print(msg)
            failures.append(msg)

    print(f"\nResult: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failures


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

def test_classification():
    failures = _run_classification_tests()
    assert not failures, "\n".join(failures)


def test_task_detection():
    failures = _run_task_detection_tests()
    assert not failures, "\n".join(failures)


def test_taxonomy():
    failures = _run_taxonomy_tests()
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_failures = []
    all_failures += _run_taxonomy_tests()
    all_failures += _run_task_detection_tests()
    all_failures += _run_classification_tests()

    print("\n" + "=" * 70)
    if all_failures:
        print(f"TOTAL FAILURES: {len(all_failures)}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED [OK]")
        sys.exit(0)
