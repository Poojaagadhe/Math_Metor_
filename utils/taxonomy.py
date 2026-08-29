"""
Central Mathematical Taxonomy for Math Mentor.

All agents MUST import topic/subtopic names from here instead of using
raw strings. This guarantees consistency across the entire pipeline.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY: Dict[str, List[str]] = {
    "Arithmetic": [
        "Addition",
        "Subtraction",
        "Multiplication",
        "Division",
        "Fractions",
        "Percentages",
        "Order of Operations",
        "General",
    ],
    "Algebra": [
        "Linear Equations",
        "Quadratic Equations",
        "Systems of Equations",
        "Polynomials",
        "Inequalities",
        "Expressions",
        "Functions",
        "General",
    ],
    "Geometry": [
        "Area",
        "Perimeter",
        "Volume",
        "Triangles",
        "Circles",
        "Coordinate Geometry",
        "General",
    ],
    "Calculus": [
        "Derivatives",
        "Integrals",
        "Limits",
        "Differential Equations",
        "General",
    ],
    "Statistics & Probability": [
        "Mean/Median/Mode",
        "Probability",
        "Distributions",
        "Combinatorics",
        "General",
    ],
    "Linear Algebra": [
        "Matrices",
        "Vectors",
        "Eigenvalues",
        "Systems",
        "General",
    ],
    "Trigonometry": [
        "Sin/Cos/Tan",
        "Identities",
        "Inverse Trig",
        "General",
    ],
    "Number Theory": [
        "Primes",
        "GCD/LCM",
        "Modular Arithmetic",
        "General",
    ],
}

DIFFICULTY_LEVELS: List[str] = ["Easy", "Medium", "Hard"]

# Flat set of all canonical topics (lowercase → canonical) for fuzzy lookup
_TOPIC_MAP: Dict[str, str] = {t.lower(): t for t in TAXONOMY}

# Aliases: words the LLM or old code might produce → canonical topic
_TOPIC_ALIASES: Dict[str, str] = {
    "arithmetic": "Arithmetic",
    "basic arithmetic": "Arithmetic",
    "basic math": "Arithmetic",
    "number sense": "Arithmetic",
    "algebra": "Algebra",
    "algebraic": "Algebra",
    "calculus": "Calculus",
    "differential calculus": "Calculus",
    "integral calculus": "Calculus",
    "probability": "Statistics & Probability",
    "statistics": "Statistics & Probability",
    "stats": "Statistics & Probability",
    "probability and statistics": "Statistics & Probability",
    "linear algebra": "Linear Algebra",
    "matrix": "Linear Algebra",
    "matrices": "Linear Algebra",
    "geometry": "Geometry",
    "trigonometry": "Trigonometry",
    "trig": "Trigonometry",
    "number theory": "Number Theory",
    "unknown": "Algebra",  # graceful fallback
}

# Flat set of all canonical subtopics (lowercase → canonical)
_SUBTOPIC_MAP: Dict[str, str] = {}
for _subtopics in TAXONOMY.values():
    for _st in _subtopics:
        _SUBTOPIC_MAP[_st.lower()] = _st

_SUBTOPIC_ALIASES: Dict[str, str] = {
    "addition": "Addition",
    "subtraction": "Subtraction",
    "multiplication": "Multiplication",
    "division": "Division",
    "fractions": "Fractions",
    "fraction": "Fractions",
    "percentages": "Percentages",
    "percent": "Percentages",
    "order of operations": "Order of Operations",
    "pemdas": "Order of Operations",
    "bodmas": "Order of Operations",
    "linear equation": "Linear Equations",
    "linear equations": "Linear Equations",
    "quadratic": "Quadratic Equations",
    "quadratic equation": "Quadratic Equations",
    "quadratic equations": "Quadratic Equations",
    "system of equations": "Systems of Equations",
    "systems of equations": "Systems of Equations",
    "polynomial": "Polynomials",
    "polynomials": "Polynomials",
    "inequality": "Inequalities",
    "inequalities": "Inequalities",
    "expression": "Expressions",
    "expressions": "Expressions",
    "function": "Functions",
    "functions": "Functions",
    "derivative": "Derivatives",
    "derivatives": "Derivatives",
    "differentiation": "Derivatives",
    "integral": "Integrals",
    "integrals": "Integrals",
    "integration": "Integrals",
    "limit": "Limits",
    "limits": "Limits",
    "differential equation": "Differential Equations",
    "differential equations": "Differential Equations",
    "probability": "Probability",
    "mean": "Mean/Median/Mode",
    "median": "Mean/Median/Mode",
    "mode": "Mean/Median/Mode",
    "mean/median/mode": "Mean/Median/Mode",
    "distribution": "Distributions",
    "distributions": "Distributions",
    "combinatorics": "Combinatorics",
    "permutations": "Combinatorics",
    "combinations": "Combinatorics",
    "matrix": "Matrices",
    "matrices": "Matrices",
    "vector": "Vectors",
    "vectors": "Vectors",
    "eigenvalue": "Eigenvalues",
    "eigenvalues": "Eigenvalues",
    "area": "Area",
    "perimeter": "Perimeter",
    "volume": "Volume",
    "triangle": "Triangles",
    "triangles": "Triangles",
    "circle": "Circles",
    "circles": "Circles",
    "coordinate geometry": "Coordinate Geometry",
    "sin/cos/tan": "Sin/Cos/Tan",
    "identities": "Identities",
    "inverse trig": "Inverse Trig",
    "primes": "Primes",
    "prime": "Primes",
    "gcd": "GCD/LCM",
    "lcm": "GCD/LCM",
    "gcd/lcm": "GCD/LCM",
    "modular": "Modular Arithmetic",
    "modular arithmetic": "Modular Arithmetic",
    "general": "General",
    "unknown": "General",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonical_topic(raw: str) -> str:
    """Return the canonical topic name, falling back to 'Algebra'."""
    key = raw.strip().lower()
    return _TOPIC_ALIASES.get(key) or _TOPIC_MAP.get(key) or "Algebra"


def canonical_subtopic(raw: str) -> str:
    """Return the canonical subtopic name, falling back to 'General'."""
    key = raw.strip().lower()
    return _SUBTOPIC_ALIASES.get(key) or _SUBTOPIC_MAP.get(key) or "General"


def validate_classification(topic: str, subtopic: str) -> Tuple[str, str, bool]:
    """
    Validate and canonicalize a (topic, subtopic) pair.

    Returns:
        (canonical_topic, canonical_subtopic, is_valid)
        `is_valid` is True when both values were already correct.
    """
    ctopic = canonical_topic(topic)
    csubtopic = canonical_subtopic(subtopic)

    # Verify the subtopic actually belongs to the topic
    valid_subtopics = [s.lower() for s in TAXONOMY.get(ctopic, ["General"])]
    if csubtopic.lower() not in valid_subtopics:
        csubtopic = "General"

    was_valid = (ctopic == topic and csubtopic == subtopic)
    return ctopic, csubtopic, was_valid


def get_subtopics_for_topic(topic: str) -> List[str]:
    """Return the list of valid subtopics for a given (possibly raw) topic."""
    ctopic = canonical_topic(topic)
    return TAXONOMY.get(ctopic, ["General"])


def taxonomy_as_prompt_text() -> str:
    """Return a compact string representation of the taxonomy for LLM prompts."""
    lines = ["Available Topics and Subtopics (use EXACT names):"]
    for topic, subtopics in TAXONOMY.items():
        lines.append(f"  • {topic}: {', '.join(subtopics)}")
    return "\n".join(lines)
