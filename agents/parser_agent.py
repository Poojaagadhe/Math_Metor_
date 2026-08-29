"""Parser Agent - Converts raw input into structured problem format"""

import json
import re
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from utils.config import Config
from utils.logger import setup_logger
from utils.taxonomy import (
    canonical_topic,
    canonical_subtopic,
    validate_classification,
    taxonomy_as_prompt_text,
)

logger = setup_logger(__name__)


class ParserAgent(BaseAgent):
    """Parses raw input and creates structured math problem representation"""

    def __init__(self):
        super().__init__(
            name="ParserAgent",
            model=Config.PARSER_MODEL
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:

        logger.info("Running ParserAgent")

        raw_text = input_data.get("text", "")
        source = input_data.get("source", "text")

        cleaned_text = self._clean_text(raw_text)

        # ── 1. Deterministic classification (no LLM needed) ──────────────
        topic = self._detect_topic(cleaned_text)
        subtopic = self._detect_subtopic(cleaned_text, topic)
        difficulty = self._detect_difficulty(cleaned_text, topic)

        # Canonicalize via taxonomy
        topic, subtopic, _ = validate_classification(topic, subtopic)

        variables = self._extract_variables(cleaned_text, topic)
        equations = self._extract_equations(cleaned_text)

        # ── 2. LLM fallback only for ambiguous non-arithmetic problems ────
        if topic == "Algebra" and subtopic == "General" and not equations:
            parsed = self._llm_parse(cleaned_text, source)
            # Canonicalize whatever the LLM returned
            raw_topic = parsed.get("topic", "Algebra")
            raw_sub = parsed.get("subtopic", "General")
            parsed["topic"], parsed["subtopic"], _ = validate_classification(
                raw_topic, raw_sub
            )
        else:
            parsed = {
                "problem_text": cleaned_text,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty,
                "variables": variables,
                "constraints": equations,
                "needs_clarification": False,
                "clarification_needed": None,
            }

        # Ensure difficulty is always present
        if "difficulty" not in parsed or not parsed["difficulty"]:
            parsed["difficulty"] = difficulty

        parsed["raw_input"] = raw_text
        parsed["source"] = source
        parsed["parser_confidence"] = 1.0 if not parsed.get("needs_clarification") else 0.5

        logger.info(
            f"Parsed problem | Topic: {parsed.get('topic')} | "
            f"Subtopic: {parsed.get('subtopic')} | "
            f"Difficulty: {parsed.get('difficulty')} | "
            f"Variables: {parsed.get('variables')}"
        )

        return parsed

    # ── Text cleaning ────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Fix common OCR errors and normalize math symbols"""

        # Handle primes (OCR often turns ' into `, ., unicode primes, or curly quotes)
        text = text.replace("\u2019", "'").replace("\u2032", "'").replace("\u2035", "'").replace("`", "'")
        text = text.replace("f.(x)", "f'(x)")
        text = text.replace("f(z)", "f(x)")  # Often misread when x has a prime next to it

        # Normalize characters
        text = text.replace("\u2212", "-")
        text = text.replace("\u00d7", "*")
        text = text.replace("\u2022", "*")

        # Fix OCR dropped exponents: e.g. x2 -> x^2, 2x2 -> 2x^2, x3 -> x^3
        # Avoid matching words like 'dx', 'dy'
        text = re.sub(r'(?<![a-zA-Z])([xyzXYZ])(\d+)', r'\1^\2', text)
        text = re.sub(r'(\d+)([xyzXYZ])(\d+)', r'\1\2^\3', text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ── Topic detection ──────────────────────────────────────────────────

    def _detect_topic(self, text: str) -> str:
        """
        Hierarchical rule-based topic classifier.

        Order matters: more specific checks first, generic fallback last.
        """
        t = text.lower().strip()

        # ── Calculus ────────────────────────────────────────────────────
        calculus_keywords = [
            "derivative", "differentiate", "d/dx", "dy/dx", "df/dx",
            "integral", "integrate", "∫", "limit", "lim(", "lim ",
            "differential equation",
        ]
        if any(kw in t for kw in calculus_keywords):
            return "Calculus"
        # Prime notation: f'(x), y', g'
        if "'" in t and ("f(" in t or "g(" in t or "y" in t):
            return "Calculus"

        # ── Linear Algebra ───────────────────────────────────────────────
        if any(kw in t for kw in ["matrix", "matrices", "vector", "eigenvalu", "eigenvect", "determinant"]):
            return "Linear Algebra"

        # ── Statistics & Probability ─────────────────────────────────────
        if any(kw in t for kw in [
            "probability", "p(", "p (", "stat", "mean", "median", "mode",
            "standard deviation", "variance", "permutation", "combination",
            "factorial", "distribution", "expected value",
        ]):
            return "Statistics & Probability"

        # ── Trigonometry ─────────────────────────────────────────────────
        trig_keywords = ["sin", "cos", "tan", "cot", "sec", "csc",
                         "arcsin", "arccos", "arctan"]
        if any(re.search(r'\b' + kw + r'\b', t) for kw in trig_keywords):
            return "Trigonometry"

        # ── Geometry ─────────────────────────────────────────────────────
        if any(kw in t for kw in [
            "area", "perimeter", "volume", "circle", "triangle", "rectangle",
            "polygon", "angle", "radius", "diameter", "hypotenuse", "pythagor",
        ]):
            return "Geometry"

        # ── Arithmetic ───────────────────────────────────────────────────
        # Heuristic: strip known math-operation words and check if only nums/operators remain
        # This catches "Simplify: (3 + 2) * 4 - 6 / 2" where all tokens are numeric
        pemdas_check = re.sub(r'\b(simplify|calculate|compute|evaluate|what is)\b', '', t)
        pemdas_check = re.sub(r'[\s\+\-\*\/\(\)\.\,\%\:]', '', pemdas_check)
        if pemdas_check.isdigit() and pemdas_check:  # only digits left -> pure numeric
            return "Arithmetic"

        # Check: is the problem a pure numeric expression (no alphabetic vars)?
        # Strip math operators/spaces and check if only digits and operators remain.
        stripped = re.sub(r'[\s\+\-\*\/\(\)\.\,\%]', '', t)
        if stripped.isdigit():
            return "Arithmetic"

        # Arithmetic keyword phrases
        arithmetic_keywords = [
            "add", "added", "addition", "sum of", "plus",
            "subtract", "subtraction", "difference", "minus",
            "multiply", "multiplication", "product of", "times",
            "divide", "division", "quotient",
            "fraction", "percent", "percentage",
            "how much", "how many", "what is",
            "calculate", "compute", "evaluate",
        ]
        # Only classify as Arithmetic if there are no algebraic variable letters
        # Note: strip 'x' that appears as a dimension separator ("4 x 6") by
        # requiring 'x' to be preceded/followed by a letter or math context.
        text_no_dim_x = re.sub(r'(?<=\d)\s*[xX]\s*(?=\d)', ' * ', t)  # "4 x 6" -> "4 * 6"
        has_variable = bool(
            re.search(r'\b[a-wyz]\b', text_no_dim_x) or  # single letter variable (not x)
            re.search(r'\bx\b(?!\s*\d)', text_no_dim_x)   # x not used as multiplier
        )
        if any(kw in t for kw in arithmetic_keywords) and not has_variable:
            return "Arithmetic"

        # Pure digit-only expression with operators (catches "25 + 37", "3 * 4")
        pure_numeric = re.fullmatch(r'[\d\s\+\-\*\/\(\)\.\%\,]+', t.strip())
        if pure_numeric:
            return "Arithmetic"

        # ── Number Theory ────────────────────────────────────────────────
        if any(kw in t for kw in ["prime", "gcd", "lcm", "modulo", "mod ", "divisib", "factor of"]):
            return "Number Theory"

        # ── Algebra (catch-all for expressions with variables) ───────────
        return "Algebra"

    # ── Subtopic detection ───────────────────────────────────────────────

    def _detect_subtopic(self, text: str, topic: str) -> str:
        """Infer the subtopic given the detected topic."""
        t = text.lower()

        if topic == "Arithmetic":
            # Order of operations wins if multiple different operators are present
            t_ops = re.findall(r'[\+\-\*\/]', t)
            has_multiple_op_types = len(set(t_ops)) >= 2
            if has_multiple_op_types or any(kw in t for kw in ["order of operations", "pemdas", "bodmas", "brackets"]):
                return "Order of Operations"
            if any(kw in t for kw in ["+", "add", "sum", "plus", "total"]):
                return "Addition"
            if any(kw in t for kw in ["-", "subtract", "minus", "difference", "less"]):
                return "Subtraction"
            if any(kw in t for kw in ["*", "×", "multiply", "product", "times"]):
                return "Multiplication"
            if any(kw in t for kw in ["/", "÷", "divide", "quotient"]):
                return "Division"
            if any(kw in t for kw in ["fraction", "numerator", "denominator"]):
                return "Fractions"
            if any(kw in t for kw in ["percent", "%"]):
                return "Percentages"
            return "General"

        if topic == "Algebra":
            # Check polynomial/factor BEFORE quadratic so "x^3" stays polynomial
            if any(kw in t for kw in ["polynomial", "degree", "leading coefficient", "factor"]):
                return "Polynomials"
            # 'simplify' with x^2 means simplifying an expression, not solving quadratic
            if any(kw in t for kw in ["simplif", "expand"]):
                return "Expressions"
            if any(kw in t for kw in ["quadratic", "discriminant"]) or re.search(r'x\^2', t):
                return "Quadratic Equations"
            # Detect algebraic variables (x,y,z,a,b,c,n,m) in the expression
            # Use pattern: letter as standalone or attached to digit (e.g. 2x, 3y)
            has_var_in_eq = bool(re.search(r'(?<!\w)([xyzabcnm])(?!\w)|(\d[xyzabcnm])', t) and "=" in t)
            if has_var_in_eq:
                if "system" in t or (t.count("=") >= 2 and re.search(r'[xy]', t)):
                    return "Systems of Equations"
                return "Linear Equations"
            if any(kw in t for kw in ["inequality", "inequalit", ">", "<", "\u2264", "\u2265"]):
                return "Inequalities"
            if any(kw in t for kw in ["expression"]):
                return "Expressions"
            if any(kw in t for kw in ["f(x)", "g(x)", "function"]):
                return "Functions"
            return "General"

        if topic == "Calculus":
            if any(kw in t for kw in ["derivative", "differentiate", "d/dx", "dy/dx", "f'", "f'"]):
                return "Derivatives"
            if any(kw in t for kw in ["integral", "integrate", "∫", "antiderivative"]):
                return "Integrals"
            if any(kw in t for kw in ["limit", "lim"]):
                return "Limits"
            if "differential equation" in t:
                return "Differential Equations"
            return "General"

        if topic == "Statistics & Probability":
            if any(kw in t for kw in ["probability", "p(", "chance", "likelihood"]):
                return "Probability"
            if any(kw in t for kw in ["mean", "average", "median", "mode"]):
                return "Mean/Median/Mode"
            if any(kw in t for kw in ["distribution", "normal", "binomial", "poisson"]):
                return "Distributions"
            if any(kw in t for kw in ["combination", "permutation", "factorial", "choose"]):
                return "Combinatorics"
            return "General"

        if topic == "Geometry":
            # Check more specific shapes BEFORE generic area/perimeter/volume
            if any(kw in t for kw in ["circle", "radius", "diameter", "circumference"]):
                return "Circles"
            if any(kw in t for kw in ["triangle", "pythagor", "hypotenuse"]):
                return "Triangles"
            if any(kw in t for kw in ["coordinate", "slope", "intercept", "midpoint", "distance"]):
                return "Coordinate Geometry"
            if any(kw in t for kw in ["area"]):
                return "Area"
            if any(kw in t for kw in ["perimeter"]):
                return "Perimeter"
            if any(kw in t for kw in ["volume"]):
                return "Volume"
            return "General"

        if topic == "Linear Algebra":
            if any(kw in t for kw in ["matrix", "matrices", "determinant"]):
                return "Matrices"
            if any(kw in t for kw in ["vector"]):
                return "Vectors"
            if any(kw in t for kw in ["eigenvalue", "eigenvector"]):
                return "Eigenvalues"
            return "General"

        if topic == "Trigonometry":
            if any(kw in t for kw in ["identity", "identities"]):
                return "Identities"
            if any(kw in t for kw in ["arcsin", "arccos", "arctan", "inverse"]):
                return "Inverse Trig"
            return "Sin/Cos/Tan"

        if topic == "Number Theory":
            if any(kw in t for kw in ["prime"]):
                return "Primes"
            if any(kw in t for kw in ["gcd", "lcm", "greatest common", "lowest common"]):
                return "GCD/LCM"
            if any(kw in t for kw in ["mod", "modulo", "modular"]):
                return "Modular Arithmetic"
            return "General"

        return "General"

    # ── Difficulty detection ─────────────────────────────────────────────

    def _detect_difficulty(self, text: str, topic: str) -> str:
        """Heuristic difficulty classification."""
        if topic == "Arithmetic":
            return "Easy"
        if topic in ("Calculus", "Linear Algebra"):
            return "Hard"
        t = text.lower()
        hard_signals = ["system of", "quadratic", "polynomial", "eigenvalue",
                        "differential equation", "double integral", "triple"]
        if any(kw in t for kw in hard_signals):
            return "Hard"
        return "Medium"

    # ── Variable extraction ──────────────────────────────────────────────

    def _extract_variables(self, text: str, topic: str) -> List[str]:
        """Extract variable names; skip for pure Arithmetic problems."""
        if topic == "Arithmetic":
            return []

        # Exclude common non-variable letters
        non_vars = {"d", "e", "i"}  # d (as in dx), e (Euler), i (imaginary)
        vars_found = re.findall(r"\b([a-zA-Z])\b", text)
        return sorted(list(set(v for v in vars_found if v.lower() not in non_vars)))

    # ── Equation extraction ──────────────────────────────────────────────

    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text"""
        matches = re.findall(
            r"[a-zA-Z0-9\^\+\-\*/\(\)\s]+=[a-zA-Z0-9\^\+\-\*/\(\)\s]+",
            text
        )
        return matches

    # ── LLM fallback ─────────────────────────────────────────────────────

    def _llm_parse(self, text: str, source: str) -> Dict[str, Any]:
        """Fallback LLM parsing for ambiguous problems"""

        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(text, source)

        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]

        response = self._call_llm(messages, max_tokens=800)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            parsed = json.loads(response[json_start:json_end])

        except Exception:
            logger.warning("LLM JSON parse failed. Using fallback parser.")
            parsed = {
                "problem_text": text,
                "topic": "Algebra",
                "subtopic": "General",
                "difficulty": "Medium",
                "variables": [],
                "constraints": [],
                "needs_clarification": True,
                "clarification_needed": "Unable to parse problem structure",
            }

        return parsed

    # ── Prompts ──────────────────────────────────────────────────────────

    def _get_system_prompt(self) -> str:

        taxonomy_text = taxonomy_as_prompt_text()

        return f"""You are a mathematical problem parser.

Your task:
1. Normalize the problem statement
2. Identify topic and subtopic using ONLY the taxonomy below
3. Estimate difficulty (Easy / Medium / Hard)
4. Extract variables and constraints

{taxonomy_text}

Return ONLY valid JSON in this format:

{{
  "problem_text": "...",
  "topic": "<exact topic from taxonomy>",
  "subtopic": "<exact subtopic from taxonomy>",
  "difficulty": "Easy|Medium|Hard",
  "variables": ["..."],
  "constraints": ["..."],
  "needs_clarification": false,
  "clarification_needed": null
}}

CRITICAL:
- Use EXACT topic and subtopic names from the taxonomy above.
- DO NOT use "algebra" as a default for numeric calculations — use "Arithmetic".
- DO NOT invent numbers, variables, or constraints not in the provided text.
- ONLY output the JSON.
"""

    def _create_user_prompt(self, text: str, source: str) -> str:

        return f"""Parse the following math problem (source: {source}):

{text}

Return structured JSON using the exact taxonomy terms.
"""