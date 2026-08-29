"""Verifier Agent - Verifies solution correctness"""
import json
import re
import ast
from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VerifierAgent(BaseAgent):
    """Verifies solution correctness and triggers HITL if needed"""

    def __init__(self):
        super().__init__(
            name="VerifierAgent",
            model=Config.VERIFIER_MODEL
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify solution correctness.

        For Arithmetic problems a deterministic SymPy/Python check is performed
        before any LLM call. For other topics the LLM verifier is used.

        Args:
            input_data: {problem_text, solution, steps, topic, subtopic}

        Returns:
            {is_correct, confidence, issues_found, verification_notes,
             checks_performed, hitl_required}
        """
        logger.info("Verifying solution...")

        problem_text = input_data.get("problem_text", "")
        solution = input_data.get("solution", "")
        steps = input_data.get("steps", [])
        topic = input_data.get("topic", "")
        subtopic = input_data.get("subtopic", "")

        # ── 1. Deterministic arithmetic verification ──────────────────────
        if topic == "Arithmetic":
            arith_result = self._verify_arithmetic(problem_text, solution)
            if arith_result is not None:
                confidence = arith_result["confidence"]
                arith_result["hitl_required"] = confidence < Config.VERIFIER_CONFIDENCE_THRESHOLD
                logger.info(
                    f"Arithmetic verification complete — "
                    f"Correct: {arith_result['is_correct']}, "
                    f"Confidence: {confidence:.2f}"
                )
                return arith_result

        # ── 2. Classification sanity check ────────────────────────────────
        classification_issues = self._check_classification(
            problem_text, solution, topic, subtopic
        )

        # ── 3. LLM verification ───────────────────────────────────────────
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(
            problem_text, solution, steps, topic, subtopic
        )

        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]

        response = self._call_llm(messages, max_tokens=1000)

        # Strip markdown code fences
        clean_response = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.MULTILINE)
        clean_response = re.sub(r'```\s*$', '', clean_response.strip(), flags=re.MULTILINE)

        try:
            json_start = clean_response.find("{")
            json_end = clean_response.rfind("}") + 1
            verification = json.loads(clean_response[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse verification response — using conservative fallback")
            verification = {
                "is_correct": False,
                "confidence": 0.5,
                "issues_found": ["Failed to verify solution"],
                "verification_notes": "Verification process encountered an error",
                "checks_performed": [],
            }

        # Merge in any classification issues
        if classification_issues:
            existing = verification.get("issues_found", [])
            verification["issues_found"] = classification_issues + existing
            verification["confidence"] = min(verification.get("confidence", 1.0), 0.7)

        confidence = verification.get("confidence", 0.5)
        verification["hitl_required"] = confidence < Config.VERIFIER_CONFIDENCE_THRESHOLD

        logger.info(
            f"Verification complete — Correct: {verification.get('is_correct')}, "
            f"Confidence: {confidence:.2f}, HITL: {verification.get('hitl_required')}"
        )

        return verification

    # ── Arithmetic deterministic check ────────────────────────────────────

    def _verify_arithmetic(self, problem_text: str, solution: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate the arithmetic expression from problem_text with Python's
        math engine and compare against the claimed answer in solution.

        Returns a verification dict or None if the expression can't be parsed.
        """
        try:
            # Extract numeric expression from problem text
            expr = self._extract_arithmetic_expr(problem_text)
            if expr is None:
                return None

            # Safe evaluation
            expected = self._safe_eval(expr)
            if expected is None:
                return None

            # Extract numeric answer from solution string
            claimed = self._extract_numeric_answer(solution)
            if claimed is None:
                return {
                    "is_correct": False,
                    "confidence": 0.6,
                    "issues_found": ["Could not extract a numeric answer from the solution"],
                    "verification_notes": f"Expected: {expected}",
                    "checks_performed": ["arithmetic_eval"],
                }

            is_correct = abs(float(expected) - float(claimed)) < 1e-9
            return {
                "is_correct": is_correct,
                "confidence": 0.99 if is_correct else 0.95,
                "issues_found": [] if is_correct else [
                    f"Incorrect answer: solution claims {claimed} but correct answer is {expected}"
                ],
                "verification_notes": (
                    f"Evaluated '{expr}' = {expected}. "
                    f"Solution claimed {claimed}. "
                    f"{'Match ✓' if is_correct else 'Mismatch ✗'}"
                ),
                "checks_performed": ["arithmetic_eval", "numeric_comparison"],
            }

        except Exception as exc:
            logger.warning(f"Arithmetic verification failed: {exc}")
            return None

    def _extract_arithmetic_expr(self, text: str) -> Optional[str]:
        """Extract a calculable numeric expression from problem text."""
        # Try: "25 + 37", "100 - 43", "7 * 8", "144 / 12", "25+37"
        pattern = r'[\d\s\+\-\*\/\(\)\.\%]+'
        text_clean = (
            text.replace("×", "*")
                .replace("÷", "/")
                .replace("plus", "+")
                .replace("minus", "-")
                .replace("times", "*")
                .replace("divided by", "/")
                .replace("add", "+")
                .replace("subtract", "-")
                .replace("multiply", "*")
                .replace("divide", "/")
        )
        matches = re.findall(pattern, text_clean)
        # Pick the longest match that contains at least one operator
        for m in sorted(matches, key=len, reverse=True):
            m = m.strip()
            if m and re.search(r'[\+\-\*\/]', m) and re.search(r'\d', m):
                return m
        return None

    @staticmethod
    def _safe_eval(expr: str):
        """Evaluate a numeric expression safely (no builtins, no names)."""
        try:
            # Only allow numeric literals and basic operators
            safe_expr = re.sub(r'[^\d\s\+\-\*\/\(\)\.]', '', expr)
            tree = ast.parse(safe_expr.strip(), mode='eval')
            # Whitelist only numeric operations
            allowed = (
                ast.Expression, ast.BinOp, ast.UnaryOp,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                ast.UAdd, ast.USub,
                ast.Constant,
            )
            for node in ast.walk(tree):
                if not isinstance(node, allowed):
                    return None
            result = eval(compile(tree, '<string>', 'eval'))  # noqa: S307 – sanitized above
            return result
        except Exception:
            return None

    @staticmethod
    def _extract_numeric_answer(text: str) -> Optional[float]:
        """Extract the final numeric answer from solution text."""
        # Common patterns: "= 62", "is 62", "answer: 62", "62", last number in text
        patterns = [
            r'=\s*([-\d]+(?:\.\d+)?)\s*$',
            r'(?:answer|result|equals?)[:\s]+([-\d]+(?:\.\d+)?)',
            r'\b([-\d]+(?:\.\d+)?)\s*$',
        ]
        for pat in patterns:
            m = re.search(pat, text.strip(), re.IGNORECASE | re.MULTILINE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

    # ── Classification sanity check ───────────────────────────────────────

    def _check_classification(
        self, problem_text: str, solution: str, topic: str, subtopic: str
    ):
        """Check for obvious mismatches between topic and solution content."""
        issues = []
        sol_lower = solution.lower()
        prob_lower = problem_text.lower()

        # If classified as Arithmetic but solution solves for a variable
        if topic == "Arithmetic":
            if re.search(r'\b[xyzabc]\s*=', sol_lower):
                issues.append(
                    "Solution appears to solve for a variable, but topic is Arithmetic. "
                    "Check classification."
                )

        # If classified as Algebra but solution is just a number
        if topic == "Algebra" and subtopic == "Linear Equations":
            if not re.search(r'[xyzabc]', prob_lower):
                issues.append(
                    "Problem has no variables but was classified as Algebra/Linear Equations."
                )

        return issues

    # ── LLM verification prompts ──────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        return """You are a math solution verifier. Your job is to:
1. Check if the solution is mathematically correct
2. Verify all steps are valid
3. Check for common mistakes
4. Validate units and domains
5. Test edge cases if applicable
6. Confirm the topic classification matches the problem content

Output ONLY valid JSON:
{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "issues_found": ["list", "of", "issues"],
  "verification_notes": "detailed notes on verification process",
  "checks_performed": ["list", "of", "verification", "checks"]
}

Common things to check:
- Hallucination: Does the solution use numbers/values NOT in the problem?
- Arithmetic errors and sign errors
- Domain restrictions (division by zero, sqrt of negative, etc.)
- Units consistency and boundary conditions
- Formula application and logical flow
- Topic/subtopic match (e.g. topic=Arithmetic but solution solves for x)

If confidence < 0.8, list specific concerns.
"""

    def _create_user_prompt(
        self,
        problem_text: str,
        solution: str,
        steps: list,
        topic: str,
        subtopic: str,
    ) -> str:
        steps_text = "\n\n".join(steps) if steps else "No detailed steps provided"

        return f"""Verify this solution:

**Problem** (Topic: {topic} | Subtopic: {subtopic}):
{problem_text}

**Proposed Solution**:
{solution}

**Solution Steps**:
{steps_text}

Provide verification JSON with your assessment."""
