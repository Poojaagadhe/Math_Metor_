"""Parser Agent - Converts raw input into structured problem format"""

import json
import re
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from utils.config import Config
from utils.logger import setup_logger

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

        # Attempt deterministic parsing first
        variables = self._extract_variables(cleaned_text)
        equations = self._extract_equations(cleaned_text)
        topic = self._detect_topic(cleaned_text)

        # If structured detection succeeded, skip LLM
        if equations or variables:

            parsed = {
                "problem_text": cleaned_text,
                "topic": topic,
                "subtopic": "general",
                "variables": variables,
                "constraints": equations,
                "needs_clarification": False,
                "clarification_needed": None
            }

        else:

            parsed = self._llm_parse(cleaned_text, source)

        parsed["raw_input"] = raw_text
        parsed["source"] = source

        parsed["parser_confidence"] = 1.0 if not parsed.get("needs_clarification") else 0.5

        logger.info(
            f"Parsed problem | Topic: {parsed.get('topic')} | Variables: {parsed.get('variables')}"
        )

        return parsed

    # -------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Fix common OCR errors and normalize math symbols"""

        # Handle primes (OCR often turns ' into `, ., unicode primes, or curly quotes)
        text = text.replace("’", "'").replace("′", "'").replace("‵", "'").replace("`", "'")
        text = text.replace("f.(x)", "f'(x)")
        text = text.replace("f(z)", "f(x)") # Often misread when x has a prime next to it

        # Normalize characters
        text = text.replace("−", "-")
        text = text.replace("×", "*")
        text = text.replace("•", "*")

        # Fix OCR dropped exponents: e.g. x2 -> x^2, 2x2 -> 2x^2, x3 -> x^3 (avoiding words like dx)
        text = re.sub(r'(?<![a-zA-Z])([xyzXYZ])(\d+)', r'\1^\2', text)
        text = re.sub(r'(\d+)([xyzXYZ])(\d+)', r'\1\2^\3', text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # -------------------------------------------------

    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names from the problem"""

        vars_found = re.findall(r"[a-zA-Z]", text)

        return sorted(list(set(vars_found)))

    # -------------------------------------------------

    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text"""

        matches = re.findall(r"[a-zA-Z0-9\^\+\-\*/\(\)\s]+=[a-zA-Z0-9\^\+\-\*/\(\)\s]+", text)

        return matches

    # -------------------------------------------------

    def _detect_topic(self, text: str) -> str:
        """Simple topic classification"""

        text = text.lower()

        # Calculus indicators
        if any(term in text for term in ["derivative", "d/dx", "dy/dx", "diff", "limit", "integral", "∫"]):
            return "calculus"
        
        if "'" in text and ("f(" in text or "y" in text):
            return "calculus"

        if "matrix" in text or "vector" in text:
            return "linear_algebra"

        if "probability" in text or "stat" in text:
            return "probability"

        return "algebra"

    # -------------------------------------------------

    def _llm_parse(self, text: str, source: str) -> Dict[str, Any]:
        """Fallback LLM parsing"""

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
                "topic": "unknown",
                "subtopic": "unknown",
                "variables": [],
                "constraints": [],
                "needs_clarification": True,
                "clarification_needed": "Unable to parse problem structure"
            }

        return parsed

    # -------------------------------------------------

    def _get_system_prompt(self) -> str:

        return """
You are a mathematical problem parser.

Your task:
1. Normalize the problem statement
2. Identify topic and subtopic
3. Extract variables and constraints

Return ONLY valid JSON in this format:

{
  "problem_text": "...",
  "topic": "algebra|calculus|probability|linear_algebra",
  "subtopic": "...",
  "variables": ["..."],
  "constraints": ["..."],
  "needs_clarification": false,
  "clarification_needed": null
}

CRITICAL: 
- DO NOT invent numbers, variables, or constraints that are not in the provided text.
- If the text is messy or missing parts of an equation (e.g. "f(x) =" with nothing after), mark "needs_clarification": true.
- ONLY output the JSON.
"""

    # -------------------------------------------------

    def _create_user_prompt(self, text: str, source: str) -> str:

        return f"""
Parse the following math problem (source: {source}):

{text}

Return structured JSON.
"""