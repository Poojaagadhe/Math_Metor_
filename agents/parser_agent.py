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

        text = text.replace("x3", "x^3")
        text = text.replace("z3", "z^3")
        text = text.replace("−", "-")
        text = text.replace("×", "*")

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

        if "derivative" in text or "d/dx" in text:
            return "calculus"

        if "matrix" in text:
            return "linear_algebra"

        if "probability" in text:
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
"""

    # -------------------------------------------------

    def _create_user_prompt(self, text: str, source: str) -> str:

        return f"""
Parse the following math problem (source: {source}):

{text}

Return structured JSON.
"""