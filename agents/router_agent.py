"""Router Agent - Classifies problem type and routes workflow"""
import json
import re
from typing import Dict, Any
from agents.base_agent import BaseAgent
from utils.config import Config
from utils.logger import setup_logger
from utils.taxonomy import (
    validate_classification,
    canonical_topic,
    taxonomy_as_prompt_text,
)

logger = setup_logger(__name__)


class RouterAgent(BaseAgent):
    """Routes problems to appropriate solving workflow"""

    def __init__(self):
        super().__init__(
            name="RouterAgent",
            model=Config.ROUTER_MODEL
        )

    # ── Task detection ────────────────────────────────────────────────────

    def detect_task(self, text: str) -> str:
        """
        Lightweight rule-based detection for math tasks.
        Returns one of: derivative | integral | arithmetic | solve | simplify
        """
        text_norm = (
            text.replace("\u2019", "'")
                .replace("\u2032", "'")
                .replace("\u2035", "'")
                .replace("`", "'")
        )
        text_lower = text_norm.lower()

        if any(kw in text_lower for kw in ["f'(x)", "f'", "derivative", "differentiate", "d/dx", "dy/dx", "df/dx"]):
            return "derivative"

        if any(kw in text_lower for kw in ["integrate", "integral", "∫"]):
            return "integral"

        # Pure numeric expressions → arithmetic task
        stripped = re.sub(r'[\s\+\-\*\/\(\)\.\,\%]', '', text_lower)
        if stripped.isdigit():
            return "arithmetic"

        if any(kw in text_lower for kw in ["solve", "evaluate", "find", "=", "calculate"]):
            return "solve"

        return "simplify"

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route problem to appropriate workflow.

        Uses parser's topic/subtopic as the authoritative source.
        Only calls the LLM for non-arithmetic, non-obvious problems.
        """
        problem_text = input_data.get("problem_text", "")
        topic = input_data.get("topic", "Algebra")
        subtopic = input_data.get("subtopic", "General")
        difficulty = input_data.get("difficulty", "Medium")

        logger.info(f"Routing problem — Topic: {topic} | Subtopic: {subtopic}")

        detected_task = self.detect_task(problem_text)
        logger.info(f"Detected task: {detected_task}")

        # ── Arithmetic Shortcut: skip LLM entirely ────────────────────────
        if topic == "Arithmetic":
            routing = self._arithmetic_routing(subtopic, difficulty, detected_task)
            logger.info(f"Arithmetic shortcut applied — Routing: {routing}")
            return routing

        # ── Calculus: also skip LLM if task is already clear ─────────────
        if topic == "Calculus" and detected_task in ("derivative", "integral"):
            routing = {
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty or "Hard",
                "required_tools": ["symbolic_solver"],
                "solution_strategy": f"Apply standard {subtopic.lower()} rules using SymPy",
                "workflow": "calculus",
                "task": detected_task,
            }
            logger.info(f"Calculus shortcut applied — task: {detected_task}")
            return routing

        # ── General: call LLM to validate / refine routing ───────────────
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(problem_text, topic, subtopic, difficulty)

        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]

        response = self._call_llm(messages, max_tokens=600)

        # Strip markdown code fences
        clean_response = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.MULTILINE)
        clean_response = re.sub(r'```\s*$', '', clean_response.strip(), flags=re.MULTILINE)

        try:
            json_start = clean_response.find("{")
            json_end = clean_response.rfind("}") + 1
            routing = json.loads(clean_response[json_start:json_end])

            # Always fill in the rule-based task
            if "task" not in routing:
                routing["task"] = detected_task

        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse routing response — using fallback")
            routing = {
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty or "Medium",
                "required_tools": ["symbolic_solver"],
                "solution_strategy": "standard",
                "workflow": "general",
                "task": detected_task,
            }

        # ── Canonicalize the LLM's topic/subtopic via taxonomy ────────────
        raw_topic = routing.get("topic", topic)
        raw_sub = routing.get("subtopic", subtopic)
        ctopic, csubtopic, was_valid = validate_classification(raw_topic, raw_sub)
        if not was_valid:
            logger.warning(
                f"LLM returned non-canonical classification: topic='{raw_topic}' "
                f"subtopic='{raw_sub}' → corrected to {ctopic}/{csubtopic}"
            )
        routing["topic"] = ctopic
        routing["subtopic"] = csubtopic

        logger.info(
            f"Routed to task: {routing.get('task')} | "
            f"Topic: {routing.get('topic')} | "
            f"Difficulty: {routing.get('difficulty')}"
        )

        return routing

    # ── Arithmetic shortcut builder ───────────────────────────────────────

    @staticmethod
    def _arithmetic_routing(subtopic: str, difficulty: str, task: str) -> Dict[str, Any]:
        return {
            "topic": "Arithmetic",
            "subtopic": subtopic,
            "difficulty": difficulty or "Easy",
            "required_tools": ["calculator"],
            "solution_strategy": f"Direct numeric calculation ({subtopic.lower()})",
            "workflow": "arithmetic",
            "task": task if task == "arithmetic" else "arithmetic",
        }

    # ── Prompts ───────────────────────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        taxonomy_text = taxonomy_as_prompt_text()
        return f"""You are a math problem router. The Parser Agent has already classified this problem.
Your job is to VALIDATE the classification and determine workflow details.

{taxonomy_text}

Output ONLY valid JSON:
{{
  "topic": "<exact topic from taxonomy>",
  "subtopic": "<exact subtopic from taxonomy>",
  "difficulty": "Easy|Medium|Hard",
  "required_tools": ["calculator|symbolic_solver|graphing|matrix_ops"],
  "solution_strategy": "brief description of recommended approach",
  "workflow": "arithmetic|algebraic|calculus|probabilistic|linear_algebra|general",
  "task": "arithmetic|solve|derivative|integral|simplify"
}}

Rules:
- NEVER change "Arithmetic" to "Algebra" — if the parser said Arithmetic, keep it.
- Prefer "symbolic_solver" for algebraic/calculus problems.
- Use "calculator" for pure numeric problems."""

    def _create_user_prompt(self, problem_text: str, topic: str, subtopic: str, difficulty: str) -> str:
        return f"""Parser Agent classified this problem as:
  Topic: {topic}
  Subtopic: {subtopic}
  Difficulty: {difficulty}

Problem:
{problem_text}

Validate or refine this classification and provide the routing JSON."""
