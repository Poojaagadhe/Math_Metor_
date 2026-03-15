from typing import Dict, Any
import re
from agents.base_agent import BaseAgent
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PlannerAgent(BaseAgent):
    """Decides the best solving strategy"""

    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            model=Config.PARSER_MODEL
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:

        problem_text = input_data.get("problem_text", "")
        topic = input_data.get("topic", "unknown")
        routing = input_data.get("routing", {})
        task = routing.get("task", "simplify")

        plan = self._rule_based_plan(problem_text, topic, task)

        return plan

    def _rule_based_plan(self, text: str, topic: str, task: str):

        text_lower = text.lower()

        # 1. Specialized Task Routing (from RouterAgent)
        if task in ["derivative", "integral"]:
            return {"strategy": "calculus_tool"}

        # 2. Specific Pattern Routing (Function Evaluation)
        if re.search(r"f\(\d+\)", text_lower):
            return {"strategy": "function_tool"}

        # 3. Solver Routing
        if task in ["solve", "simplify"]:
            return {"strategy": "sympy_tool"}

        # 4. Topic-based Fallback
        if topic == "probability":
            return {"strategy": "rag_reasoning"}

        # 5. Default Fallback
        return {"strategy": "llm_reasoning"}

    def plan_ocr_correction(self, text: str, feedback: str) -> str:
        """Correct OCR mistakes based on feedback from the verifier."""
        logger.info("PlannerAgent planning OCR correction...")
        
        system_prompt = """You are an expert at correcting mathematical OCR mistakes.
You will be given the original text extracted by an OCR engine, and feedback on why it failed to be parsed or solved.
Your job is to fix the OCR transcription errors (e.g. confusing 1 and l, x and \\times, missing operators, poorly formatted powers, unbalanced parentheses) to produce a logically sound mathematical expression or problem.
CRITICAL: When formatting powers or exponents, use standard LaTeX syntax (e.g. x^{3} or x^3), NOT Python's double asterisk (**).
Return ONLY the corrected text, nothing else."""

        user_prompt = f"""Original OCR Text:
{text}

Feedback/Issues found during verification/parsing:
{feedback}

Please provide the corrected math problem text exactly as it should be processed:"""

        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]

        response = self._call_llm(messages, max_tokens=500)
        return response.strip()