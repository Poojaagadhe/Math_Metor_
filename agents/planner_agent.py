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

        plan = self._rule_based_plan(problem_text, topic)

        return plan

    def _rule_based_plan(self, text: str, topic: str):

        text = text.lower()

        if "derivative" in text or "d/dx" in text:
            return {"strategy": "calculus_tool"}

        if re.search(r"f\(\d+\)", text):
            return {"strategy": "function_eval"}

        if "=" in text:
            return {"strategy": "symbolic_solver"}

        if topic == "probability":
            return {"strategy": "rag_reasoning"}

        return {"strategy": "llm_reasoning"}

    def plan_ocr_correction(self, text: str, feedback: str) -> str:
        """Correct OCR mistakes based on feedback from the verifier."""
        logger.info("PlannerAgent planning OCR correction...")
        
        system_prompt = """You are an expert at correcting mathematical OCR mistakes.
You will be given the original text extracted by an OCR engine, and feedback on why it failed to be parsed or solved.
Your job is to fix the OCR transcription errors (e.g. confusing 1 and l, x and \\times, missing operators, poorly formatted powers, unbalanced parentheses) to produce a logically sound mathematical expression or problem.
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