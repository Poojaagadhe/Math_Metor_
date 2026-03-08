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