"""Router Agent - Classifies problem type and routes workflow"""
import json
import re
from typing import Dict, Any
from agents.base_agent import BaseAgent
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RouterAgent(BaseAgent):
    """Routes problems to appropriate solving workflow"""
    
    def __init__(self):
        super().__init__(
            name="RouterAgent",
            model=Config.ROUTER_MODEL
        )
        
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route problem to appropriate workflow
        
        Args:
            input_data: Parsed problem from ParserAgent
                
        Returns:
            Dictionary containing:
                - topic: Main topic
                - subtopic: Specific subtopic
                - difficulty: easy|medium|hard
                - required_tools: List of tools needed
                - solution_strategy: Recommended approach
                - workflow: Workflow identifier
        """
        logger.info(f"Routing problem - Topic: {input_data.get('topic')}")
        
        problem_text = input_data.get('problem_text', '')
        topic = input_data.get('topic', 'unknown')
        subtopic = input_data.get('subtopic', '')
        
        # Create prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(problem_text, topic, subtopic)
        
        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]
        
        # Call LLM
        response = self._call_llm(messages, max_tokens=800)

        # Strip markdown code fences if present (e.g. ```json ... ```)
        clean_response = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.MULTILINE)
        clean_response = re.sub(r'```\s*$', '', clean_response.strip(), flags=re.MULTILINE)

        # Parse JSON response – also handle embedded JSON object
        try:
            json_start = clean_response.find("{")
            json_end = clean_response.rfind("}") + 1
            routing = json.loads(clean_response[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse routing response – using fallback")
            routing = {
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": "medium",
                "required_tools": ["calculator"],
                "solution_strategy": "standard",
                "workflow": "general"
            }
            
        logger.info(f"Routed to workflow: {routing.get('workflow')}, Difficulty: {routing.get('difficulty')}")
        
        return routing
        
    def _get_system_prompt(self) -> str:
        """Get system prompt for router"""
        return """You are a math problem router. Analyze the problem and determine:
1. The specific topic and subtopic
2. Difficulty level (easy/medium/hard)
3. Tools needed (calculator, symbolic_solver, graphing, etc.)
4. Best solution strategy
5. Workflow to use

Output ONLY valid JSON in this format:
{
  "topic": "algebra|calculus|probability|linear_algebra",
  "subtopic": "specific subtopic",
  "difficulty": "easy|medium|hard",
  "required_tools": ["tool1", "tool2"],
  "solution_strategy": "brief description of recommended approach",
  "workflow": "workflow_identifier"
}

Available tools:
- calculator: For numerical calculations
- symbolic_solver: For symbolic math (SymPy)
- graphing: For plotting
- matrix_ops: For matrix operations

Workflows:
- algebraic: For algebraic manipulation
- calculus: For calculus problems
- probabilistic: For probability problems
- linear_algebra: For matrix/vector problems
- general: For mixed or unclear problems"""
        
    def _create_user_prompt(self, problem_text: str, topic: str, subtopic: str) -> str:
        """Create user prompt"""
        return f"""Analyze and route this problem:

Topic: {topic}
Subtopic: {subtopic}

Problem:
{problem_text}

Provide the routing JSON."""
