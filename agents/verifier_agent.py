"""Verifier Agent - Verifies solution correctness"""
import json
import re
from typing import Dict, Any
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
        Verify solution correctness
        
        Args:
            input_data: Dictionary containing:
                - problem_text: Original problem
                - solution: Proposed solution
                - steps: Solution steps
                - topic: Problem topic
                
        Returns:
            Dictionary containing:
                - is_correct: Boolean
                - confidence: Confidence score (0-1)
                - issues_found: List of issues
                - verification_notes: Notes on verification
                - hitl_required: Whether HITL is needed
        """
        logger.info("Verifying solution...")
        
        problem_text = input_data.get('problem_text', '')
        solution = input_data.get('solution', '')
        steps = input_data.get('steps', [])
        topic = input_data.get('topic', '')
        
        # Create verification prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(problem_text, solution, steps, topic)
        
        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]
        
        # Call LLM
        response = self._call_llm(messages, max_tokens=1000)

        # Strip markdown code fences if present (e.g. ```json ... ```)
        clean_response = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.MULTILINE)
        clean_response = re.sub(r'```\s*$', '', clean_response.strip(), flags=re.MULTILINE)

        # Parse JSON response – also handle embedded JSON object
        try:
            json_start = clean_response.find("{")
            json_end = clean_response.rfind("}") + 1
            verification = json.loads(clean_response[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse verification response – using conservative fallback")
            verification = {
                "is_correct": False,
                "confidence": 0.5,
                "issues_found": ["Failed to verify solution"],
                "verification_notes": "Verification process encountered an error",
                "hitl_required": True
            }
            
        # Determine if HITL is required
        confidence = verification.get('confidence', 0.5)
        if confidence < Config.VERIFIER_CONFIDENCE_THRESHOLD:
            verification['hitl_required'] = True
        else:
            verification['hitl_required'] = False
            
        logger.info(f"Verification complete - Correct: {verification.get('is_correct')}, Confidence: {confidence:.2f}, HITL: {verification.get('hitl_required')}")
        
        return verification
        
    def _get_system_prompt(self) -> str:
        """Get system prompt for verifier"""
        return """You are a math solution verifier. Your job is to:
1. Check if the solution is mathematically correct
2. Verify all steps are valid
3. Check for common mistakes
4. Validate units and domains
5. Test edge cases if applicable

Output ONLY valid JSON in this format:
{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "issues_found": ["list", "of", "issues"],
  "verification_notes": "detailed notes on verification process",
  "checks_performed": ["list", "of", "verification", "checks"]
}

Be thorough and critical. Common things to check:
- Hallucination of Data: Does the solution use numbers, values, or points (e.g., "x=2") that are NOT in the problem text?
- Arithmetic errors
- Sign errors
- Domain restrictions (division by zero, sqrt of negative, etc.)
- Units consistency
- Boundary conditions
- Formula application
- Logical flow

If the solution hallucinates missing information or solves a different problem than what was asked, set "is_correct": false and list the specific hallucinated data as an issue.
If you're not confident (< 0.8), list specific concerns.
"""
        
    def _create_user_prompt(
        self,
        problem_text: str,
        solution: str,
        steps: list,
        topic: str
    ) -> str:
        """Create user prompt"""
        steps_text = "\n\n".join(steps) if steps else "No detailed steps provided"
        
        return f"""Verify this solution:

**Problem** ({topic}):
{problem_text}

**Proposed Solution**:
{solution}

**Solution Steps**:
{steps_text}

Provide verification JSON with your assessment."""
