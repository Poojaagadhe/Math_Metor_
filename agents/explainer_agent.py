"""Explainer Agent - Generates student-friendly explanations"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.config import Config
from utils.logger import setup_logger
from utils.visualizer import MathVisualizer
import re

logger = setup_logger(__name__)

class ExplainerAgent(BaseAgent):
    """Generates clear, student-friendly explanations"""
    
    def __init__(self):
        super().__init__(
            name="ExplainerAgent",
            model=Config.EXPLAINER_MODEL
        )
        self.visualizer = MathVisualizer()
        
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate student-friendly explanation
        
        Args:
            input_data: Dictionary containing:
                - problem_text: Original problem
                - solution: Final answer
                - steps: Solution steps
                - topic: Problem topic
                - retrieved_context: Context used
                
        Returns:
            Dictionary containing:
                - explanation: Formatted explanation
                - key_concepts: List of key concepts used
                - formulas_used: List of formulas with citations
        """
        logger.info("Generating explanation...")
        
        problem_text = input_data.get('problem_text', '')
        solution = input_data.get('solution', '')
        steps = input_data.get('steps', [])
        topic = input_data.get('topic', '')
        contexts = input_data.get('retrieved_context', [])
        
        # Create explanation prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(problem_text, solution, steps, topic, contexts)
        
        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]
        
        # Call LLM
        response = self._call_llm(messages, max_tokens=1500)
        
        # Extract key concepts and formulas
        key_concepts = self._extract_concepts(response, topic)
        formulas = self._extract_formulas(contexts)
        
        # Generate visualizations - DISABLED (producing incorrect graphs)
        # images = self._generate_visualizations(problem_text, solution, steps, topic)
        images = []  # Disabled for now
        
        logger.info(f"Explanation generated")
        
        return {
            "explanation": response,
            "key_concepts": key_concepts,
            "formulas_used": formulas,
            "visualization_images": images
        }
        
    def _get_system_prompt(self) -> str:
        """Get system prompt for explainer"""
        return """You are a patient math tutor. Your job is to:
1. Explain the solution in a clear, student-friendly way
2. Break down complex steps into simple concepts
3. Provide intuition for why each step works
4. Cite formulas and theorems used
5. Highlight common mistakes to avoid

Format your explanation as:

# Problem Understanding
[Explain what the problem is asking in simple terms]

# Solution Approach
[Explain the overall strategy]

# Step-by-Step Explanation

## Step 1: [Title]
[Clear explanation with intuition]

## Step 2: [Title]
[Clear explanation with intuition]

[Continue for all steps...]

# Final Answer
[Restate the answer clearly]

# Key Takeaways
- [Important concept 1]
- [Important concept 2]
- [Common mistake to avoid]

Use clear language. Explain WHY, not just WHAT. Make it educational."""
        
    def _create_user_prompt(
        self,
        problem_text: str,
        solution: str,
        steps: list,
        topic: str,
        contexts: list
    ) -> str:
        """Create user prompt"""
        steps_text = "\n\n".join(steps) if steps else "Solution provided without detailed steps"
        
        # Extract relevant formulas from context
        formulas_text = ""
        if contexts:
            formulas_text = "\n\nRelevant formulas and concepts:\n"
            for ctx in contexts[:2]:  # Use top 2 contexts
                formulas_text += f"- {ctx.get('content', '')[:200]}...\n"
        
        return f"""Create a student-friendly explanation for this solution:

**Problem** ({topic}):
{problem_text}

**Solution**:
{solution}

**Steps**:
{steps_text}
{formulas_text}

Provide a clear, educational explanation that helps students understand not just the answer, but the reasoning."""
        
    def _extract_concepts(self, explanation: str, topic: str) -> list:
        """Extract key concepts from explanation"""
        # Simple extraction - look for common patterns
        concepts = [topic]
        
        # Look for mathematical terms
        terms = [
            'derivative', 'integral', 'limit', 'equation', 'formula',
            'theorem', 'property', 'rule', 'method', 'technique'
        ]
        
        for term in terms:
            if term in explanation.lower():
                concepts.append(term)
                
        return list(set(concepts))[:5]  # Return up to 5 unique concepts
        
    def _extract_formulas(self, contexts: list) -> list:
        """Extract formulas from retrieved contexts"""
        formulas = []
        
        for ctx in contexts:
            content = ctx.get('content', '')
            source = ctx.get('source', 'unknown')
            
            # Look for formula patterns (lines with =, ^, /, etc.)
            lines = content.split('\n')
            for line in lines:
                if any(char in line for char in ['=', '^', '/', '*']) and len(line) < 100:
                    formulas.append({
                        "formula": line.strip(),
                        "source": source
                    })
                    
        return formulas[:3]  # Return up to 3 formulas
    
    def _generate_visualizations(self, problem_text: str, solution: str, steps: list, topic: str) -> List[str]:
        """
        Generate visual explanations based on problem type
        
        Args:
            problem_text: Original problem
            solution: Final answer
            steps: Solution steps
            topic: Problem topic
            
        Returns:
            List of image file paths
        """
        try:
            # Detect problem type and extract parameters
            problem_type, problem_data = self._detect_problem_type(problem_text, solution, topic)
            
            if not problem_type:
                logger.info("No visualization available for this problem type")
                return []
            
            # Generate visualization
            images = self.visualizer.generate_visualization(problem_type, problem_data)
            return images
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return []
    
    def _detect_problem_type(self, problem_text: str, solution: str, topic: str) -> tuple:
        """
        Detect problem type and extract relevant parameters
        
        Returns:
            Tuple of (problem_type, problem_data)
        """
        text = problem_text.lower()
        
        # Derivative detection
        if any(word in text for word in ['derivative', "d/dx", "dy/dx", "differentiate"]) or ("'" in text and "f(" in text):
            func = self._extract_function(problem_text)
            if func:
                return ("derivative", {"function": func, "x_range": (-5, 5)})
        
        # Integral detection
        if any(word in text for word in ['integral', 'integrate', '∫']):
            func = self._extract_function(problem_text)
            bounds = self._extract_bounds(problem_text)
            if func:
                return ("integral", {"function": func, "bounds": bounds or (0, 2)})
        
        # Quadratic equation detection
        if 'x^2' in text or 'x²' in text or 'quadratic' in text:
            coeffs = self._extract_quadratic_coeffs(problem_text, solution)
            if coeffs:
                roots = self._extract_roots(solution)
                return ("quadratic", {**coeffs, "roots": roots})
        
        # System of equations detection
        if 'system' in text or ('+' in text and '=' in text and text.count('=') >= 2):
            equations = self._extract_system_equations(problem_text)
            sol = self._extract_system_solution(solution)
            if equations:
                return ("system_equations", {**equations, "solution": sol})
        
        # Circle detection
        if 'circle' in text and 'radius' in text:
            radius = self._extract_number(problem_text, 'radius')
            if radius:
                return ("geometry_circle", {"radius": radius})
        
        # Triangle detection
        if 'triangle' in text:
            sides = self._extract_triangle_sides(problem_text)
            tri_type = 'right' if 'right' in text or 'pythagorean' in text else 'general'
            if sides:
                return ("geometry_triangle", {"sides": sides, "type": tri_type})
        
        # Rectangle detection
        if 'rectangle' in text:
            length = self._extract_number(problem_text, 'length')
            width = self._extract_number(problem_text, 'width')
            if length and width:
                return ("geometry_rectangle", {"length": length, "width": width})
        
        return (None, {})
    
    def _extract_function(self, text: str) -> str:
        """Extract mathematical function from text"""
        # Look for common patterns like "x^2", "x**2", "2x", etc.
        # Simple extraction - look for expression after "of" or ":"
        patterns = [
            r'of[:\s]+([x0-9\+\-\*\/\^\(\)\s]+)',
            r':[:\s]+([x0-9\+\-\*\/\^\(\)\s]+)',
            r'f\(x\)\s*=\s*([x0-9\+\-\*\/\^\(\)\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                func_str = match.group(1).strip()
                # Convert to Python syntax
                func_str = func_str.replace('^', '**').replace(' ', '')
                return func_str
        
        return None
    
    def _extract_bounds(self, text: str) -> tuple:
        """Extract integration bounds"""
        # Look for "from a to b" or "[a, b]"
        patterns = [
            r'from\s+([-\d.]+)\s+to\s+([-\d.]+)',
            r'\[([-\d.]+),\s*([-\d.]+)\]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return (float(match.group(1)), float(match.group(2)))
        
        return None
    
    def _extract_quadratic_coeffs(self, text: str, solution: str) -> dict:
        """Extract quadratic coefficients a, b, c"""
        # Look for ax^2 + bx + c pattern
        pattern = r'([-\d.]*)x\^2\s*([+\-])\s*([-\d.]*)x\s*([+\-])\s*([-\d.]+)'
        match = re.search(pattern, text.replace(' ', ''))
        
        if match:
            a = float(match.group(1) or '1')
            b = float(match.group(3) or '1') * (1 if match.group(2) == '+' else -1)
            c = float(match.group(5)) * (1 if match.group(4) == '+' else -1)
            return {"a": a, "b": b, "c": c}
        
        # Default quadratic for demonstration
        return {"a": 1, "b": -5, "c": 6}
    
    def _extract_roots(self, solution: str) -> list:
        """Extract roots from solution"""
        # Look for x = a, x = b pattern
        roots = re.findall(r'x\s*=\s*([-\d.]+)', solution)
        return [float(r) for r in roots] if roots else []
    
    def _extract_system_equations(self, text: str) -> dict:
        """Extract system of equations coefficients"""
        # Look for ax + by = c pattern
        lines = text.split('\n')
        equations = []
        
        for line in lines:
            # Pattern: ax + by = c
            pattern = r'([-\d.]*)x\s*([+\-])\s*([-\d.]*)y\s*=\s*([-\d.]+)'
            match = re.search(pattern, line.replace(' ', ''))
            if match:
                a = float(match.group(1) or '1')
                b = float(match.group(3) or '1') * (1 if match.group(2) == '+' else -1)
                c = float(match.group(4))
                equations.append({"a": a, "b": b, "c": c})
        
        if len(equations) >= 2:
            return {"equation1": equations[0], "equation2": equations[1]}
        
        # Default system for demonstration
        return {
            "equation1": {"a": 3, "b": 2, "c": 12},
            "equation2": {"a": 1, "b": -1, "c": 1}
        }
    
    def _extract_system_solution(self, solution: str) -> dict:
        """Extract x, y solution from text"""
        x_match = re.search(r'x\s*=\s*([-\d./]+)', solution)
        y_match = re.search(r'y\s*=\s*([-\d./]+)', solution)
        
        if x_match and y_match:
            # Handle fractions
            x_val = eval(x_match.group(1)) if '/' in x_match.group(1) else float(x_match.group(1))
            y_val = eval(y_match.group(1)) if '/' in y_match.group(1) else float(y_match.group(1))
            return {"x": x_val, "y": y_val}
        
        return {}
    
    def _extract_number(self, text: str, keyword: str) -> float:
        """Extract number after a keyword"""
        pattern = f'{keyword}[:\s]+([-\d.]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None
    
    def _extract_triangle_sides(self, text: str) -> list:
        """Extract triangle side lengths"""
        # Look for numbers that might be sides
        numbers = re.findall(r'\b(\d+)\b', text)
        if len(numbers) >= 2:
            return [float(n) for n in numbers[:3]]
        return [3, 4, 5]  # Default right triangle
