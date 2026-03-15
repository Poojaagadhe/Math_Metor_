"""Solver Agent - Solves math problems using symbolic math, RAG, and LLM reasoning"""

from typing import Dict, Any, Optional
import sympy as sp
import re

from agents.base_agent import BaseAgent
from rag.retriever import Retriever
from utils.config import Config
from utils.logger import setup_logger
from math_engine.expression_translator import ExpressionTranslator

logger = setup_logger(__name__)


class SolverAgent(BaseAgent):
    """Main reasoning and computation agent"""

    def __init__(self, retriever: Optional[Retriever] = None):

        super().__init__(
            name="SolverAgent",
            model=Config.SOLVER_MODEL
        )

        self.retriever = retriever or Retriever()
        self.translator = ExpressionTranslator()

    # --------------------------------------------------

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:

        logger.info("SolverAgent started")

        problem_text = input_data.get("problem_text", "")
        topic = input_data.get("topic", "unknown")
        routing = input_data.get("routing", {})
        parsed_data = input_data.get("parsed_data", {})
        plan = input_data.get("plan", {})

        # ----------------------------------------------
        # STEP 1: Attempt deterministic solving
        # ----------------------------------------------

        symbolic_solution = self._solve_symbolically(problem_text, parsed_data)

        if symbolic_solution:

            logger.info("Solved using symbolic math")

            return {
                "solution": symbolic_solution,
                "steps": [
                    "Converted problem to symbolic form",
                    "Solved using SymPy"
                ],
                "intermediate_results": [],
                "retrieved_context": [],
                "tools_used": ["sympy"]
            }

        # ----------------------------------------------
        # STEP 2: Retrieve knowledge from RAG
        # ----------------------------------------------

        contexts = self.retriever.retrieve(
            query=problem_text,
            topic=topic,
            n_results=3
        )

        formatted_context = self.retriever.format_context_for_prompt(contexts)

        # ----------------------------------------------
        # STEP 3: LLM reasoning fallback
        # ----------------------------------------------

        response = self._run_llm_reasoning(
            problem_text,
            formatted_context,
            routing
        )

        solution_data = self._parse_solution(response)

        solution_data["retrieved_context"] = contexts
        solution_data["raw_response"] = response

        return solution_data

    # --------------------------------------------------
    # SYMBOLIC SOLVER
    # --------------------------------------------------

    def _solve_symbolically(self, text: str, parsed_data: Dict[str, Any]) -> Optional[str]:

        # ---- 1. Expression Translator ----

        translated = self.translator.translate(parsed_data)

        if translated:

            equations = translated.get("sympy_equations", [])

            try:

                if len(equations) == 1:
                    result = sp.solve(equations[0])

                else:
                    result = sp.solve(equations)

                if result:
                    return str(result)

            except Exception as e:
                logger.warning(f"Symbolic solve failed: {e}")

        # ---- 2. Function Evaluation ----

        try:

            pattern = r"f\((\w)\)\s*=\s*(.+)"
            match = re.search(pattern, text)

            if match:

                var = sp.symbols(match.group(1))
                expr = match.group(2).replace("^", "**")

                sym_expr = sp.sympify(expr)

                call_match = re.search(r"f\((\d+)\)", text)

                if call_match:

                    value = int(call_match.group(1))
                    result = sym_expr.subs(var, value)

                    return f"f({value}) = {result}"

        except Exception as e:
            logger.warning(f"Function evaluation failed: {e}")

        # ---- 3. Derivative Detection ----

        try:
            # Look for "f'(x)" or "derivative of ..."
            derivative_requested = False
            expr_to_diff = None
            
            # Pattern for f'(x)
            if "f'(x)" in text or "derivative" in text.lower() or "d/dx" in text.lower():
                derivative_requested = True
                
            if derivative_requested:
                # Try to extract expression from f(x) = ...
                expr_match = re.search(r"f\(x\)\s*=\s*([^,\?]+)", text)
                if expr_match:
                    expr_to_diff = expr_match.group(1).strip()
                elif "=" in text:
                    # Fallback to text after =
                    expr_to_diff = text.split("=")[-1].strip()
                    # Clean up trailing punctuation
                    expr_to_diff = re.sub(r"[,\?\.!]+$", "", expr_to_diff)

                if expr_to_diff:
                    x = sp.symbols("x")
                    # Convert ^ to ** and perform sympify
                    sp_expr = sp.sympify(expr_to_diff.replace("^", "**"))
                    derivative = sp.diff(sp_expr, x)
                    
                    return f"f'(x) = {derivative}"

        except Exception as e:
            logger.warning(f"Derivative computation failed: {e}")

        return None

    # --------------------------------------------------
    # LLM REASONING
    # --------------------------------------------------

    def _run_llm_reasoning(self, problem_text, context, routing):

        system_prompt = self._get_system_prompt()

        user_prompt = self._create_user_prompt(
            problem_text,
            context,
            routing
        )

        messages = [
            self._create_system_message(system_prompt),
            self._create_user_message(user_prompt)
        ]

        return self._call_llm(messages, max_tokens=2000)

    # --------------------------------------------------
    # PROMPTS
    # --------------------------------------------------

    def _get_system_prompt(self):

        return """
You are an expert mathematics tutor. Solve the given problem completely with clear step-by-step working.

REQUIRED FORMAT – you MUST follow this structure exactly:

## Solution Strategy
Briefly explain the approach you will use (1-2 sentences).

## Step-by-Step Solution

### Step 1: [Descriptive title]
Show the work for this step. Explain WHY this step is done.

### Step 2: [Descriptive title]
Continue working. Use math notation like x^2, x^3, f'(x), sqrt(x).

### Step 3: [Descriptive title]
(Continue for as many steps as needed)

## Final Answer
State the answer clearly and concisely. Example: f'(x) = 3x^2

RULES:
- ALWAYS include all section headers exactly as shown above.
- ALWAYS show at least 2 steps, even for simple problems.
- Denote exponents with ^  (e.g. x^3, not x3 or x³).
- Denote derivatives with prime notation: f'(x), f''(x).
- Show every algebraic step explicitly – do not skip steps.
- If the problem is ambiguous, state your assumptions at the top.
"""


    def _create_user_prompt(self, problem_text, context, routing):

        return f"""
Solve this math problem.

Problem:
{problem_text}

Reference context:
{context}

Difficulty: {routing.get('difficulty','medium')}
Suggested strategy: {routing.get('solution_strategy','standard')}

Provide a complete solution.
"""

    # --------------------------------------------------
    # RESPONSE PARSER
    # --------------------------------------------------

    def _parse_solution(self, response: str):
        """
        Parse the LLM response into structured components.

        Handles multiple heading styles that different LLMs produce:
          - ## Step 1 / ### Step 1
          - **Step 1:** / **Step 1 –**
          - Step 1: (plain)
        Always falls back to the full response so nothing is ever lost.
        """
        import re

        solution = ""
        steps = []

        lines = response.split("\n")
        in_final = False
        current_step = ""

        # Patterns that mark the start of a step
        step_pattern = re.compile(
            r"^(#{1,3}\s*step\s*\d|"          # ## Step 1, ### Step 1
            r"\*{1,2}step\s*\d|"              # **Step 1
            r"step\s*\d+\s*[:–-])",           # Step 1: / Step 1 –
            re.IGNORECASE
        )
        # Patterns that mark the Final Answer section
        final_pattern = re.compile(
            r"(final\s*answer|answer\s*:|result\s*:)", re.IGNORECASE
        )

        for line in lines:
            stripped = line.strip()

            # Detect start of Final Answer section
            if final_pattern.search(stripped):
                in_final = True
                if current_step:
                    steps.append(current_step.strip())
                    current_step = ""
                continue

            # Collect final answer text
            if in_final and stripped:
                solution += stripped + " "
                continue

            # Detect step headings
            if step_pattern.match(stripped):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line + "\n"
            elif current_step:
                current_step += line + "\n"

        # Don't lose the last step
        if current_step:
            steps.append(current_step.strip())

        # If regex parsing missed the solution text, fall back to full response
        final_answer = solution.strip() or self._extract_last_paragraph(response)

        return {
            "solution": final_answer,
            "steps": steps,
            "full_solution": response,   # always keep the full formatted text
            "tools_used": ["llm", "rag"]
        }

    def _extract_last_paragraph(self, response: str) -> str:
        """Return the last non-empty paragraph as a fallback answer."""
        paragraphs = [p.strip() for p in response.strip().split("\n\n") if p.strip()]
        return paragraphs[-1] if paragraphs else response.strip()