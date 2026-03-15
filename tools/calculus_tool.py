import sympy as sp
from tools.base_tool import BaseTool

class CalculusTool(BaseTool):

    name = "calculus_solver"

    def run(self, query: str):

        try:
            q_lower = query.lower()
            if "derivative" in q_lower or "differentiate" in q_lower or "d/dx" in q_lower:

                # Extract expression after = or after the keyword
                if "=" in query:
                    expr_str = query.split("=")[-1].strip()
                else:
                    # Try to find math after the keyword
                    match = re.search(r"(derivative|differentiate|d/dx)\s+of?\s*(.*)", q_lower)
                    if match:
                        expr_str = match.group(2).strip()
                    else:
                        # Just take the whole string if no specific match
                        expr_str = query

                # Clean up any remaining words if it's a mixed sentence
                expr_str = re.sub(r"^[^\d x\(]+", "", expr_str).strip()

                x = sp.symbols('x')
                # Sympy normalization (just in case)
                expr_str = expr_str.replace("^", "**")
                
                result = sp.diff(expr_str, x)

                return f"Derivative = {result}"

        except Exception as e:
            return None

        return None