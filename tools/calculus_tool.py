import re
import sympy as sp
from tools.base_tool import BaseTool

class CalculusTool(BaseTool):

    name = "calculus_solver"

    def run(self, query: str):

        try:
            q_norm = query.replace("’", "'").replace("′", "'").replace("‵", "'").replace("`", "'")
            q_lower = q_norm.lower()

            if any(kw in q_lower for kw in ["derivative", "differentiate", "d/dx", "f'(x)", "f'"]):

                # If the problem has f(x) = <expr>
                match_fn = re.search(r"f\([a-zA-Z]\)\s*=\s*([^,\?]+)", q_norm)
                if match_fn:
                    expr_str = match_fn.group(1).strip()
                elif "=" in q_norm:
                    # Expression after =
                    expr_str = q_norm.split("=")[-1].strip()
                    expr_str = re.split(r"[,;]|\b(?:find|where|calculate|evaluate)\b", expr_str, flags=re.IGNORECASE)[0].strip()
                else:
                    match = re.search(r"(?:derivative|differentiate|d/dx)\s+of?\s*(.*)", q_lower)
                    if match:
                        expr_str = match.group(1).strip()
                    else:
                        expr_str = q_norm

                # Clean trailing punctuation
                expr_str = re.sub(r"[,\?\.\!]+$", "", expr_str).strip()

                # Fix OCR dropped exponents: e.g. x2 -> x^2, 2x2 -> 2*x^2
                expr_str = re.sub(r'(\d+)([a-zA-Z])(\d+)', r'\1*\2^\3', expr_str)
                expr_str = re.sub(r'(?<![a-zA-Z])([a-zA-Z])(\d+)', r'\1^\2', expr_str)

                # Add implicit multiplication for numbers followed by variables (e.g. 5x -> 5*x, 2x^2 -> 2*x^2)
                expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)

                # SymPy normalization
                expr_str = expr_str.replace("^", "**")

                x = sp.symbols('x')
                sym_expr = sp.sympify(expr_str)
                result = sp.diff(sym_expr, x)

                return f"f'(x) = {result}".replace('**', '^')

        except Exception:
            return None

        return None