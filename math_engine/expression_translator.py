import sympy as sp
import re
from typing import Dict, Any, Optional


class ExpressionTranslator:
    """
    Converts parsed math expressions into SymPy objects.
    """

    def __init__(self):
        self.symbol_cache = {}

    def translate(self, parsed_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert constraints or expressions into SymPy form.
        """

        equations = parsed_data.get("constraints", [])

        if not equations:
            return None

        sympy_equations = []

        for eq in equations:

            eq = self._normalize(eq)

            try:
                if "=" in eq:
                    left, right = eq.split("=")
                    sym_eq = sp.Eq(
                        sp.sympify(left),
                        sp.sympify(right)
                    )
                else:
                    sym_eq = sp.sympify(eq)

                sympy_equations.append(sym_eq)

            except Exception:
                continue

        if not sympy_equations:
            return None

        return {
            "sympy_equations": sympy_equations
        }

    def _normalize(self, expr: str) -> str:
        """
        Normalize expression syntax.
        """

        expr = expr.replace("^", "**")
        expr = expr.replace("×", "*")
        expr = expr.replace("-", "-")

        # Remove spaces around operators
        expr = re.sub(r"\s+", "", expr)

        return expr