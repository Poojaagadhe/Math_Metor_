import sympy as sp
from tools.base_tool import BaseTool

class SympyTool(BaseTool):

    name = "sympy_solver"

    def run(self, query: str):

        try:
            expr = query.replace("^","**")
            result = sp.sympify(expr)

            return str(result)

        except Exception:
            return None