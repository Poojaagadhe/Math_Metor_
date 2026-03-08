import sympy as sp
from tools.base_tool import BaseTool

class CalculusTool(BaseTool):

    name = "calculus_solver"

    def run(self, query: str):

        try:
            if "derivative" in query.lower():

                expr = query.split("=")[1]
                x = sp.symbols('x')

                result = sp.diff(expr,x)

                return f"Derivative = {result}"

        except:
            return None

        return None