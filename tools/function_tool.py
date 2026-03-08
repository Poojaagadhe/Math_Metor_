import sympy as sp
import re
from tools.base_tool import BaseTool

class FunctionTool(BaseTool):

    name = "function_solver"

    def run(self, query: str):

        pattern = r"f\((\w)\)\s*=\s*(.+)"
        match = re.search(pattern, query)

        if not match:
            return None

        var = sp.symbols(match.group(1))
        expr = match.group(2).replace("^","**")

        expression = sp.sympify(expr)

        call = re.search(r"f\((\d+)\)", query)

        if call:
            value = int(call.group(1))
            result = expression.subs(var,value)

            return f"f({value}) = {result}"

        return None