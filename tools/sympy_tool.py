import sympy as sp
from tools.base_tool import BaseTool

class SympyTool(BaseTool):

    name = "sympy_solver"

    def run(self, query: str):

        try:
            # Remove instructional keywords
            clean_query = re.sub(r'^(solve|evaluate|find|calculate|simplify|result)\b', '', query, flags=re.IGNORECASE).strip()
            clean_query = clean_query.lstrip(':').strip()

            # Handle ^ and other common OCR-to-SymPy mismatches
            expr = clean_query.replace("^","**")
            
            # SymPy solve can handle equations if it's in a single string like "x + 2 = 5"
            # though sympify might need sp.Eq if we want to be very precise.
            # But sympify often handles = by returning an Equality object.
            result = sp.sympify(expr)

            return str(result)

        except Exception:
            return None