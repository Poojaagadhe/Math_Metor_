from typing import Any

class BaseTool:
    """Base class for computational tools"""

    name = "base_tool"
    description = ""

    def run(self, query: str) -> Any:
        raise NotImplementedError