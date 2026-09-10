# pearl/tools/registry.py
from typing import Dict, List, Optional
from pearl.tools.base import BaseTool
from pearl.tools.search import GoogleSearchTool
from pearl.tools.finance import StockDataTool
from pearl.tools.math import MathTool
from pearl.tools.weather import WeatherTool

class ToolRegistry:
    """Passive catalog for storing and looking up available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_descriptions(self) -> str:
        """Formats tools for LLM prompts."""
        return "\n".join([f"- '{t.name}': {t.description}" for t in self._tools.values()])

    @classmethod
    def default(cls) -> "ToolRegistry":
        reg = cls()
        reg.register(GoogleSearchTool())
        reg.register(StockDataTool())
        reg.register(MathTool())
        reg.register(WeatherTool())
        return reg