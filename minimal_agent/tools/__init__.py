from .base import ToolRegistry, ToolSpec
from .calculator import calculator_tool
from .expense_tracker import expense_tracker_tool
from .search import search_tool
from .weather import weather_tool


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (calculator_tool(), search_tool(), weather_tool(), expense_tracker_tool()):
        registry.register(tool)
    return registry


__all__ = ["ToolRegistry", "ToolSpec", "default_registry"]
