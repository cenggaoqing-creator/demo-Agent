from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError
from ..errors import ToolExecutionError, ToolValidationError


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Callable[..., Any]
    side_effect: str = "read"
    requires_confirmation: bool = False
    confirmation_check: Callable[[BaseModel], bool] | None = None

    def needs_confirmation(self, args: BaseModel) -> bool:
        return self.requires_confirmation and (self.confirmation_check is None or self.confirmation_check(args))

    @property
    def parameters(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def validate(self, arguments: dict[str, Any]) -> BaseModel:
        try:
            return self.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(f"工具 {self.name} 参数错误: {exc}") from exc

    def invoke(self, arguments: dict[str, Any], *, session_state: dict[str, Any]) -> Any:
        args = self.validate(arguments)
        try:
            return self.handler(args, session_state=session_state)
        except ToolValidationError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"工具 {self.name} 执行失败: {exc}") from exc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具已注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未知工具: {name}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters, "side_effect": t.side_effect, "requires_confirmation": t.requires_confirmation} for t in self._tools.values()]
