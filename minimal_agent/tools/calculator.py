from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

from .base import ToolSpec


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200, description="只包含数字、括号和 + - * / // % ** 的数学表达式")


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_MAX_ABS_VALUE = 1_000_000_000_000
_MAX_POWER = 12


def _validate_result(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("表达式结果不是有限数值")
    if abs(value) > _MAX_ABS_VALUE:
        raise ValueError("表达式结果超出允许范围")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只允许数字常量")
        return _validate_result(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return _validate_result(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("除数不能为 0")
        if isinstance(node.op, ast.Pow) and (abs(right) > _MAX_POWER or abs(left) > _MAX_ABS_VALUE):
            raise ValueError("幂运算超出允许范围")
        return _validate_result(_BINARY_OPERATORS[type(node.op)](left, right))
    raise ValueError("表达式只允许数字、括号和基本算术运算符")


def _handle(args: CalculatorArgs, *, session_state: dict[str, Any]) -> dict[str, Any]:
    try:
        tree = ast.parse(args.expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("不是合法的数学表达式") from exc
    result = _evaluate(tree.body)
    return {"expression": args.expression, "result": result}


def calculator_tool() -> ToolSpec:
    return ToolSpec(
        "calculator",
        "安全计算四则运算、整除、取模和有限范围的幂运算；不执行变量、函数或任意代码",
        CalculatorArgs,
        _handle,
        side_effect="compute",
    )
