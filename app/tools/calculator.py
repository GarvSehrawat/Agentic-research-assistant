"""A calculator tool that evaluates arithmetic expressions safely.

We deliberately do NOT use Python's eval()/exec(). Instead we parse the
expression into an AST and only allow a small whitelist of numeric
operators, rejecting anything else (names, calls, attribute access,
imports, comprehensions, etc.).
"""
from __future__ import annotations

import ast
import operator
from typing import Any

from app.schemas.tools import CalculatorArgs
from app.tools.base import BaseTool, ToolResult

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(Exception):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise UnsafeExpressionError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise UnsafeExpressionError(f"Disallowed expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    if len(expression) > 200:
        raise UnsafeExpressionError("Expression too long.")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression: {exc}") from exc
    return _eval_node(parsed)


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression (+, -, *, /, //, %, **)."
    args_schema = CalculatorArgs

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            args = CalculatorArgs.model_validate(kwargs)
            result = safe_eval(args.expression)
            return ToolResult(success=True, output=str(result))
        except (UnsafeExpressionError, ZeroDivisionError, ValueError) as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Calculator failed: {exc}")
