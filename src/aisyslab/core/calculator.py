"""Arithmetic as a tool: the model chooses the expression, code computes the number.

Models are unreliable at mental arithmetic and very reliable at writing ``45 - 6 * 5``.
Only numbers, parentheses and + - * / // % ** are accepted; there are no names to call.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from pydantic import BaseModel, Field

from .tools import Tool, ToolError

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate(expression: str) -> float:
    if len(expression) > 200:
        raise ToolError("Expression longer than 200 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Not an arithmetic expression: {exc.msg}.") from exc

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 10:
                    raise ToolError("Exponent too large (max 10).")
                # Nested powers grow without bound ((9**9)**9...): cap the result, not just the exponent.
                if abs(left) > 1 and abs(right) * math.log10(abs(left)) > 300:
                    raise ToolError("Result too large (more than 300 digits).")
            try:
                return _BINARY[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ToolError("Division by zero.") from exc
            except OverflowError as exc:
                raise ToolError("Result too large.") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise ToolError("Only numbers, parentheses and + - * / // % ** are allowed.")

    value = walk(tree)
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("Result too large.")
    return value


class CalculateArgs(BaseModel):
    expression: str = Field(description="Arithmetic only, e.g. '(45 + 12) - 6 * 5'.")


def calculator_tool() -> Tool:
    def run(args: CalculateArgs) -> dict[str, object]:
        value = evaluate(args.expression)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return {"expression": args.expression, "result": value}

    return Tool(
        "calculate",
        "Evaluate an arithmetic expression exactly. Use it instead of mental arithmetic.",
        CalculateArgs,
        run,
    )
