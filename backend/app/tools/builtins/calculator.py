"""Safe calculator tool using AST parsing."""

from __future__ import annotations

import ast
import operator
from typing import Any

from backend.app.tools.errors import ToolExecutionError
from backend.app.tools.models import RiskLevel, ToolCategory, ToolMetadata

_OPERATORS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool:
    """Evaluates basic arithmetic expressions safely using AST parsing."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="Evaluate a basic arithmetic expression (add, subtract, multiply, divide).",
            category=ToolCategory.UTILITY,
            risk_level=RiskLevel.SAFE,
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ToolExecutionError("Expression must be a non-empty string.")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = self._eval_node(tree.body)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except SyntaxError as exc:
            raise ToolExecutionError(f"Syntax error in expression: {exc}") from exc

    def _eval_node(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            ):
                return node.value
            raise ToolExecutionError(
                f"Unsupported constant type: {type(node.value).__name__}"
            )

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _OPERATORS:
                raise ToolExecutionError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ToolExecutionError("Division by zero.")
            if op_type is ast.Pow and right > 1000:
                raise ToolExecutionError("Exponent too large.")
            return _OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _OPERATORS:
                raise ToolExecutionError(
                    f"Unsupported unary operator: {op_type.__name__}"
                )
            operand = self._eval_node(node.operand)
            return _OPERATORS[op_type](operand)

        raise ToolExecutionError(
            f"Unsupported or unsafe expression syntax: {type(node).__name__}"
        )
