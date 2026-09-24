"""Built-in safe tools for AURA."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.tools.builtins.calculator import CalculatorTool
from backend.app.tools.builtins.current_time import CurrentTimeTool
from backend.app.tools.builtins.echo import EchoTool

if TYPE_CHECKING:
    from backend.app.tools.protocols import ToolRegistryProtocol


def register_builtin_tools(registry: ToolRegistryProtocol) -> None:
    """Register all standard built-in tools into the provided registry."""
    registry.register(EchoTool())
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())


__all__ = [
    "CalculatorTool",
    "CurrentTimeTool",
    "EchoTool",
    "register_builtin_tools",
]
