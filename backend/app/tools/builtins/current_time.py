"""Current time tool returning UTC timestamp."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.app.tools.errors import ToolExecutionError
from backend.app.tools.models import RiskLevel, ToolCategory, ToolMetadata


class CurrentTimeTool:
    """Returns the current UTC date and time. No side effects."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_current_time",
            description="Get the current date and time in UTC.",
            category=ToolCategory.UTILITY,
            risk_level=RiskLevel.SAFE,
            input_schema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "strftime format string.",
                        "default": "%Y-%m-%d %H:%M:%S UTC",
                    }
                },
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        fmt = arguments.get("format", "%Y-%m-%d %H:%M:%S UTC")
        try:
            return datetime.now(timezone.utc).strftime(fmt)
        except (ValueError, TypeError) as exc:
            raise ToolExecutionError(f"Invalid strftime format: {exc}") from exc
