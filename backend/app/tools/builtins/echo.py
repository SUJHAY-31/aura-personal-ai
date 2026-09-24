"""Echo tool for pipeline testing and verification."""

from __future__ import annotations

from typing import Any

from backend.app.tools.models import RiskLevel, ToolCategory, ToolMetadata


class EchoTool:
    """Returns the input message unchanged. Used for testing the tool pipeline."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="echo",
            description="Echo back the provided message. For testing.",
            category=ToolCategory.UTILITY,
            risk_level=RiskLevel.SAFE,
            input_schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo back.",
                    }
                },
                "required": ["message"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        return str(arguments["message"])
