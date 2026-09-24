"""In-memory tool registry, default permission gate, and audit sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.tools.errors import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
)
from backend.app.tools.models import (
    AuditEvent,
    ToolCategory,
    ToolDefinition,
    ToolInvocation,
    ToolMetadata,
)

if TYPE_CHECKING:
    from backend.app.tools.protocols import ToolProtocol


class ToolRegistry:
    """In-memory registry for tool registration, lookup, and discovery."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolProtocol) -> None:
        """
        Register a tool.

        Raises:
            ToolAlreadyRegisteredError: If a tool with the same name exists.
        """
        name = tool.metadata.name
        if name in self._tools:
            raise ToolAlreadyRegisteredError(f"Tool '{name}' is already registered.")
        self._tools[name] = ToolDefinition(
            metadata=tool.metadata,
            implementation=tool,
        )

    def unregister(self, name: str) -> None:
        """
        Remove a tool by name.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found.")
        del self._tools[name]

    def get(self, name: str) -> ToolDefinition:
        """
        Retrieve a registered tool definition by name.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found.")
        return self._tools[name]

    def list_tools(
        self,
        *,
        category: ToolCategory | None = None,
        enabled_only: bool = True,
    ) -> list[ToolDefinition]:
        """List registered tools, optionally filtered by category and enabled status."""
        results: list[ToolDefinition] = []
        for tool_def in self._tools.values():
            if enabled_only and not tool_def.metadata.enabled:
                continue
            if category is not None and tool_def.metadata.category != category:
                continue
            results.append(tool_def)
        return results

    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()


class AllowAllGate:
    """Default stub permission gate that permits all tool invocations."""

    def check(
        self,
        invocation: ToolInvocation,
        tool_metadata: ToolMetadata,
    ) -> bool:
        """Always return True to permit execution in v0.4."""
        return True


class InMemoryAuditSink:
    """In-memory audit sink that stores audit events in an inspectable list."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        """Record an audit event."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear all stored audit events."""
        self.events.clear()
