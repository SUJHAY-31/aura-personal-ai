"""Protocols defining dependencies for the AURA tool system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from backend.app.permissions.models import PermissionResult

from backend.app.tools.models import (
    AuditEvent,
    ToolCategory,
    ToolDefinition,
    ToolInvocation,
    ToolMetadata,
)


@runtime_checkable
class ToolProtocol(Protocol):
    """Interface every tool must satisfy."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return immutable tool metadata."""
        ...

    def execute(self, arguments: dict[str, Any]) -> Any:
        """
        Execute the tool with validated arguments.

        Returns:
            Tool-specific output value (str, dict, number, etc.)

        Raises:
            ToolExecutionError: On any execution failure.
            ValueError: On invalid arguments that passed schema validation.
        """
        ...


@runtime_checkable
class PermissionGateProtocol(Protocol):
    """Interface for permission gates and security engines."""

    def check(
        self,
        invocation: ToolInvocation,
        tool_metadata: ToolMetadata,
    ) -> PermissionResult | bool:
        """
        Evaluate tool invocation against security policies.

        Returns:
            PermissionResult (ALLOW, DENY, REQUIRE_CONFIRMATION) or bool for legacy test doubles.
        """
        ...


@runtime_checkable
class AuditSinkProtocol(Protocol):
    """Interface for recording audit events."""

    def record(self, event: AuditEvent) -> None:
        """Persist or emit an audit event."""
        ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Interface for tool lookup and discovery."""

    def get(self, name: str) -> ToolDefinition:
        """Retrieve a registered tool by name. Raises ToolNotFoundError if missing."""
        ...

    def list_tools(
        self,
        *,
        category: ToolCategory | None = None,
        enabled_only: bool = True,
    ) -> list[ToolDefinition]:
        """List registered tools, optionally filtered."""
        ...

    def is_registered(self, name: str) -> bool:
        """Check if a tool name is registered."""
        ...
