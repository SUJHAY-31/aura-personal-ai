"""Protocols defining interfaces for the AURA permission engine."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from backend.app.permissions.models import PermissionResult, PolicyMode
from backend.app.tools.models import ToolInvocation, ToolMetadata


@runtime_checkable
class PermissionPolicyProtocol(Protocol):
    """Interface for permission policy strategy."""

    @property
    def mode(self) -> PolicyMode:
        """Return the operational mode of the policy."""
        ...

    def evaluate(
        self,
        invocation: ToolInvocation,
        metadata: ToolMetadata,
    ) -> PermissionResult:
        """Evaluate invocation and tool metadata against the policy."""
        ...


@runtime_checkable
class PermissionEngineProtocol(Protocol):
    """Interface for the core permission evaluation engine."""

    def evaluate(
        self,
        invocation: ToolInvocation,
        metadata: ToolMetadata,
    ) -> PermissionResult:
        """Evaluate a tool invocation and return a typed PermissionResult."""
        ...

    def check(
        self,
        invocation: ToolInvocation,
        tool_metadata: ToolMetadata,
    ) -> PermissionResult:
        """Compatibility method for permission gate protocol."""
        ...
