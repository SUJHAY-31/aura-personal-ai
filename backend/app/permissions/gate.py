"""Permission gate and engine evaluating tool capability invocations."""

from __future__ import annotations

from typing import Any

from backend.app.permissions.models import (
    PermissionDecision,
    PermissionResult,
    PolicyMode,
)
from backend.app.permissions.policy import PermissionPolicy
from backend.app.permissions.protocols import (
    PermissionEngineProtocol,
    PermissionPolicyProtocol,
)
from backend.app.tools.models import ToolInvocation, ToolMetadata


class PermissionEngine(PermissionEngineProtocol):
    """
    Security boundary evaluating tool invocations against permission policies.

    Default policy is STANDARD. If initialized with policy=None,
    all invocations are denied by default.
    """

    def __init__(
        self,
        policy: PermissionPolicyProtocol | None = None,
    ) -> None:
        self._policy = policy

    @property
    def policy(self) -> PermissionPolicyProtocol | None:
        """Return the active policy protocol instance."""
        return self._policy

    def set_policy(self, policy: PermissionPolicyProtocol | None) -> None:
        """Update or clear the active permission policy."""
        self._policy = policy

    def evaluate(
        self,
        invocation: ToolInvocation,
        metadata: ToolMetadata,
        context: dict[str, Any] | None = None,
    ) -> PermissionResult:
        """
        Evaluate a tool invocation and metadata against the active policy.

        Denies by default if no policy is configured.
        """
        if self._policy is None:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason="No permission policy configured; denied by default.",
                tool_name=metadata.name,
                risk_level=metadata.risk_level,
            )

        return self._policy.evaluate(invocation, metadata)

    def check(
        self,
        invocation: ToolInvocation,
        tool_metadata: ToolMetadata,
    ) -> PermissionResult:
        """
        Permission gate adapter checking tool invocation.

        Implements PermissionGateProtocol returning a typed PermissionResult.
        """
        return self.evaluate(invocation, tool_metadata)


class DenyByDefaultGate:
    """Safe gate that denies all tool invocations by default."""

    def check(
        self,
        invocation: ToolInvocation,
        tool_metadata: ToolMetadata,
    ) -> PermissionResult:
        return PermissionResult(
            decision=PermissionDecision.DENY,
            reason="Deny by default: no permission policy configured.",
            tool_name=tool_metadata.name,
            risk_level=tool_metadata.risk_level,
        )


def create_default_permission_gate() -> PermissionEngine:
    """Factory creating the production default permission engine with STANDARD policy."""
    return PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
