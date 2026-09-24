"""Policy implementation governing capability access and risk evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.permissions.models import (
    PermissionDecision,
    PermissionResult,
    PolicyMode,
)
from backend.app.tools.models import RiskLevel, ToolInvocation, ToolMetadata


@dataclass
class PermissionPolicy:
    """
    Configurable security policy governing capability access.

    Attributes:
        mode: PolicyMode governing baseline evaluation (STRICT, STANDARD, AUTONOMOUS).
        autonomous_allowed_medium_tools: Tool names permitted to run automatically in AUTONOMOUS mode.
        session_allowlists: Map of session IDs to sets of allowed tool names.
    """

    mode: PolicyMode = PolicyMode.STANDARD
    autonomous_allowed_medium_tools: set[str] = field(default_factory=set)
    session_allowlists: dict[str, set[str]] = field(default_factory=dict)

    def allow_session_tool(self, session_id: str, tool_name: str) -> None:
        """Add a tool to a session's allowlist."""
        self.session_allowlists.setdefault(session_id, set()).add(tool_name)

    def set_session_allowlist(
        self, session_id: str, tools: set[str] | list[str]
    ) -> None:
        """Set the complete allowlist for a session."""
        self.session_allowlists[session_id] = set(tools)

    def clear_session_allowlist(self, session_id: str) -> None:
        """Clear allowlist for a session."""
        self.session_allowlists.pop(session_id, None)

    def allow_autonomous_medium_tool(self, tool_name: str) -> None:
        """Explicitly permit a medium-risk tool to run in AUTONOMOUS mode."""
        self.autonomous_allowed_medium_tools.add(tool_name)

    def evaluate(
        self,
        invocation: ToolInvocation,
        metadata: ToolMetadata,
    ) -> PermissionResult:
        """Evaluate a tool invocation and metadata against this policy."""
        tool_name = metadata.name
        risk = metadata.risk_level

        # 1. Disabled tools can never be authorized
        if not metadata.enabled:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason=f"Tool '{tool_name}' is disabled and cannot be authorized.",
                tool_name=tool_name,
                risk_level=risk,
            )

        # 2. CRITICAL tools are always denied; autonomous mode and session allowlists cannot override this
        if risk == RiskLevel.CRITICAL:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason=f"Tool '{tool_name}' has CRITICAL risk level and is denied.",
                tool_name=tool_name,
                risk_level=risk,
            )

        # 3. Check if tool metadata explicitly demands confirmation
        if metadata.requires_confirmation:
            return PermissionResult(
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                reason=f"Tool '{tool_name}' explicitly requires confirmation.",
                tool_name=tool_name,
                risk_level=risk,
                confirmation_prompt=f"Please confirm execution of '{tool_name}'.",
            )

        # 4. Check per-session allowlist (CRITICAL and requires_confirmation are already handled)
        if invocation.session_id:
            allowed_for_session = self.session_allowlists.get(
                invocation.session_id, set()
            )
            if tool_name in allowed_for_session:
                return PermissionResult(
                    decision=PermissionDecision.ALLOW,
                    reason=f"Tool '{tool_name}' is explicitly permitted by session allowlist for session '{invocation.session_id}'.",
                    tool_name=tool_name,
                    risk_level=risk,
                )

        # 5. Evaluate baseline by mode
        if self.mode == PolicyMode.STRICT:
            if risk == RiskLevel.SAFE:
                return PermissionResult(
                    decision=PermissionDecision.ALLOW,
                    reason=f"Safe tool '{tool_name}' is allowed under STRICT policy.",
                    tool_name=tool_name,
                    risk_level=risk,
                )
            if risk in (RiskLevel.LOW, RiskLevel.MEDIUM):
                return PermissionResult(
                    decision=PermissionDecision.REQUIRE_CONFIRMATION,
                    reason=f"{risk.value.capitalize()}-risk tool '{tool_name}' requires confirmation under STRICT policy.",
                    tool_name=tool_name,
                    risk_level=risk,
                    confirmation_prompt=f"Confirm execution of {risk.value}-risk tool '{tool_name}'.",
                )
            # HIGH
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason=f"High-risk tool '{tool_name}' is denied under STRICT policy.",
                tool_name=tool_name,
                risk_level=risk,
            )

        if self.mode == PolicyMode.STANDARD:
            if risk in (RiskLevel.SAFE, RiskLevel.LOW):
                return PermissionResult(
                    decision=PermissionDecision.ALLOW,
                    reason=f"{risk.value.capitalize()}-risk tool '{tool_name}' is allowed under STANDARD policy.",
                    tool_name=tool_name,
                    risk_level=risk,
                )
            # MEDIUM, HIGH
            return PermissionResult(
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                reason=f"{risk.value.capitalize()}-risk tool '{tool_name}' requires confirmation under STANDARD policy.",
                tool_name=tool_name,
                risk_level=risk,
                confirmation_prompt=f"Confirm execution of {risk.value}-risk tool '{tool_name}'.",
            )

        if self.mode == PolicyMode.AUTONOMOUS:
            if risk in (RiskLevel.SAFE, RiskLevel.LOW):
                return PermissionResult(
                    decision=PermissionDecision.ALLOW,
                    reason=f"{risk.value.capitalize()}-risk tool '{tool_name}' is allowed under AUTONOMOUS policy.",
                    tool_name=tool_name,
                    risk_level=risk,
                )
            if risk == RiskLevel.MEDIUM:
                if tool_name in self.autonomous_allowed_medium_tools:
                    return PermissionResult(
                        decision=PermissionDecision.ALLOW,
                        reason=f"Medium-risk tool '{tool_name}' is explicitly permitted by AUTONOMOUS policy.",
                        tool_name=tool_name,
                        risk_level=risk,
                    )
                return PermissionResult(
                    decision=PermissionDecision.REQUIRE_CONFIRMATION,
                    reason=f"Medium-risk tool '{tool_name}' requires confirmation in AUTONOMOUS mode unless explicitly permitted by policy.",
                    tool_name=tool_name,
                    risk_level=risk,
                    confirmation_prompt=f"Confirm execution of medium-risk tool '{tool_name}'.",
                )
            # HIGH
            return PermissionResult(
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                reason=f"High-risk tool '{tool_name}' requires confirmation under AUTONOMOUS policy.",
                tool_name=tool_name,
                risk_level=risk,
                confirmation_prompt=f"Confirm execution of high-risk tool '{tool_name}'.",
            )

        # Fallthrough / Unknown mode: deny by default
        return PermissionResult(
            decision=PermissionDecision.DENY,
            reason=f"Unknown policy mode '{self.mode}'; denied by default.",
            tool_name=tool_name,
            risk_level=risk,
        )
