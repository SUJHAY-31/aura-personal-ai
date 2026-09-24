"""Tool execution engine orchestrating validation, permissions, and auditing."""

from __future__ import annotations

import time
from typing import Any

from backend.app.tools.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from backend.app.tools.models import (
    AuditEvent,
    ToolInvocation,
    ToolResult,
    ToolResultStatus,
)
from backend.app.tools.protocols import (
    AuditSinkProtocol,
    PermissionGateProtocol,
    ToolRegistryProtocol,
)
from backend.app.tools.registry import InMemoryAuditSink


class ToolExecutor:
    """
    Orchestrates the validate -> permission -> execute -> audit lifecycle.

    Acts as a fault boundary so tool exceptions never propagate to callers.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistryProtocol,
        permission_gate: PermissionGateProtocol | None = None,
        audit_sink: AuditSinkProtocol | None = None,
    ) -> None:
        self._registry = registry
        if permission_gate is not None:
            self._gate = permission_gate
        else:
            from backend.app.permissions.gate import create_default_permission_gate

            self._gate = create_default_permission_gate()
        self._audit = audit_sink or InMemoryAuditSink()

    def execute(self, invocation: ToolInvocation) -> ToolResult:
        """
        Execute a single tool invocation through the full lifecycle.

        1. Resolve tool definition from registry
        2. Verify tool is enabled
        3. Validate invocation arguments against schema
        4. Check permission gate
        5. Execute tool implementation
        6. Record audit event
        7. Return typed ToolResult
        """
        start_time = time.monotonic()

        # 1. Lookup tool
        try:
            tool_def = self._registry.get(invocation.tool_name)
        except ToolNotFoundError:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolResultStatus.ERROR,
                error_message=f"Tool '{invocation.tool_name}' not found.",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # 2. Check if tool is enabled
        if not tool_def.metadata.enabled:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolResultStatus.ERROR,
                error_message=f"Tool '{invocation.tool_name}' is currently disabled.",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # 3. Validate arguments
        try:
            self._validate_arguments(
                invocation.arguments, tool_def.metadata.input_schema
            )
        except ToolValidationError as exc:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolResultStatus.INVALID_INPUT,
                error_message=str(exc),
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # 4. Check permission gate
        permission_outcome = self._gate.check(invocation, tool_def.metadata)
        duration_ms = (time.monotonic() - start_time) * 1000

        if isinstance(permission_outcome, bool):
            if not permission_outcome:
                result = ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status=ToolResultStatus.DENIED,
                    error_message="Permission denied.",
                    duration_ms=duration_ms,
                    metadata={"permission_decision": "deny"},
                )
                self._record_audit(
                    invocation, result, metadata={"permission_decision": "deny"}
                )
                return result
        elif hasattr(permission_outcome, "decision"):
            from backend.app.permissions.models import PermissionDecision

            decision = permission_outcome.decision
            if decision == PermissionDecision.DENY:
                result = ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status=ToolResultStatus.DENIED,
                    error_message=permission_outcome.reason or "Permission denied.",
                    duration_ms=duration_ms,
                    metadata={
                        "permission_decision": decision.value,
                        "risk_level": tool_def.metadata.risk_level.value,
                    },
                )
                self._record_audit(
                    invocation,
                    result,
                    metadata={"permission_decision": decision.value},
                )
                return result
            if decision == PermissionDecision.REQUIRE_CONFIRMATION:
                result = ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status=ToolResultStatus.REQUIRE_CONFIRMATION,
                    error_message=permission_outcome.reason
                    or "Execution blocked pending confirmation.",
                    duration_ms=duration_ms,
                    metadata={
                        "permission_decision": decision.value,
                        "risk_level": tool_def.metadata.risk_level.value,
                        "confirmation_prompt": permission_outcome.confirmation_prompt,
                    },
                )
                self._record_audit(
                    invocation,
                    result,
                    metadata={"permission_decision": decision.value},
                )
                return result
            if decision != PermissionDecision.ALLOW:
                result = ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status=ToolResultStatus.DENIED,
                    error_message=f"Unknown permission decision '{decision}'; denied by default.",
                    duration_ms=duration_ms,
                    metadata={"permission_decision": "deny"},
                )
                self._record_audit(
                    invocation, result, metadata={"permission_decision": "deny"}
                )
                return result
        else:
            result = ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolResultStatus.DENIED,
                error_message="Invalid permission gate response; denied by default.",
                duration_ms=duration_ms,
                metadata={"permission_decision": "deny"},
            )
            self._record_audit(
                invocation, result, metadata={"permission_decision": "deny"}
            )
            return result

        # 5. Execute tool implementation
        try:
            output = tool_def.implementation.execute(invocation.arguments)
            duration_ms = (time.monotonic() - start_time) * 1000
            result = ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolResultStatus.SUCCESS,
                output=output,
                duration_ms=duration_ms,
                metadata={"permission_decision": "allow"},
            )
        except ToolExecutionError as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            result = ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolResultStatus.ERROR,
                error_message=str(exc),
                duration_ms=duration_ms,
                metadata={"permission_decision": "allow"},
            )
        except Exception:  # noqa: BLE001
            duration_ms = (time.monotonic() - start_time) * 1000
            result = ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolResultStatus.ERROR,
                error_message="Unexpected tool failure.",
                duration_ms=duration_ms,
                metadata={"permission_decision": "allow"},
            )

        # 6. Audit recording
        self._record_audit(
            invocation, result, metadata={"permission_decision": "allow"}
        )

        # 7. Return outcome
        return result

    def _validate_arguments(
        self,
        arguments: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> None:
        """
        Validate arguments against a JSON-Schema-like input_schema dictionary.

        Raises ToolValidationError on any discrepancy.
        """
        if not schema:
            return

        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in arguments:
                raise ToolValidationError(f"Missing required argument: '{field}'")

        properties = schema.get("properties", {})

        # Check allowed types
        for key, value in arguments.items():
            if key in properties:
                prop_schema = properties[key]
                expected_type = prop_schema.get("type")
                if expected_type == "string" and not isinstance(value, str):
                    raise ToolValidationError(
                        f"Argument '{key}' must be a string, got {type(value).__name__}"
                    )
                elif expected_type == "integer" and (
                    not isinstance(value, int) or isinstance(value, bool)
                ):
                    raise ToolValidationError(
                        f"Argument '{key}' must be an integer, got {type(value).__name__}"
                    )
                elif expected_type == "number" and (
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                ):
                    raise ToolValidationError(
                        f"Argument '{key}' must be a number, got {type(value).__name__}"
                    )
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ToolValidationError(
                        f"Argument '{key}' must be a boolean, got {type(value).__name__}"
                    )
                elif expected_type == "array" and not isinstance(value, (list, tuple)):
                    raise ToolValidationError(
                        f"Argument '{key}' must be an array, got {type(value).__name__}"
                    )
                elif expected_type == "object" and not isinstance(value, dict):
                    raise ToolValidationError(
                        f"Argument '{key}' must be an object, got {type(value).__name__}"
                    )

        # Check extraneous fields if additionalProperties is False
        if schema.get("additionalProperties") is False:
            for key in arguments:
                if key not in properties:
                    raise ToolValidationError(
                        f"Extraneous argument '{key}' is not allowed."
                    )

    def _record_audit(
        self,
        invocation: ToolInvocation,
        result: ToolResult,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an audit event to the configured audit sink."""
        event_metadata = dict(result.metadata)
        if metadata:
            event_metadata.update(metadata)

        event = AuditEvent(
            invocation_id=invocation.invocation_id,
            tool_name=invocation.tool_name,
            session_id=invocation.session_id,
            requested_by=invocation.requested_by,
            arguments=invocation.arguments,
            status=result.status,
            error_message=result.error_message,
            duration_ms=result.duration_ms,
            metadata=event_metadata,
        )
        self._audit.record(event)
