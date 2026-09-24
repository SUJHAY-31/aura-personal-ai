"""Tests for AURA v0.4 Permission Engine Core."""

from __future__ import annotations

from typing import Any

from backend.app.permissions import (
    ConfirmationRequiredError,
    DenyByDefaultGate,
    PermissionDecision,
    PermissionDeniedError,
    PermissionDomainError,
    PermissionEngine,
    PermissionEngineProtocol,
    PermissionPolicy,
    PermissionPolicyProtocol,
    PermissionResult,
    PolicyConfigurationError,
    PolicyMode,
    create_default_permission_gate,
)
from backend.app.tools import (
    InMemoryAuditSink,
    RiskLevel,
    ToolCategory,
    ToolExecutor,
    ToolInvocation,
    ToolMetadata,
    ToolRegistry,
    ToolResultStatus,
)


class MockSpyTool:
    """Mock tool that records executions and allows arbitrary risk levels."""

    def __init__(
        self,
        name: str = "mock_tool",
        risk_level: RiskLevel = RiskLevel.SAFE,
        enabled: bool = True,
        requires_confirmation: bool = False,
    ) -> None:
        self._metadata = ToolMetadata(
            name=name,
            description="Spy tool for testing execution.",
            category=ToolCategory.UTILITY,
            risk_level=risk_level,
            enabled=enabled,
            requires_confirmation=requires_confirmation,
        )
        self.execution_count = 0
        self.last_arguments: dict[str, Any] | None = None

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(self, arguments: dict[str, Any]) -> Any:
        self.execution_count += 1
        self.last_arguments = arguments
        return {"executed": True, "count": self.execution_count}


# ---------------------------------------------------------------------------
# Requirement 1: SAFE tool allowed under STRICT
# ---------------------------------------------------------------------------
def test_safe_tool_allowed_under_strict() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STRICT))
    tool = MockSpyTool(name="safe_tool", risk_level=RiskLevel.SAFE)
    invocation = ToolInvocation(tool_name="safe_tool")

    result = engine.evaluate(invocation, tool.metadata)

    assert result.decision == PermissionDecision.ALLOW
    assert result.risk_level == RiskLevel.SAFE
    assert result.tool_name == "safe_tool"
    assert "allowed" in result.reason.lower()


# ---------------------------------------------------------------------------
# Requirement 2: LOW tool confirmation under STRICT
# ---------------------------------------------------------------------------
def test_low_tool_confirmation_under_strict() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STRICT))
    tool = MockSpyTool(name="low_tool", risk_level=RiskLevel.LOW)
    invocation = ToolInvocation(tool_name="low_tool")

    result = engine.evaluate(invocation, tool.metadata)

    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert result.risk_level == RiskLevel.LOW
    assert result.confirmation_prompt is not None


# ---------------------------------------------------------------------------
# Requirement 3: HIGH tool denied under STRICT
# ---------------------------------------------------------------------------
def test_high_tool_denied_under_strict() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STRICT))
    tool = MockSpyTool(name="high_tool", risk_level=RiskLevel.HIGH)
    invocation = ToolInvocation(tool_name="high_tool")

    result = engine.evaluate(invocation, tool.metadata)

    assert result.decision == PermissionDecision.DENY
    assert result.risk_level == RiskLevel.HIGH
    assert "denied" in result.reason.lower()


# ---------------------------------------------------------------------------
# Requirement 4: CRITICAL tool denied
# ---------------------------------------------------------------------------
def test_critical_tool_denied_in_all_modes() -> None:
    tool = MockSpyTool(name="critical_tool", risk_level=RiskLevel.CRITICAL)
    invocation = ToolInvocation(tool_name="critical_tool")

    for mode in [PolicyMode.STRICT, PolicyMode.STANDARD, PolicyMode.AUTONOMOUS]:
        engine = PermissionEngine(policy=PermissionPolicy(mode=mode))
        result = engine.evaluate(invocation, tool.metadata)

        assert result.decision == PermissionDecision.DENY
        assert result.risk_level == RiskLevel.CRITICAL
        assert "critical" in result.reason.lower()


# ---------------------------------------------------------------------------
# Requirement 5: STANDARD low-risk automatic execution
# ---------------------------------------------------------------------------
def test_standard_low_risk_automatic_execution() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    safe_tool = MockSpyTool(name="safe_tool", risk_level=RiskLevel.SAFE)
    low_tool = MockSpyTool(name="low_tool", risk_level=RiskLevel.LOW)

    safe_res = engine.evaluate(ToolInvocation(tool_name="safe_tool"), safe_tool.metadata)
    low_res = engine.evaluate(ToolInvocation(tool_name="low_tool"), low_tool.metadata)

    assert safe_res.decision == PermissionDecision.ALLOW
    assert low_res.decision == PermissionDecision.ALLOW


# ---------------------------------------------------------------------------
# Requirement 6: STANDARD medium-risk confirmation
# ---------------------------------------------------------------------------
def test_standard_medium_risk_confirmation() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    medium_tool = MockSpyTool(name="medium_tool", risk_level=RiskLevel.MEDIUM)

    result = engine.evaluate(
        ToolInvocation(tool_name="medium_tool"), medium_tool.metadata
    )

    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert result.risk_level == RiskLevel.MEDIUM
    assert result.confirmation_prompt is not None


# ---------------------------------------------------------------------------
# Requirement 7: STANDARD high-risk confirmation
# ---------------------------------------------------------------------------
def test_standard_high_risk_confirmation() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    high_tool = MockSpyTool(name="high_tool", risk_level=RiskLevel.HIGH)

    result = engine.evaluate(
        ToolInvocation(tool_name="high_tool"), high_tool.metadata
    )

    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert result.risk_level == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# Requirement 8: AUTONOMOUS medium-risk behavior requires explicit policy allowance
# ---------------------------------------------------------------------------
def test_autonomous_medium_risk_requires_explicit_allowance() -> None:
    policy = PermissionPolicy(mode=PolicyMode.AUTONOMOUS)
    engine = PermissionEngine(policy=policy)
    medium_tool = MockSpyTool(name="fetch_network", risk_level=RiskLevel.MEDIUM)
    invocation = ToolInvocation(tool_name="fetch_network")

    # Without explicit allowance, it must block pending confirmation
    initial_res = engine.evaluate(invocation, medium_tool.metadata)
    assert initial_res.decision == PermissionDecision.REQUIRE_CONFIRMATION

    # Once explicitly allowed in policy, it is granted ALLOW
    policy.allow_autonomous_medium_tool("fetch_network")
    allowed_res = engine.evaluate(invocation, medium_tool.metadata)
    assert allowed_res.decision == PermissionDecision.ALLOW
    assert "explicitly permitted" in allowed_res.reason.lower()


# ---------------------------------------------------------------------------
# Requirement 9: AUTONOMOUS high-risk still requires confirmation
# ---------------------------------------------------------------------------
def test_autonomous_high_risk_still_requires_confirmation() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.AUTONOMOUS))
    high_tool = MockSpyTool(name="modify_file", risk_level=RiskLevel.HIGH)

    result = engine.evaluate(
        ToolInvocation(tool_name="modify_file"), high_tool.metadata
    )

    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert result.risk_level == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# Requirement 10: CRITICAL remains denied in AUTONOMOUS
# ---------------------------------------------------------------------------
def test_critical_remains_denied_in_autonomous() -> None:
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.AUTONOMOUS))
    critical_tool = MockSpyTool(name="shell_exec", risk_level=RiskLevel.CRITICAL)

    result = engine.evaluate(
        ToolInvocation(tool_name="shell_exec"), critical_tool.metadata
    )

    assert result.decision == PermissionDecision.DENY
    assert result.risk_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Requirement 11: Unknown/missing policy denies by default
# ---------------------------------------------------------------------------
def test_missing_or_unknown_policy_denies_by_default() -> None:
    # 1. Missing policy (None)
    engine_no_policy = PermissionEngine(policy=None)
    tool = MockSpyTool(name="any_tool", risk_level=RiskLevel.SAFE)
    res_none = engine_no_policy.evaluate(
        ToolInvocation(tool_name="any_tool"), tool.metadata
    )

    assert res_none.decision == PermissionDecision.DENY
    assert "no permission policy configured" in res_none.reason.lower()

    # 2. DenyByDefaultGate directly
    gate = DenyByDefaultGate()
    res_gate = gate.check(ToolInvocation(tool_name="any_tool"), tool.metadata)
    assert res_gate.decision == PermissionDecision.DENY

    # 3. Unknown policy mode
    policy_invalid_mode = PermissionPolicy()
    policy_invalid_mode.mode = "INVALID_MODE"  # type: ignore[assignment]
    engine_invalid = PermissionEngine(policy=policy_invalid_mode)
    res_invalid = engine_invalid.evaluate(
        ToolInvocation(tool_name="any_tool"), tool.metadata
    )
    assert res_invalid.decision == PermissionDecision.DENY


# ---------------------------------------------------------------------------
# Requirement 12: Disabled tools cannot be authorized by permission engine
# ---------------------------------------------------------------------------
def test_disabled_tools_cannot_be_authorized() -> None:
    policy = PermissionPolicy(mode=PolicyMode.STANDARD)
    policy.set_session_allowlist("session-1", {"disabled_safe_tool"})
    engine = PermissionEngine(policy=policy)

    tool = MockSpyTool(
        name="disabled_safe_tool",
        risk_level=RiskLevel.SAFE,
        enabled=False,
    )
    invocation = ToolInvocation(
        tool_name="disabled_safe_tool",
        session_id="session-1",
    )

    result = engine.evaluate(invocation, tool.metadata)

    assert result.decision == PermissionDecision.DENY
    assert "disabled" in result.reason.lower()


# ---------------------------------------------------------------------------
# Requirement 13: Session allowlist allows explicitly permitted tool
# ---------------------------------------------------------------------------
def test_session_allowlist_permits_tool() -> None:
    policy = PermissionPolicy(mode=PolicyMode.STRICT)
    policy.set_session_allowlist("session-xyz", {"write_note"})
    engine = PermissionEngine(policy=policy)

    # In STRICT mode, HIGH risk is normally DENIED
    tool = MockSpyTool(name="write_note", risk_level=RiskLevel.HIGH)

    # Different session -> denied
    res_other = engine.evaluate(
        ToolInvocation(tool_name="write_note", session_id="session-other"),
        tool.metadata,
    )
    assert res_other.decision == PermissionDecision.DENY

    # Allowed session -> allowed
    res_allowed = engine.evaluate(
        ToolInvocation(tool_name="write_note", session_id="session-xyz"),
        tool.metadata,
    )
    assert res_allowed.decision == PermissionDecision.ALLOW
    assert "session allowlist" in res_allowed.reason.lower()


# ---------------------------------------------------------------------------
# Requirement 14: Session allowlist does not bypass CRITICAL restriction
# ---------------------------------------------------------------------------
def test_session_allowlist_does_not_bypass_critical() -> None:
    policy = PermissionPolicy(mode=PolicyMode.AUTONOMOUS)
    policy.set_session_allowlist("session-root", {"danger_tool"})
    engine = PermissionEngine(policy=policy)

    critical_tool = MockSpyTool(
        name="danger_tool",
        risk_level=RiskLevel.CRITICAL,
    )
    invocation = ToolInvocation(
        tool_name="danger_tool",
        session_id="session-root",
    )

    result = engine.evaluate(invocation, critical_tool.metadata)

    assert result.decision == PermissionDecision.DENY
    assert result.risk_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Requirement 15: REQUIRE_CONFIRMATION never executes the tool
# ---------------------------------------------------------------------------
def test_require_confirmation_never_executes_tool() -> None:
    registry = ToolRegistry()
    spy_tool = MockSpyTool(name="medium_action", risk_level=RiskLevel.MEDIUM)
    registry.register(spy_tool)

    # STANDARD policy requires confirmation for MEDIUM risk
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    audit_sink = InMemoryAuditSink()
    executor = ToolExecutor(
        registry=registry,
        permission_gate=engine,
        audit_sink=audit_sink,
    )

    invocation = ToolInvocation(tool_name="medium_action")
    result = executor.execute(invocation)

    # Tool implementation must NEVER have been called
    assert spy_tool.execution_count == 0
    assert result.status == ToolResultStatus.REQUIRE_CONFIRMATION
    assert "confirmation" in (result.error_message or "").lower()
    assert result.metadata["permission_decision"] == PermissionDecision.REQUIRE_CONFIRMATION.value


# ---------------------------------------------------------------------------
# Requirement 16: DENY never executes the tool
# ---------------------------------------------------------------------------
def test_deny_never_executes_tool() -> None:
    registry = ToolRegistry()
    spy_tool = MockSpyTool(name="critical_action", risk_level=RiskLevel.CRITICAL)
    registry.register(spy_tool)

    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    executor = ToolExecutor(registry=registry, permission_gate=engine)

    invocation = ToolInvocation(tool_name="critical_action")
    result = executor.execute(invocation)

    assert spy_tool.execution_count == 0
    assert result.status == ToolResultStatus.DENIED
    assert "critical" in (result.error_message or "").lower()


# ---------------------------------------------------------------------------
# Requirement 17: ALLOW executes the tool
# ---------------------------------------------------------------------------
def test_allow_executes_tool() -> None:
    registry = ToolRegistry()
    spy_tool = MockSpyTool(name="safe_action", risk_level=RiskLevel.SAFE)
    registry.register(spy_tool)

    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    executor = ToolExecutor(registry=registry, permission_gate=engine)

    invocation = ToolInvocation(tool_name="safe_action", arguments={"val": 42})
    result = executor.execute(invocation)

    assert spy_tool.execution_count == 1
    assert spy_tool.last_arguments == {"val": 42}
    assert result.status == ToolResultStatus.SUCCESS
    assert result.output == {"executed": True, "count": 1}


# ---------------------------------------------------------------------------
# Requirement 18: Permission decision is included in audit behavior
# ---------------------------------------------------------------------------
def test_permission_decision_included_in_audit_event() -> None:
    registry = ToolRegistry()
    safe_tool = MockSpyTool(name="safe_action", risk_level=RiskLevel.SAFE)
    med_tool = MockSpyTool(name="med_action", risk_level=RiskLevel.MEDIUM)
    crit_tool = MockSpyTool(name="crit_action", risk_level=RiskLevel.CRITICAL)

    registry.register(safe_tool)
    registry.register(med_tool)
    registry.register(crit_tool)

    audit_sink = InMemoryAuditSink()
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    executor = ToolExecutor(
        registry=registry,
        permission_gate=engine,
        audit_sink=audit_sink,
    )

    # 1. ALLOW audit event
    executor.execute(ToolInvocation(tool_name="safe_action"))
    # 2. REQUIRE_CONFIRMATION audit event
    executor.execute(ToolInvocation(tool_name="med_action"))
    # 3. DENY audit event
    executor.execute(ToolInvocation(tool_name="crit_action"))

    assert len(audit_sink.events) == 3

    safe_event = audit_sink.events[0]
    assert safe_event.status == ToolResultStatus.SUCCESS
    assert safe_event.metadata.get("permission_decision") == "allow"

    med_event = audit_sink.events[1]
    assert med_event.status == ToolResultStatus.REQUIRE_CONFIRMATION
    assert med_event.metadata.get("permission_decision") == "require_confirmation"

    crit_event = audit_sink.events[2]
    assert crit_event.status == ToolResultStatus.DENIED
    assert crit_event.metadata.get("permission_decision") == "deny"


# ---------------------------------------------------------------------------
# Requirement 19: Dependencies are injectable
# ---------------------------------------------------------------------------
def test_dependencies_are_injectable() -> None:
    custom_policy = PermissionPolicy(mode=PolicyMode.STRICT)
    custom_engine = PermissionEngine(policy=custom_policy)
    custom_registry = ToolRegistry()
    custom_audit = InMemoryAuditSink()

    executor = ToolExecutor(
        registry=custom_registry,
        permission_gate=custom_engine,
        audit_sink=custom_audit,
    )

    assert executor._registry is custom_registry
    assert executor._gate is custom_engine
    assert executor._audit is custom_audit


# ---------------------------------------------------------------------------
# Domain Exceptions & Protocols check
# ---------------------------------------------------------------------------
def test_protocols_and_errors() -> None:
    policy = PermissionPolicy()
    assert isinstance(policy, PermissionPolicyProtocol)

    engine = PermissionEngine(policy=policy)
    assert isinstance(engine, PermissionEngineProtocol)

    # Test error hierarchy
    assert issubclass(PolicyConfigurationError, PermissionDomainError)
    assert issubclass(PermissionDeniedError, PermissionDomainError)
    assert issubclass(ConfirmationRequiredError, PermissionDomainError)


def test_tool_requires_confirmation_flag_honored() -> None:
    # A SAFE tool that explicitly sets requires_confirmation=True must require confirmation
    engine = PermissionEngine(policy=PermissionPolicy(mode=PolicyMode.STANDARD))
    tool = MockSpyTool(
        name="explicit_confirm_tool",
        risk_level=RiskLevel.SAFE,
        requires_confirmation=True,
    )

    result = engine.evaluate(
        ToolInvocation(tool_name="explicit_confirm_tool"), tool.metadata
    )

    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert "explicitly requires confirmation" in result.reason.lower()


def test_create_default_permission_gate_factory() -> None:
    gate = create_default_permission_gate()
    assert isinstance(gate, PermissionEngine)
    assert gate.policy is not None
    assert gate.policy.mode == PolicyMode.STANDARD

    tool = MockSpyTool(name="safe_factory_tool", risk_level=RiskLevel.SAFE)
    res = gate.evaluate(ToolInvocation(tool_name="safe_factory_tool"), tool.metadata)
    assert isinstance(res, PermissionResult)
    assert res.decision == PermissionDecision.ALLOW


def test_low_risk_requires_confirmation_overrides_session_allowlist() -> None:
    """Security rule: requires_confirmation=True cannot be bypassed by session allowlist."""
    policy = PermissionPolicy(mode=PolicyMode.STANDARD)
    policy.set_session_allowlist("session-special", {"confirm_low_tool"})
    engine = PermissionEngine(policy=policy)

    registry = ToolRegistry()
    spy_tool = MockSpyTool(
        name="confirm_low_tool",
        risk_level=RiskLevel.LOW,
        requires_confirmation=True,
    )
    registry.register(spy_tool)

    invocation = ToolInvocation(
        tool_name="confirm_low_tool",
        session_id="session-special",
    )

    # 1. Permission engine check
    res = engine.evaluate(invocation, spy_tool.metadata)
    assert res.decision == PermissionDecision.REQUIRE_CONFIRMATION

    # 2. ToolExecutor execution check
    executor = ToolExecutor(registry=registry, permission_gate=engine)
    result = executor.execute(invocation)

    # Verify tool implementation was NOT executed
    assert spy_tool.execution_count == 0
    assert result.status == ToolResultStatus.REQUIRE_CONFIRMATION


def test_critical_tool_denied_with_session_allowlist() -> None:
    """CRITICAL tools remain denied even when present in a session allowlist."""
    policy = PermissionPolicy(mode=PolicyMode.STANDARD)
    policy.set_session_allowlist("session-override", {"critical_wipe"})
    engine = PermissionEngine(policy=policy)

    registry = ToolRegistry()
    spy_tool = MockSpyTool(name="critical_wipe", risk_level=RiskLevel.CRITICAL)
    registry.register(spy_tool)

    invocation = ToolInvocation(
        tool_name="critical_wipe",
        session_id="session-override",
    )

    res = engine.evaluate(invocation, spy_tool.metadata)
    assert res.decision == PermissionDecision.DENY

    executor = ToolExecutor(registry=registry, permission_gate=engine)
    result = executor.execute(invocation)

    assert spy_tool.execution_count == 0
    assert result.status == ToolResultStatus.DENIED


def test_critical_tool_denied_with_autonomous_allow_rules() -> None:
    """CRITICAL tools cannot be bypassed by autonomous medium allow rules."""
    policy = PermissionPolicy(mode=PolicyMode.AUTONOMOUS)
    policy.allow_autonomous_medium_tool("critical_danger")
    engine = PermissionEngine(policy=policy)

    tool = MockSpyTool(name="critical_danger", risk_level=RiskLevel.CRITICAL)
    invocation = ToolInvocation(tool_name="critical_danger")

    res = engine.evaluate(invocation, tool.metadata)
    assert res.decision == PermissionDecision.DENY


def test_disabled_tool_denied_with_session_allowlist() -> None:
    """Disabled tools cannot be authorized even if present in a session allowlist."""
    policy = PermissionPolicy(mode=PolicyMode.STANDARD)
    policy.set_session_allowlist("session-dis", {"disabled_allowed_tool"})
    engine = PermissionEngine(policy=policy)

    registry = ToolRegistry()
    spy_tool = MockSpyTool(
        name="disabled_allowed_tool",
        risk_level=RiskLevel.SAFE,
        enabled=False,
    )
    registry.register(spy_tool)

    invocation = ToolInvocation(
        tool_name="disabled_allowed_tool",
        session_id="session-dis",
    )

    res = engine.evaluate(invocation, spy_tool.metadata)
    assert res.decision == PermissionDecision.DENY

    executor = ToolExecutor(registry=registry, permission_gate=engine)
    result = executor.execute(invocation)

    assert spy_tool.execution_count == 0
    assert result.status == ToolResultStatus.ERROR
    assert "disabled" in (result.error_message or "").lower()
