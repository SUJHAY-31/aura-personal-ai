"""Tests for AURA v0.4 Capability / Tool System."""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.tools import (
    AllowAllGate,
    CalculatorTool,
    CurrentTimeTool,
    EchoTool,
    InMemoryAuditSink,
    RiskLevel,
    ToolAlreadyRegisteredError,
    ToolCategory,
    ToolDefinition,
    ToolExecutionError,
    ToolExecutor,
    ToolInvocation,
    ToolMetadata,
    ToolNotFoundError,
    ToolProtocol,
    ToolRegistry,
    ToolResultStatus,
    register_builtin_tools,
)


class DummyCustomTool:
    """Mock tool for custom testing."""

    def __init__(
        self,
        name: str = "dummy_tool",
        category: ToolCategory = ToolCategory.UTILITY,
        enabled: bool = True,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._metadata = ToolMetadata(
            name=name,
            description="A dummy tool for testing.",
            category=category,
            enabled=enabled,
            input_schema=schema,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(self, arguments: dict[str, Any]) -> Any:
        return {"received": arguments}


class FailingTool:
    """Tool that raises ToolExecutionError."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="failing_tool",
            description="Always raises ToolExecutionError.",
            category=ToolCategory.UTILITY,
        )

    def execute(self, arguments: dict[str, Any]) -> Any:
        raise ToolExecutionError("Controlled tool execution failure.")


class CrashingTool:
    """Tool that raises an unexpected RuntimeError."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="crashing_tool",
            description="Always raises RuntimeError.",
            category=ToolCategory.UTILITY,
        )

    def execute(self, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("Secret internal failure.")


class DenyAllGate:
    """Permission gate that denies all requests."""

    def check(self, invocation: ToolInvocation, tool_metadata: ToolMetadata) -> bool:
        return False


# -----------------------------------------------------------------------------
# Registry Tests (1 - 10)
# -----------------------------------------------------------------------------


def test_register_tool() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    assert registry.is_registered("echo")
    tool_def = registry.get("echo")
    assert tool_def.metadata.name == "echo"
    assert tool_def.implementation is tool


def test_register_duplicate_raises() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    with pytest.raises(ToolAlreadyRegisteredError, match="already registered"):
        registry.register(tool)


def test_get_registered_tool() -> None:
    registry = ToolRegistry()
    calculator = CalculatorTool()
    registry.register(calculator)
    retrieved = registry.get("calculator")
    assert isinstance(retrieved, ToolDefinition)
    assert retrieved.metadata.name == "calculator"
    assert retrieved.implementation is calculator


def test_get_unregistered_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="not found"):
        registry.get("non_existent_tool")


def test_unregister_tool() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    assert registry.is_registered("echo")
    registry.unregister("echo")
    assert not registry.is_registered("echo")
    with pytest.raises(ToolNotFoundError):
        registry.get("echo")


def test_list_all_tools() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(CalculatorTool())
    tools = registry.list_tools()
    assert len(tools) == 2
    names = {td.metadata.name for td in tools}
    assert names == {"echo", "calculator"}


def test_list_by_category() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(
        DummyCustomTool(name="net_tool", category=ToolCategory.NETWORK)
    )

    util_tools = registry.list_tools(category=ToolCategory.UTILITY)
    assert len(util_tools) == 1
    assert util_tools[0].metadata.name == "echo"

    net_tools = registry.list_tools(category=ToolCategory.NETWORK)
    assert len(net_tools) == 1
    assert net_tools[0].metadata.name == "net_tool"

    empty_tools = registry.list_tools(category=ToolCategory.FILESYSTEM)
    assert len(empty_tools) == 0


def test_list_enabled_only() -> None:
    registry = ToolRegistry()
    registry.register(DummyCustomTool(name="active", enabled=True))
    registry.register(DummyCustomTool(name="inactive", enabled=False))

    enabled = registry.list_tools(enabled_only=True)
    assert len(enabled) == 1
    assert enabled[0].metadata.name == "active"

    all_tools = registry.list_tools(enabled_only=False)
    assert len(all_tools) == 2


def test_is_registered() -> None:
    registry = ToolRegistry()
    assert not registry.is_registered("echo")
    registry.register(EchoTool())
    assert registry.is_registered("echo")


def test_registry_clear() -> None:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    assert len(registry.list_tools()) == 3
    registry.clear()
    assert len(registry.list_tools()) == 0


# -----------------------------------------------------------------------------
# Builtin Tools Tests (11 - 16)
# -----------------------------------------------------------------------------


def test_echo_tool_execute() -> None:
    tool = EchoTool()
    result = tool.execute({"message": "Hello AURA"})
    assert result == "Hello AURA"


def test_calculator_tool_basic() -> None:
    tool = CalculatorTool()
    assert tool.execute({"expression": "2 + 3 * 4"}) == "14"
    assert tool.execute({"expression": "10 / 2"}) == "5"
    assert tool.execute({"expression": "2 ** 3"}) == "8"
    assert tool.execute({"expression": "-5 + 8"}) == "3"


def test_calculator_tool_rejects_code_injection() -> None:
    tool = CalculatorTool()
    with pytest.raises(ToolExecutionError, match="Unsupported or unsafe"):
        tool.execute({"expression": "__import__('os').system('ls')"})

    with pytest.raises(ToolExecutionError):
        tool.execute({"expression": "open('test.txt')"})

    with pytest.raises(ToolExecutionError):
        tool.execute({"expression": "eval('1+1')"})


def test_current_time_tool_default_format() -> None:
    tool = CurrentTimeTool()
    result = tool.execute({})
    assert "UTC" in result
    # format should match "%Y-%m-%d %H:%M:%S UTC"
    assert len(result.split(" ")) == 3


def test_current_time_tool_custom_format() -> None:
    tool = CurrentTimeTool()
    result = tool.execute({"format": "%Y"})
    assert result.isdigit()
    assert len(result) == 4


def test_tool_metadata_properties() -> None:
    meta = ToolMetadata(
        name="test_meta",
        description="A test tool.",
        category=ToolCategory.AUTOMATION,
        version="2.1.0",
        risk_level=RiskLevel.MEDIUM,
        requires_confirmation=True,
        enabled=True,
        input_schema={"type": "object"},
        tags=("test", "automation"),
    )
    assert meta.name == "test_meta"
    assert meta.category == ToolCategory.AUTOMATION
    assert meta.risk_level == RiskLevel.MEDIUM
    assert meta.version == "2.1.0"
    assert meta.requires_confirmation is True
    assert meta.tags == ("test", "automation")


# -----------------------------------------------------------------------------
# ToolExecutor Tests (17 - 28)
# -----------------------------------------------------------------------------


def test_executor_success_path() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    audit_sink = InMemoryAuditSink()
    executor = ToolExecutor(registry=registry, audit_sink=audit_sink)

    invocation = ToolInvocation(
        tool_name="echo",
        arguments={"message": "hello world"},
    )
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.SUCCESS
    assert result.output == "hello world"
    assert result.error_message is None
    assert result.duration_ms >= 0.0
    assert len(audit_sink.events) == 1


def test_executor_tool_not_found() -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry)

    invocation = ToolInvocation(tool_name="unknown_tool")
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.ERROR
    assert "not found" in (result.error_message or "")


def test_executor_disabled_tool() -> None:
    registry = ToolRegistry()
    registry.register(DummyCustomTool(name="disabled_tool", enabled=False))
    executor = ToolExecutor(registry=registry)

    invocation = ToolInvocation(tool_name="disabled_tool")
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.ERROR
    assert "disabled" in (result.error_message or "")


def test_executor_validation_failure() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry=registry)

    # EchoTool requires "message"
    invocation = ToolInvocation(tool_name="echo", arguments={})
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.INVALID_INPUT
    assert "Missing required argument" in (result.error_message or "")


def test_executor_type_validation_failure() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry=registry)

    # EchoTool expects message to be string, provide integer
    invocation = ToolInvocation(tool_name="echo", arguments={"message": 12345})
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.INVALID_INPUT
    assert "must be a string" in (result.error_message or "")


def test_executor_permission_denied() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    audit_sink = InMemoryAuditSink()
    executor = ToolExecutor(
        registry=registry,
        permission_gate=DenyAllGate(),
        audit_sink=audit_sink,
    )

    invocation = ToolInvocation(
        tool_name="echo",
        arguments={"message": "secret"},
    )
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.DENIED
    assert result.error_message == "Permission denied."
    assert result.output is None
    assert len(audit_sink.events) == 1
    assert audit_sink.events[0].status == ToolResultStatus.DENIED


def test_executor_tool_execution_error() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())
    executor = ToolExecutor(registry=registry)

    invocation = ToolInvocation(tool_name="failing_tool")
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.ERROR
    assert "Controlled tool execution failure" in (result.error_message or "")


def test_executor_unexpected_exception() -> None:
    registry = ToolRegistry()
    registry.register(CrashingTool())
    executor = ToolExecutor(registry=registry)

    invocation = ToolInvocation(tool_name="crashing_tool")
    result = executor.execute(invocation)

    assert result.status == ToolResultStatus.ERROR
    # Error message should be sanitized, not exposing internal exception
    assert result.error_message == "Unexpected tool failure."


def test_executor_audit_event_recorded() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    audit_sink = InMemoryAuditSink()
    executor = ToolExecutor(registry=registry, audit_sink=audit_sink)

    invocation = ToolInvocation(
        tool_name="echo",
        arguments={"message": "audit me"},
        session_id="session-123",
        requested_by="tester",
    )
    result = executor.execute(invocation)

    assert len(audit_sink.events) == 1
    event = audit_sink.events[0]
    assert event.invocation_id == invocation.invocation_id
    assert event.tool_name == "echo"
    assert event.session_id == "session-123"
    assert event.requested_by == "tester"
    assert event.arguments == {"message": "audit me"}
    assert event.status == ToolResultStatus.SUCCESS
    assert event.error_message is None
    assert event.duration_ms == result.duration_ms


def test_executor_audit_on_denial() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    audit_sink = InMemoryAuditSink()
    executor = ToolExecutor(
        registry=registry,
        permission_gate=DenyAllGate(),
        audit_sink=audit_sink,
    )

    invocation = ToolInvocation(
        tool_name="echo",
        arguments={"message": "test"},
    )
    executor.execute(invocation)

    assert len(audit_sink.events) == 1
    event = audit_sink.events[0]
    assert event.status == ToolResultStatus.DENIED
    assert event.error_message == "Permission denied."


def test_executor_duration_tracking() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry=registry)

    invocation = ToolInvocation(
        tool_name="echo",
        arguments={"message": "timing"},
    )
    result = executor.execute(invocation)
    assert result.duration_ms >= 0.0


def test_invocation_id_propagation() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry=registry)

    invocation = ToolInvocation(
        tool_name="echo",
        arguments={"message": "id-check"},
    )
    result = executor.execute(invocation)
    assert result.invocation_id == invocation.invocation_id


# -----------------------------------------------------------------------------
# Protocol and Conformance Tests (29 - 32)
# -----------------------------------------------------------------------------


def test_allow_all_gate_permits_everything() -> None:
    gate = AllowAllGate()
    invocation = ToolInvocation(tool_name="any")
    metadata = ToolMetadata(
        name="any",
        description="Any tool",
        category=ToolCategory.UTILITY,
    )
    assert gate.check(invocation, metadata) is True


def test_tool_protocol_runtime_check() -> None:
    tool = DummyCustomTool()
    assert isinstance(tool, ToolProtocol)


def test_builtin_tools_satisfy_protocol() -> None:
    assert isinstance(EchoTool(), ToolProtocol)
    assert isinstance(CalculatorTool(), ToolProtocol)
    assert isinstance(CurrentTimeTool(), ToolProtocol)


def test_builtin_registration_helper() -> None:
    registry = ToolRegistry()
    assert len(registry.list_tools()) == 0
    register_builtin_tools(registry)
    tools = registry.list_tools()
    assert len(tools) == 3
    names = {t.metadata.name for t in tools}
    assert names == {"echo", "calculator", "get_current_time"}
