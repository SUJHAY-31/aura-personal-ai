"""AURA Capability / Tool System."""

from __future__ import annotations

from backend.app.tools.builtins import (
    CalculatorTool,
    CurrentTimeTool,
    EchoTool,
    register_builtin_tools,
)
from backend.app.tools.errors import (
    ToolAlreadyRegisteredError,
    ToolDisabledError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
    ToolTimeoutError,
    ToolValidationError,
)
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.models import (
    AuditEvent,
    RiskLevel,
    ToolCategory,
    ToolDefinition,
    ToolInvocation,
    ToolMetadata,
    ToolResult,
    ToolResultStatus,
)
from backend.app.tools.protocols import (
    AuditSinkProtocol,
    PermissionGateProtocol,
    ToolProtocol,
    ToolRegistryProtocol,
)
from backend.app.tools.registry import (
    AllowAllGate,
    InMemoryAuditSink,
    ToolRegistry,
)

__all__ = [
    "AllowAllGate",
    "AuditEvent",
    "AuditSinkProtocol",
    "CalculatorTool",
    "CurrentTimeTool",
    "EchoTool",
    "InMemoryAuditSink",
    "PermissionGateProtocol",
    "RiskLevel",
    "ToolAlreadyRegisteredError",
    "ToolCategory",
    "ToolDefinition",
    "ToolDisabledError",
    "ToolError",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolInvocation",
    "ToolMetadata",
    "ToolNotFoundError",
    "ToolPermissionDeniedError",
    "ToolProtocol",
    "ToolRegistry",
    "ToolRegistryProtocol",
    "ToolResult",
    "ToolResultStatus",
    "ToolTimeoutError",
    "ToolValidationError",
    "register_builtin_tools",
]
