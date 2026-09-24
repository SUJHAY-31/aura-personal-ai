"""Domain models for the AURA tool system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ToolCategory(str, Enum):
    """Functional grouping for tool discovery and UI presentation."""

    UTILITY = "utility"
    FILESYSTEM = "filesystem"
    APPLICATION = "application"
    NETWORK = "network"
    AUTOMATION = "automation"
    DEVICE = "device"


class RiskLevel(str, Enum):
    """Assessed risk tier governing permission requirements."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolResultStatus(str, Enum):
    """Outcome classification of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass(frozen=True)
class ToolMetadata:
    """Immutable descriptive and policy metadata for a tool."""

    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    risk_level: RiskLevel = RiskLevel.SAFE
    requires_confirmation: bool = False
    enabled: bool = True
    input_schema: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolInvocation:
    """A request to execute a specific tool with arguments."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    requested_by: str = "orchestrator"


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a single tool execution."""

    invocation_id: str
    tool_name: str
    status: ToolResultStatus
    output: Any = None
    error_message: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditEvent:
    """Immutable record of a tool invocation for security audit and debugging."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    invocation_id: str = ""
    tool_name: str = ""
    session_id: str | None = None
    requested_by: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    error_message: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Composite: bundles metadata with the executable implementation reference.
# Uses TYPE_CHECKING to avoid circular imports with protocols.
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.tools.protocols import ToolProtocol


@dataclass
class ToolDefinition:
    """A registered tool: metadata paired with its executable implementation."""

    metadata: ToolMetadata
    implementation: ToolProtocol
