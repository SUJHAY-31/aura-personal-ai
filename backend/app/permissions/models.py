"""Domain models for the AURA permission system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.app.tools.models import RiskLevel


class PermissionDecision(str, Enum):
    """Explicit decision made by the permission engine."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"


class PolicyMode(str, Enum):
    """Operational mode defining how capability permissions are enforced."""

    STRICT = "strict"
    STANDARD = "standard"
    AUTONOMOUS = "autonomous"


@dataclass(frozen=True)
class PermissionResult:
    """Typed outcome of a permission check evaluation."""

    decision: PermissionDecision
    reason: str
    tool_name: str
    risk_level: RiskLevel
    confirmation_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
