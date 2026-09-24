"""AURA Permission Engine and Security Layer."""

from __future__ import annotations

from backend.app.permissions.errors import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    PermissionDomainError,
    PolicyConfigurationError,
)
from backend.app.permissions.gate import (
    DenyByDefaultGate,
    PermissionEngine,
    create_default_permission_gate,
)
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

__all__ = [
    "ConfirmationRequiredError",
    "DenyByDefaultGate",
    "PermissionDecision",
    "PermissionDeniedError",
    "PermissionDomainError",
    "PermissionEngine",
    "PermissionEngineProtocol",
    "PermissionPolicy",
    "PermissionPolicyProtocol",
    "PermissionResult",
    "PolicyConfigurationError",
    "PolicyMode",
    "create_default_permission_gate",
]
