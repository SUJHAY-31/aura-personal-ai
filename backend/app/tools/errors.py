"""Exceptions raised by the AURA tool system."""

from __future__ import annotations


class ToolError(Exception):
    """Base exception for all tool-system failures."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool name is not present in the registry."""


class ToolAlreadyRegisteredError(ToolError):
    """Raised when attempting to register a tool with a name that already exists."""


class ToolDisabledError(ToolError):
    """Raised when a tool exists but is currently disabled."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail schema validation."""


class ToolExecutionError(ToolError):
    """Raised by a tool implementation when execution encounters a controlled error."""


class ToolPermissionDeniedError(ToolError):
    """Raised when the permission gate refuses a tool invocation."""


class ToolTimeoutError(ToolError):
    """Raised when tool execution exceeds its time budget."""
