"""Domain exceptions for the AURA permission system."""

from __future__ import annotations


class PermissionDomainError(Exception):
    """Base exception for all permission-related domain errors."""


class PolicyConfigurationError(PermissionDomainError):
    """Raised when permission policy configuration is invalid or missing."""


class PermissionDeniedError(PermissionDomainError):
    """Raised when an operation is explicitly denied by permission policy."""


class ConfirmationRequiredError(PermissionDomainError):
    """Raised when an operation requires explicit confirmation before proceeding."""
