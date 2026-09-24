"""Observation sanitizer ensuring tool execution outputs are safe for LLM context."""

from __future__ import annotations

import json
import re
from typing import Any

from backend.app.orchestrator.models import Observation
from backend.app.orchestrator.protocols import ObservationSanitizerProtocol
from backend.app.tools.models import ToolResult, ToolResultStatus

DEFAULT_MAX_OBSERVATION_BYTES = 16384  # 16 KB in UTF-8 bytes
DEFAULT_MAX_OBSERVATION_LENGTH = DEFAULT_MAX_OBSERVATION_BYTES

# Defensive secret redaction patterns
# Disclaimer: This is defensive redaction, not perfect secret detection.
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"
)
BEARER_PATTERN = re.compile(
    r"""(?i)\bBearer\s+(?!\[REDACTED)[a-zA-Z0-9_\-\.+=/]{4,}"""
)
QUOTED_SECRET_PATTERN = re.compile(
    r"""(?i)(["']?\b(?:api[_-]?key|secret|password|token|auth|authorization|access[_-]?token)\b["']?\s*[:=]\s*)(["'])(?!\[REDACTED)(?:(?!\2).)*\2"""
)
UNQUOTED_SECRET_PATTERN = re.compile(
    r"""(?i)(["']?\b(?:api[_-]?key|secret|password|token|auth|authorization|access[_-]?token)\b["']?\s*[:=]\s*)(?!["'])(?!Bearer\b)(?!\[REDACTED)([^\s,;}{]+)"""
)


class ObservationSanitizer(ObservationSanitizerProtocol):
    """
    Sanitizes, redacts, truncates, and wraps untrusted ToolResult payloads.

    Contract:
    ObservationSanitizer returns a sanitized Observation.
    Observation.to_context_string() is the required method for producing the
    final untrusted wrapper before the LLM sees the content.
    The final LLM context must never receive raw ToolResult.output.
    """

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_OBSERVATION_BYTES,
        redact_secrets: bool = True,
        *,
        max_length: int | None = None,
    ) -> None:
        if max_length is not None:
            self._max_bytes = max_length
        else:
            self._max_bytes = max_bytes
        self._redact_secrets = redact_secrets

    def sanitize(self, tool_result: ToolResult) -> Observation:
        """
        Sanitize a ToolResult into an untrusted Observation container.

        Enforces byte size limits, credential redaction, boundary tag escaping,
        and safe serialization. Fails closed on any unexpected exception.
        """
        try:
            return self._do_sanitize(tool_result)
        except Exception as exc:  # noqa: BLE001
            return Observation(
                tool_name=tool_result.tool_name,
                status="error",
                content="Sanitization Failure: The output from this tool could not be safely serialized.",
                raw_length=0,
                duration_ms=tool_result.duration_ms,
                is_truncated=False,
                is_sanitization_failure=True,
                error_message=f"Sanitization error: {type(exc).__name__}",
            )

    def _do_sanitize(self, tool_result: ToolResult) -> Observation:
        status_val = (
            tool_result.status.value
            if hasattr(tool_result.status, "value")
            else str(tool_result.status)
        )

        sanitized_err: str | None = None
        if tool_result.error_message is not None:
            sanitized_err = self._sanitize_error_message(tool_result.error_message)

        # 1. Format content based on result status
        if tool_result.status != ToolResultStatus.SUCCESS:
            err_msg = sanitized_err or "Tool execution did not succeed."
            if tool_result.output is not None:
                serialized = f"Error ({status_val}): {err_msg}\nOutput: {self._serialize_value(tool_result.output)}"
            else:
                serialized = f"Error ({status_val}): {err_msg}"
        else:
            serialized = self._serialize_value(tool_result.output)

        raw_length = len(serialized.encode("utf-8"))

        # 2. Secret Redaction
        if self._redact_secrets:
            serialized = self._apply_redactions(serialized)

        # 3. Prompt Injection / Tag Breakout Defense
        serialized = self._escape_boundary_tags(serialized)

        # 4. Byte-based Truncation
        serialized, is_truncated = self._truncate_utf8(serialized, self._max_bytes)

        return Observation(
            tool_name=tool_result.tool_name,
            status=status_val,
            content=serialized,
            raw_length=raw_length,
            duration_ms=tool_result.duration_ms,
            is_truncated=is_truncated,
            is_sanitization_failure=False,
            error_message=sanitized_err,
        )

    def _sanitize_error_message(self, err_msg: str) -> str:
        """Sanitize an error message against secrets, tag breakouts, and size limits."""
        if self._redact_secrets:
            err_msg = self._apply_redactions(err_msg)
        err_msg = self._escape_boundary_tags(err_msg)
        truncated_err, _ = self._truncate_utf8(err_msg, self._max_bytes)
        return truncated_err

    def _serialize_value(self, value: Any) -> str:
        """
        Deterministically serialize tool output values.

        Primitives (str, int, float, bool, None) and containers (list, dict)
        are serialized deterministically. Unsupported custom objects are
        safely replaced with a fixed representation without calling __str__ or __repr__.
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, dict)):
            clean_value = self._sanitize_structure(value)
            return json.dumps(clean_value, indent=2, sort_keys=True)
        return "[UNSUPPORTED_TOOL_OUTPUT_TYPE]"

    def _sanitize_structure(self, obj: Any) -> Any:
        """Recursively clean dicts and lists without executing custom object code."""
        if obj is None or isinstance(obj, (int, float, str)):
            return obj
        if isinstance(obj, list):
            return [self._sanitize_structure(item) for item in obj]
        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, (str, int, float, bool)) else "[UNSUPPORTED_KEY]": self._sanitize_structure(v)
                for k, v in obj.items()
            }
        return "[UNSUPPORTED_TOOL_OUTPUT_TYPE]"

    def _apply_redactions(self, text: str) -> str:
        """
        Scrub credential and secret patterns defensively.

        Disclaimer: This is defensive redaction, not perfect secret detection.
        """
        text = PRIVATE_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY]", text)
        text = BEARER_PATTERN.sub("Bearer [REDACTED_SECRET]", text)
        text = QUOTED_SECRET_PATTERN.sub(r"\g<1>\g<2>[REDACTED_SECRET]\g<2>", text)
        text = UNQUOTED_SECRET_PATTERN.sub(r"\g<1>[REDACTED_SECRET]", text)
        return text

    def _escape_boundary_tags(self, text: str) -> str:
        """Escape boundary observation tags to prevent prompt injection breakouts."""
        text = text.replace("</observation>", "&lt;/observation&gt;")
        text = text.replace("<observation", "&lt;observation")
        return text

    def _truncate_utf8(self, text: str, max_bytes: int) -> tuple[str, bool]:
        """
        Truncate text so its UTF-8 byte length never exceeds max_bytes.

        Guarantees:
        - len(final_text.encode('utf-8')) <= max_bytes
        - Multi-byte UTF-8 sequences are never split or corrupted.
        - Truncation marker length is calculated dynamically.
        """
        encoded = text.encode("utf-8")
        total_bytes = len(encoded)
        if total_bytes <= max_bytes:
            return text, False

        omitted = total_bytes - max_bytes
        marker = f"\n[... Output truncated: {omitted} bytes omitted to protect context window ...]"
        marker_bytes = len(marker.encode("utf-8"))

        if marker_bytes >= max_bytes:
            truncated_marker = marker.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
            return truncated_marker, True

        content_budget = max_bytes - marker_bytes
        preserved_text = encoded[:content_budget].decode("utf-8", errors="ignore")
        preserved_bytes = len(preserved_text.encode("utf-8"))
        actual_omitted = total_bytes - preserved_bytes
        marker = f"\n[... Output truncated: {actual_omitted} bytes omitted to protect context window ...]"

        while (
            len(preserved_text.encode("utf-8")) + len(marker.encode("utf-8")) > max_bytes
            and content_budget > 0
        ):
            content_budget -= 1
            preserved_text = encoded[:content_budget].decode("utf-8", errors="ignore")
            preserved_bytes = len(preserved_text.encode("utf-8"))
            actual_omitted = total_bytes - preserved_bytes
            marker = f"\n[... Output truncated: {actual_omitted} bytes omitted to protect context window ...]"

        final_content = preserved_text + marker
        return final_content, True
