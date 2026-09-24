"""Action protocol parser converting raw LLM output into typed actions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from backend.app.orchestrator.models import (
    DirectResponseAction,
    ParsedAction,
    ParseFailureAction,
    ToolCallAction,
)
from backend.app.orchestrator.protocols import ActionProtocolParserProtocol

if TYPE_CHECKING:
    from backend.app.tools.models import ToolDefinition

OPEN_TAG = "<aura_action>"
CLOSE_TAG = "</aura_action>"


class ActionProtocolParser(ActionProtocolParserProtocol):
    """
    Parses raw language model text into structured, typed actions.

    Enforces the single-envelope <aura_action> protocol. Never interprets
    arbitrary natural language as an executable tool command.
    """

    def parse(self, raw_text: str | None) -> ParsedAction:
        """
        Convert raw LLM output into DirectResponseAction, ToolCallAction,
        or ParseFailureAction.

        Only a valid <aura_action> envelope can produce a ToolCallAction.
        """
        if raw_text is None:
            return DirectResponseAction(content="")

        text = raw_text

        count_open = text.count(OPEN_TAG)
        count_close = text.count(CLOSE_TAG)

        # Case 1: No action tags present -> Direct conversational response
        if count_open == 0 and count_close == 0:
            return DirectResponseAction(content=text.strip())

        # Case 2: Multiple action blocks detected
        if count_open > 1 or count_close > 1:
            return ParseFailureAction(
                raw_text=text,
                error_message=(
                    "Multiple action envelopes detected in a single response; "
                    "only one action per turn is permitted."
                ),
            )

        # Case 3: Mismatched or unclosed tags
        if count_open != 1 or count_close != 1:
            return ParseFailureAction(
                raw_text=text,
                error_message="Malformed action envelope: unclosed or mismatched <aura_action> tags.",
            )

        start_index = text.find(OPEN_TAG)
        end_index = text.find(CLOSE_TAG)

        if end_index < start_index:
            return ParseFailureAction(
                raw_text=text,
                error_message="Malformed action envelope: closing tag appears before opening tag.",
            )

        # Extract content within the envelope
        inner_content = text[start_index + len(OPEN_TAG) : end_index].strip()

        if not inner_content:
            return ParseFailureAction(
                raw_text=text,
                error_message="Action envelope contains empty payload.",
            )

        # Parse JSON
        try:
            payload = json.loads(inner_content)
        except json.JSONDecodeError as exc:
            return ParseFailureAction(
                raw_text=text,
                error_message=f"Invalid JSON in action envelope: {exc.msg}.",
            )

        # Must be a JSON object
        if not isinstance(payload, dict):
            return ParseFailureAction(
                raw_text=text,
                error_message="Action envelope must be a JSON object.",
            )

        # Validate 'action' field
        if "action" not in payload:
            return ParseFailureAction(
                raw_text=text,
                error_message="Missing required field 'action'.",
            )
        action_val = payload["action"]
        if action_val != "tool_call":
            return ParseFailureAction(
                raw_text=text,
                error_message=f"Invalid action '{action_val}'. Expected 'tool_call'.",
            )

        # Validate 'tool_name' field
        if "tool_name" not in payload:
            return ParseFailureAction(
                raw_text=text,
                error_message="Missing required field 'tool_name'.",
            )
        tool_name = payload["tool_name"]
        if not isinstance(tool_name, str) or not tool_name.strip():
            return ParseFailureAction(
                raw_text=text,
                error_message="Field 'tool_name' must be a non-empty string.",
            )

        # Validate 'arguments' field
        if "arguments" not in payload:
            return ParseFailureAction(
                raw_text=text,
                error_message="Missing required field 'arguments'.",
            )
        arguments = payload["arguments"]
        if not isinstance(arguments, dict):
            return ParseFailureAction(
                raw_text=text,
                error_message="Field 'arguments' must be a JSON object.",
            )

        # Optional 'intent'
        raw_intent = payload.get("intent", "")
        intent = str(raw_intent).strip() if raw_intent is not None else ""

        return ToolCallAction(
            tool_name=tool_name.strip(),
            arguments=arguments,
            intent=intent,
        )

    def format_tool_prompt(self, tools: list[ToolDefinition]) -> str:
        """
        Format registered tools and action protocol instructions for prompt inclusion.

        Does not execute any tools.
        """
        if not tools:
            return ""

        tool_descriptions: list[str] = []
        for tool_def in tools:
            meta = tool_def.metadata
            schema_str = (
                f" Parameters: {json.dumps(meta.input_schema, sort_keys=True)}"
                if meta.input_schema
                else ""
            )
            tool_descriptions.append(f"- {meta.name}: {meta.description}{schema_str}")

        tools_block = "\n".join(tool_descriptions)
        return (
            "Available Tools:\n"
            f"{tools_block}\n\n"
            "Action Protocol Instructions:\n"
            "To invoke a tool, emit a single JSON action envelope enclosed in <aura_action> tags:\n"
            "<aura_action>\n"
            "{\n"
            '  "action": "tool_call",\n'
            '  "tool_name": "<name>",\n'
            '  "arguments": { ... },\n'
            '  "intent": "<brief explanation of purpose>"\n'
            "}\n"
            "</aura_action>\n"
            "If no tool is required, respond directly with natural conversational text."
        )
