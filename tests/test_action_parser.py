"""Tests for AURA v0.5 Action Protocol Parser."""

from __future__ import annotations

from typing import Any

from backend.app.orchestrator import (
    ActionProtocolParser,
    ActionProtocolParserProtocol,
    ActionType,
    DirectResponseAction,
    ParseFailureAction,
    ProtocolParseError,
    ToolCallAction,
)
from backend.app.tools import (
    CalculatorTool,
    CurrentTimeTool,
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
)


class DummyExecutableTool:
    """Mock tool that records whether execute() was invoked."""

    def __init__(self) -> None:
        self.metadata = ToolMetadata(
            name="dummy_tool",
            description="Tool to ensure parser never executes code.",
            category=ToolCategory.UTILITY,
        )
        self.was_executed = False

    def execute(self, arguments: dict[str, Any]) -> Any:
        self.was_executed = True
        return "executed"


# ---------------------------------------------------------------------------
# 1. Plain direct response
# ---------------------------------------------------------------------------
def test_plain_direct_response() -> None:
    parser = ActionProtocolParser()
    raw = "Hello AURA! How are you today?"

    action = parser.parse(raw)

    assert isinstance(action, DirectResponseAction)
    assert action.content == "Hello AURA! How are you today?"


# ---------------------------------------------------------------------------
# 2. Valid tool action
# ---------------------------------------------------------------------------
def test_valid_tool_action() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        "{\n"
        '  "action": "tool_call",\n'
        '  "tool_name": "current_time",\n'
        '  "arguments": {"format": "%Y-%m-%d"},\n'
        '  "intent": "Retrieve local server timestamp"\n'
        "}\n"
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ToolCallAction)
    assert action.tool_name == "current_time"
    assert action.arguments == {"format": "%Y-%m-%d"}
    assert action.intent == "Retrieve local server timestamp"
    assert len(action.call_id) > 0


# ---------------------------------------------------------------------------
# 3. Natural language before action
# ---------------------------------------------------------------------------
def test_natural_language_before_action() -> None:
    parser = ActionProtocolParser()
    raw = (
        "I will check the current time for you right now.\n"
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "current_time", "arguments": {}}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ToolCallAction)
    assert action.tool_name == "current_time"
    assert action.arguments == {}


# ---------------------------------------------------------------------------
# 4. Natural language after action
# ---------------------------------------------------------------------------
def test_natural_language_after_action() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "current_time", "arguments": {}}\n'
        "</aura_action>\n"
        "Please wait while I retrieve this data."
    )

    action = parser.parse(raw)

    assert isinstance(action, ToolCallAction)
    assert action.tool_name == "current_time"
    assert action.arguments == {}


# ---------------------------------------------------------------------------
# 5. Whitespace around tags
# ---------------------------------------------------------------------------
def test_whitespace_around_tags() -> None:
    parser = ActionProtocolParser()
    raw = (
        "   \n\t  <aura_action>   \n"
        '  { "action": "tool_call", "tool_name": "calculator", "arguments": {"expression": "2+2"} }  \n'
        "   </aura_action>   \n\t "
    )

    action = parser.parse(raw)

    assert isinstance(action, ToolCallAction)
    assert action.tool_name == "calculator"
    assert action.arguments == {"expression": "2+2"}


# ---------------------------------------------------------------------------
# 6. Malformed JSON
# ---------------------------------------------------------------------------
def test_malformed_json_returns_parse_failure() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "calc", "arguments": { broken_json... }\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert action.raw_text == raw
    assert "Invalid JSON" in action.error_message


# ---------------------------------------------------------------------------
# 7. Missing action field
# ---------------------------------------------------------------------------
def test_missing_action_field() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"tool_name": "current_time", "arguments": {}}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert "Missing required field 'action'" in action.error_message


# ---------------------------------------------------------------------------
# 8. Wrong action value
# ---------------------------------------------------------------------------
def test_wrong_action_value() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "execute_shell", "tool_name": "current_time", "arguments": {}}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert "Invalid action 'execute_shell'" in action.error_message


# ---------------------------------------------------------------------------
# 9. Missing tool_name
# ---------------------------------------------------------------------------
def test_missing_tool_name() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "arguments": {}}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert "Missing required field 'tool_name'" in action.error_message


# ---------------------------------------------------------------------------
# 10. Empty tool_name
# ---------------------------------------------------------------------------
def test_empty_tool_name() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "   ", "arguments": {}}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert "non-empty string" in action.error_message


# ---------------------------------------------------------------------------
# 11. Missing arguments
# ---------------------------------------------------------------------------
def test_missing_arguments() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "current_time"}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert "Missing required field 'arguments'" in action.error_message


# ---------------------------------------------------------------------------
# 12. Arguments not object
# ---------------------------------------------------------------------------
def test_arguments_not_object() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "current_time", "arguments": "invalid_string"}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert "Field 'arguments' must be a JSON object" in action.error_message


# ---------------------------------------------------------------------------
# 13. Multiple action envelopes
# ---------------------------------------------------------------------------
def test_multiple_action_envelopes_rejected() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "tool_one", "arguments": {}}\n'
        "</aura_action>\n"
        "And also another tool:\n"
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "tool_two", "arguments": {}}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ParseFailureAction)
    assert "Multiple action envelopes detected" in action.error_message


# ---------------------------------------------------------------------------
# 14. Empty input
# ---------------------------------------------------------------------------
def test_empty_input_handled_cleanly() -> None:
    parser = ActionProtocolParser()

    action_empty = parser.parse("")
    assert isinstance(action_empty, DirectResponseAction)
    assert action_empty.content == ""

    action_whitespace = parser.parse("   \n\t  ")
    assert isinstance(action_whitespace, DirectResponseAction)
    assert action_whitespace.content == ""

    action_none = parser.parse(None)
    assert isinstance(action_none, DirectResponseAction)
    assert action_none.content == ""


# ---------------------------------------------------------------------------
# 15. Action with nested arguments
# ---------------------------------------------------------------------------
def test_action_with_nested_arguments() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        "{\n"
        '  "action": "tool_call",\n'
        '  "tool_name": "complex_tool",\n'
        '  "arguments": {\n'
        '    "query": "SELECT *",\n'
        '    "filters": {"active": True, "ids": [1, 2, 3]},\n'
        '    "options": {"retry": 3}\n'
        "  }\n"
        "}\n"
        "</aura_action>"
    ).replace("True", "true")

    action = parser.parse(raw)

    assert isinstance(action, ToolCallAction)
    assert action.tool_name == "complex_tool"
    assert action.arguments["filters"]["ids"] == [1, 2, 3]
    assert action.arguments["options"]["retry"] == 3


# ---------------------------------------------------------------------------
# 16. Deterministic parse result
# ---------------------------------------------------------------------------
def test_deterministic_parse_result() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        '{"action": "tool_call", "tool_name": "calculator", "arguments": {"expression": "10 * 5"}}\n'
        "</aura_action>"
    )

    action1 = parser.parse(raw)
    action2 = parser.parse(raw)

    assert isinstance(action1, ToolCallAction)
    assert isinstance(action2, ToolCallAction)
    assert action1.tool_name == action2.tool_name
    assert action1.arguments == action2.arguments
    assert action1.intent == action2.intent


# ---------------------------------------------------------------------------
# 17. Intent field preserved
# ---------------------------------------------------------------------------
def test_intent_field_preserved() -> None:
    parser = ActionProtocolParser()
    raw = (
        "<aura_action>\n"
        "{\n"
        '  "action": "tool_call",\n'
        '  "tool_name": "echo",\n'
        '  "arguments": {"message": "hello"},\n'
        '  "intent": "Echoing greeting back to caller"\n'
        "}\n"
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ToolCallAction)
    assert action.intent == "Echoing greeting back to caller"


# ---------------------------------------------------------------------------
# 18. Only <aura_action> can trigger ToolCallAction
# ---------------------------------------------------------------------------
def test_only_aura_action_can_trigger_tool_call() -> None:
    parser = ActionProtocolParser()
    # Fake tags or alternative tags
    raw = (
        "<tool_call>\n"
        '{"action": "tool_call", "tool_name": "current_time", "arguments": {}}\n'
        "</tool_call>"
    )

    action = parser.parse(raw)

    # Must NOT parse as ToolCallAction!
    assert isinstance(action, DirectResponseAction)
    assert "<tool_call>" in action.content


# ---------------------------------------------------------------------------
# 19. Arbitrary text MUST remain DirectResponseAction
# ---------------------------------------------------------------------------
def test_arbitrary_natural_language_remains_direct_response() -> None:
    parser = ActionProtocolParser()
    phrases = [
        "Run calculator with 2+2",
        "Please execute tool 'current_time'",
        "tool_call: calculator(expression='4+4')",
        "Can you call current_time for me?",
        "```json\n{\"action\": \"tool_call\", \"tool_name\": \"echo\"}\n```",
    ]

    for phrase in phrases:
        action = parser.parse(phrase)
        assert isinstance(action, DirectResponseAction)
        assert action.content == phrase


# ---------------------------------------------------------------------------
# 20. Parser performs no tool execution
# ---------------------------------------------------------------------------
def test_parser_performs_no_tool_execution() -> None:
    parser = ActionProtocolParser()
    dummy = DummyExecutableTool()

    raw = (
        "<aura_action>\n"
        f'{{"action": "tool_call", "tool_name": "{dummy.metadata.name}", "arguments": {{"val": 123}}}}\n'
        "</aura_action>"
    )

    action = parser.parse(raw)

    assert isinstance(action, ToolCallAction)
    # Ensure tool execute was never called during parsing
    assert dummy.was_executed is False


# ---------------------------------------------------------------------------
# Additional coverage: format_tool_prompt & protocol runtime checks
# ---------------------------------------------------------------------------
def test_format_tool_prompt_and_protocols() -> None:
    parser = ActionProtocolParser()
    assert isinstance(parser, ActionProtocolParserProtocol)

    calc = CalculatorTool()
    clock = CurrentTimeTool()
    tools = [
        ToolDefinition(metadata=calc.metadata, implementation=calc),
        ToolDefinition(metadata=clock.metadata, implementation=clock),
    ]

    prompt = parser.format_tool_prompt(tools)

    assert "Available Tools:" in prompt
    assert "calculator:" in prompt
    assert "current_time:" in prompt
    assert "<aura_action>" in prompt
    assert "tool_call" in prompt

    # Empty tools
    assert parser.format_tool_prompt([]) == ""

    # ProtocolParseError is subclass of OrchestratorError
    assert issubclass(ProtocolParseError, Exception)
    assert ActionType.TOOL_CALL == "tool_call"
