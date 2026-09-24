"""Tool loop controller coordinating LLM reasoning, parsing, permissions, execution, and sanitization."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from backend.app.orchestrator.errors import (
    LoopLimitExceededError,
    TurnCancelledError,
    TurnTimeoutError,
)
from backend.app.orchestrator.models import (
    DirectResponseAction,
    LoopState,
    Observation,
    OrchestratorResult,
    ParseFailureAction,
    StepRecord,
    ToolCallAction,
    ToolLoopConfig,
)
from backend.app.orchestrator.protocols import (
    ActionProtocolParserProtocol,
    LLMServiceProtocol,
    ObservationSanitizerProtocol,
    ToolLoopControllerProtocol,
)
from backend.app.permissions.models import PermissionDecision
from backend.app.permissions.protocols import PermissionEngineProtocol
from backend.app.tools.models import ToolInvocation, ToolResultStatus
from backend.app.tools.protocols import ToolRegistryProtocol

if TYPE_CHECKING:
    import threading

    from backend.app.models.memory import ConversationTurn
    from backend.app.tools.executor import ToolExecutor


def compute_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Compute a deterministic canonical SHA-256 signature for a tool call.

    Recursively sorts dictionary keys, preserves list order, and serializes
    as compact canonical JSON.
    """

    def _normalize(val: Any) -> Any:
        if isinstance(val, dict):
            return {k: _normalize(val[k]) for k in sorted(val.keys())}
        if isinstance(val, list):
            return [_normalize(item) for item in val]
        return val

    normalized_args = _normalize(arguments)
    payload = {
        "arguments": normalized_args,
        "tool_name": tool_name,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class ToolLoopController(ToolLoopControllerProtocol):
    """
    Bounded multi-step reasoning loop coordinating:
    LLM -> ActionProtocolParser -> ToolRegistry -> PermissionEngine -> ToolExecutor -> ObservationSanitizer -> LLM.

    Security Invariants:
    - LLM proposes actions, but can NEVER execute actions directly.
    - ToolRegistry determines what tools exist and whether they are enabled.
    - PermissionEngine determines what is allowed (ALLOW, DENY, REQUIRE_CONFIRMATION).
    - ToolExecutor performs only approved tool execution.
    - ObservationSanitizer determines what tool output can reach the LLM context.
    - No component may bypass these boundaries.
    - No chain-of-thought or hidden reasoning traces are captured or stored.

    Cooperative Deadline Semantics:
    - Deadlines are cooperative orchestration boundaries checked before starting operations.
    - Checks occur before starting each LLM inference and before starting tool execution.
    - Because current LLMService.generate() and ToolExecutor APIs are synchronous,
      an in-flight operation cannot be forcibly killed or aborted asynchronously.
    - Once a deadline expires, NO NEW LLM generation or tool execution will begin.
    - Future revisions of LLMService and ToolExecutor may accept explicit timeout
      or cancellation tokens for fine-grained per-operation interruption.

    Observation Distinction:
    - ToolResult.output is untrusted external tool data and MUST be sanitized through ObservationSanitizer.
    - Controller-generated observations (parser errors, tool missing/disabled, duplicate blocked)
      represent trusted internal AURA control metadata and use safe deterministic strings.
    """

    def __init__(
        self,
        *,
        llm_service: LLMServiceProtocol,
        parser: ActionProtocolParserProtocol,
        tool_registry: ToolRegistryProtocol,
        permission_engine: PermissionEngineProtocol,
        tool_executor: ToolExecutor,
        sanitizer: ObservationSanitizerProtocol,
        config: ToolLoopConfig | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._parser = parser
        self._tool_registry = tool_registry
        self._permission_engine = permission_engine
        self._tool_executor = tool_executor
        self._sanitizer = sanitizer
        self._config = config or ToolLoopConfig()

    def run_loop(
        self,
        *,
        session_id: str,
        user_message: str,
        history: list[ConversationTurn],
        cancellation_token: threading.Event | None = None,
        timeout_seconds: float | None = None,
    ) -> OrchestratorResult:
        """
        Execute the bounded multi-step tool reasoning loop.

        Deadlines and cancellations are cooperatively checked around every
        LLM generation and tool execution boundary.
        """
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._config.overall_timeout_seconds
        )
        deadline = time.monotonic() + timeout

        state = LoopState(
            iteration_count=0,
            max_iterations=self._config.max_iterations,
            deadline_monotonic=deadline,
            cancellation_token=cancellation_token,
        )

        while True:
            # 1. Iteration limit check
            if state.iteration_count >= state.max_iterations:
                # If we have collected useful observations, attempt best-effort final synthesis
                if state.observations:
                    return self._attempt_final_synthesis(
                        session_id=session_id,
                        user_message=user_message,
                        history=history,
                        state=state,
                    )
                raise LoopLimitExceededError(
                    f"Reasoning loop exceeded maximum iterations ({state.max_iterations}).",
                    state=state,
                )

            # 2. Cancellation check before LLM call
            self._check_cancellation(state)

            # 3. Deadline check before LLM call
            self._check_deadline(state)

            # 4. Construct prompt and generate LLM completion
            prompt = self._build_turn_prompt(user_message, history, state.observations)
            raw_response = self._llm_service.generate(prompt)

            state.iteration_count += 1

            # 5. Parse action envelope
            action = self._parser.parse(raw_response)

            # 6. Branch based on action classification
            if isinstance(action, DirectResponseAction):
                state.step_records.append(
                    StepRecord(
                        iteration=state.iteration_count,
                        action=action,
                    )
                )
                return OrchestratorResult(
                    response=action.content,
                    session_id=session_id,
                    turns_count=state.iteration_count,
                    metadata={
                        "iterations": state.iteration_count,
                        "step_records": state.step_records,
                        "observations_count": len(state.observations),
                    },
                )

            if isinstance(action, ParseFailureAction):
                # Recoverable parse failure: record safe observation and allow model to retry
                err_obs = Observation(
                    tool_name="parser",
                    status="error",
                    content=(
                        f"Action Protocol Error: {action.error_message}. "
                        "Format your tool call as valid JSON inside an <aura_action>...</aura_action> block "
                        "or provide a direct response."
                    ),
                    raw_length=0,
                    is_truncated=False,
                    is_sanitization_failure=False,
                    error_message=action.error_message,
                )
                state.observations.append(err_obs)
                state.step_records.append(
                    StepRecord(
                        iteration=state.iteration_count,
                        action=action,
                        observation=err_obs,
                    )
                )
                continue

            if isinstance(action, ToolCallAction):
                # Cooperative checks before tool operations
                self._check_cancellation(state)
                self._check_deadline(state)

                # Duplicate call protection
                sig = compute_call_signature(action.tool_name, action.arguments)
                dup_count = state.call_signatures.get(sig, 0)

                if dup_count >= self._config.duplicate_call_threshold:
                    state.call_signatures[sig] = dup_count + 1
                    blocked_obs = Observation(
                        tool_name=action.tool_name,
                        status="error",
                        content=(
                            f"Duplicate Call Blocked: Tool '{action.tool_name}' with identical arguments "
                            f"has already been called {dup_count} times. "
                            "You must proceed to final response or call a different tool."
                        ),
                        raw_length=0,
                        is_truncated=False,
                        is_sanitization_failure=False,
                        error_message="Duplicate call threshold exceeded",
                    )
                    state.observations.append(blocked_obs)
                    state.step_records.append(
                        StepRecord(
                            iteration=state.iteration_count,
                            action=action,
                            observation=blocked_obs,
                            intent=action.intent,
                        )
                    )
                    continue

                if dup_count == 1 and sig in state.cached_results:
                    state.call_signatures[sig] = dup_count + 1
                    cached = state.cached_results[sig]
                    repeat_obs = Observation(
                        tool_name=action.tool_name,
                        status=cached.status,
                        content=f"[Duplicate Call Cached Result]\n{cached.content}",
                        raw_length=cached.raw_length,
                        duration_ms=0.0,
                        is_truncated=cached.is_truncated,
                        is_sanitization_failure=False,
                        error_message=cached.error_message,
                    )
                    state.observations.append(repeat_obs)
                    state.step_records.append(
                        StepRecord(
                            iteration=state.iteration_count,
                            action=action,
                            observation=repeat_obs,
                            intent=action.intent,
                        )
                    )
                    continue

                # Tool resolution in registry
                if not self._tool_registry.is_registered(action.tool_name):
                    missing_obs = Observation(
                        tool_name=action.tool_name,
                        status="error",
                        content=f"ToolNotFoundError: Tool '{action.tool_name}' is not registered.",
                        raw_length=0,
                        is_truncated=False,
                        is_sanitization_failure=False,
                        error_message=f"Tool '{action.tool_name}' not found",
                    )
                    state.observations.append(missing_obs)
                    state.step_records.append(
                        StepRecord(
                            iteration=state.iteration_count,
                            action=action,
                            observation=missing_obs,
                            intent=action.intent,
                        )
                    )
                    continue

                tool_def = self._tool_registry.get(action.tool_name)
                if not tool_def.metadata.enabled:
                    disabled_obs = Observation(
                        tool_name=action.tool_name,
                        status="error",
                        content=f"ToolDisabledError: Tool '{action.tool_name}' is currently disabled.",
                        raw_length=0,
                        is_truncated=False,
                        is_sanitization_failure=False,
                        error_message=f"Tool '{action.tool_name}' is disabled",
                    )
                    state.observations.append(disabled_obs)
                    state.step_records.append(
                        StepRecord(
                            iteration=state.iteration_count,
                            action=action,
                            observation=disabled_obs,
                            intent=action.intent,
                        )
                    )
                    continue

                # Prepare invocation
                invocation = ToolInvocation(
                    tool_name=action.tool_name,
                    arguments=action.arguments,
                    session_id=session_id,
                    invocation_id=action.call_id,
                )

                # Permission evaluation
                perm_result = self._permission_engine.evaluate(invocation, tool_def.metadata)

                if perm_result.decision == PermissionDecision.DENY:
                    denied_obs = Observation(
                        tool_name=action.tool_name,
                        status="denied",
                        content=f"Permission Denied: {perm_result.reason}",
                        raw_length=0,
                        is_truncated=False,
                        is_sanitization_failure=False,
                        error_message=perm_result.reason,
                    )
                    state.observations.append(denied_obs)
                    state.step_records.append(
                        StepRecord(
                            iteration=state.iteration_count,
                            action=action,
                            observation=denied_obs,
                            intent=action.intent,
                        )
                    )
                    continue

                if perm_result.decision == PermissionDecision.REQUIRE_CONFIRMATION:
                    state.pending_action = action
                    state.is_paused = True
                    conf_prompt = (
                        perm_result.confirmation_prompt
                        or f"Action '{action.tool_name}' requires explicit confirmation."
                    )
                    state.step_records.append(
                        StepRecord(
                            iteration=state.iteration_count,
                            action=action,
                            observation=None,
                            intent=action.intent,
                        )
                    )
                    return OrchestratorResult(
                        response=conf_prompt,
                        session_id=session_id,
                        turns_count=state.iteration_count,
                        metadata={
                            "iterations": state.iteration_count,
                            "step_records": state.step_records,
                            "observations_count": len(state.observations),
                        },
                        requires_confirmation=True,
                        confirmation_prompt=conf_prompt,
                        pending_action=action,
                    )

                if perm_result.decision == PermissionDecision.ALLOW:
                    # Check cancellation before tool execution
                    self._check_cancellation(state)
                    # Check deadline before tool execution
                    self._check_deadline(state)

                    # Execute approved tool through ToolExecutor
                    tool_result = self._tool_executor.execute(invocation)

                    # Sanitize through ObservationSanitizer
                    obs = self._sanitizer.sanitize(tool_result)

                    state.observations.append(obs)
                    state.executed_invocation_ids.append(invocation.invocation_id)
                    state.call_signatures[sig] = dup_count + 1
                    if tool_result.status == ToolResultStatus.SUCCESS:
                        state.cached_results[sig] = obs

                    state.step_records.append(
                        StepRecord(
                            iteration=state.iteration_count,
                            action=action,
                            observation=obs,
                            intent=action.intent,
                        )
                    )

                    # Check cancellation between tool execution and next iteration
                    self._check_cancellation(state)
                    continue

    def _attempt_final_synthesis(
        self,
        *,
        session_id: str,
        user_message: str,
        history: list[ConversationTurn],
        state: LoopState,
    ) -> OrchestratorResult:
        """
        Perform a best-effort final synthesis when max_iterations is reached.

        Never initiates another tool call. If the model produces a direct response,
        returns that response. If the model insists on further tool calls or cannot
        safely synthesize, raises LoopLimitExceededError.
        """
        self._check_cancellation(state)
        self._check_deadline(state)

        synthesis_prompt = self._build_turn_prompt(
            user_message,
            history,
            state.observations,
            final_synthesis=True,
        )
        raw_response = self._llm_service.generate(synthesis_prompt)
        action = self._parser.parse(raw_response)

        if isinstance(action, DirectResponseAction):
            state.step_records.append(
                StepRecord(
                    iteration=state.iteration_count + 1,
                    action=action,
                )
            )
            return OrchestratorResult(
                response=action.content,
                session_id=session_id,
                turns_count=state.iteration_count,
                metadata={
                    "iterations": state.iteration_count,
                    "step_records": state.step_records,
                    "observations_count": len(state.observations),
                    "final_synthesis_applied": True,
                },
            )

        raise LoopLimitExceededError(
            f"Reasoning loop reached maximum iterations ({state.max_iterations}) and could not synthesize a final response.",
            state=state,
        )

    def _build_turn_prompt(
        self,
        user_message: str,
        history: list[ConversationTurn],
        observations: list[Observation],
        *,
        final_synthesis: bool = False,
    ) -> str:
        """
        Assemble the turn prompt for the LLM using registered tools and sanitized observations.

        Distinction:
        - ToolResult.output is untrusted external data and MUST be sanitized through ObservationSanitizer.
        - Controller-generated observations are trusted AURA control metadata formatted as safe deterministic strings.
        """
        sections: list[str] = []

        if not final_synthesis:
            registered_tools = self._tool_registry.list_tools()
            tool_prompt = self._parser.format_tool_prompt(registered_tools)
            if tool_prompt:
                sections.append(tool_prompt)
        else:
            sections.append(
                "Instructions: Maximum reasoning steps reached. "
                "Synthesize a final direct response to the user based on the observations collected so far. "
                "Do NOT request any further tool actions."
            )

        if history:
            history_blocks = ["Conversation History:"]
            for turn in history:
                role_label = turn.role.capitalize()
                history_blocks.append(f"{role_label}: {turn.content}")
            sections.append("\n".join(history_blocks))

        sections.append(f"User: {user_message}")

        if observations:
            obs_blocks = ["Tool Execution Observations:"]
            for obs in observations:
                obs_blocks.append(obs.to_context_string())
            sections.append("\n".join(obs_blocks))

        return "\n\n".join(sections)

    def _check_cancellation(self, state: LoopState) -> None:
        """
        Check cancellation token cooperatively.

        Raises TurnCancelledError if the token is set.
        Note: This is a cooperative check before beginning a new operation;
        in-flight synchronous calls are not forcibly aborted.
        """
        if state.cancellation_token is not None and state.cancellation_token.is_set():
            raise TurnCancelledError("Turn execution was cancelled by token.", state=state)

    def _check_deadline(self, state: LoopState) -> None:
        """
        Check monotonic deadline cooperatively.

        Raises TurnTimeoutError if the monotonic deadline has expired.
        Note: This is a cooperative check before beginning a new operation;
        in-flight synchronous calls are not forcibly aborted.
        """
        if time.monotonic() >= state.deadline_monotonic:
            raise TurnTimeoutError("Turn execution exceeded overall deadline.", state=state)
