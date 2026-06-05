"""Streaming turn events for the chat UI, driven by the LangGraph agent."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent.graph import get_graph
from agent.intent import IntentResult, classify_intent
from agent.pipeline import TurnResult, build_initial_state, state_to_result
from agent.state import AgentState
from agent.tool_trace import TOOL_LABELS
from guardrails import check_pre_input
from guardrails.types import GuardrailReport
from scoring.trustworthiness import compute_turn_trustworthiness
from observability.langfuse_integration import langfuse_enabled
from observability.turn_log import log_turn_summary
from observability.turn_trace import stream_compliance_turn

StreamKind = Literal[
    "intent",
    "phase",
    "blocked",
    "tool_start",
    "tool_end",
    "answer_start",
    "token",
    "done",
]


@dataclass
class StreamEvent:
    """One live event emitted while a turn is in progress.

    Attributes
    ----------
    kind : StreamKind
        Event type discriminator.
    intent : IntentResult or None
        Populated when ``kind`` is ``"intent"``.
    phase : str or None
        Human-readable pipeline phase label.
    tool_name : str or None
        Internal tool identifier.
    tool_label : str or None
        Display label for the tool.
    token : str or None
        Answer token fragment when ``kind`` is ``"token"``.
    result : TurnResult or None
        Final turn result when ``kind`` is ``"done"`` or ``"blocked"``.
    """

    kind: StreamKind
    intent: IntentResult | None = None
    phase: str | None = None
    tool_name: str | None = None
    tool_label: str | None = None
    token: str | None = None
    result: TurnResult | None = None


def _tool_label(name: str) -> str:
    """Return a display label for a tool name."""
    return TOOL_LABELS.get(name, name.replace("_", " ").title())


def stream_turn(
    user_message: str,
    history: list[BaseMessage] | None = None,
    *,
    session_id: str | None = None,
) -> Iterator[StreamEvent]:
    """Yield live events for one user turn.

    The ReAct loop only chooses and runs tools. Answer tokens stream exclusively
    from the exposition node, so the final answer has a single grounded source.
    The final event is always ``kind="done"`` with a populated ``TurnResult``.

    Parameters
    ----------
    user_message : str
        The user's question.
    history : list[BaseMessage] or None, optional
        Prior conversation turns.
    session_id : str or None, optional
        Langfuse session identifier when tracing is enabled.

    Yields
    ------
    StreamEvent
        Progress events ending with ``done`` (and optionally ``blocked``).
    """
    if langfuse_enabled():

        def _stream(
            graph_config: dict,
            _handler: object,
            trace_id: str | None,
        ) -> Iterator[StreamEvent]:
            init = (
                build_initial_state(user_message, history, langfuse_trace_id=trace_id)
                if trace_id
                else None
            )
            yield from _stream_turn_core(
                user_message,
                history,
                graph_config=graph_config,
                init=init,
            )

        yield from stream_compliance_turn(
            _stream,
            session_id=session_id,
            user_message=user_message,
        )
        return

    for event in _stream_turn_core(user_message, history):
        if event.kind == "done" and event.result:
            log_turn_summary(event.result)
        yield event


def _stream_turn_core(
    user_message: str,
    history: list[BaseMessage] | None,
    *,
    graph_config: dict | None = None,
    init: AgentState | None = None,
) -> Iterator[StreamEvent]:
    """Core streaming loop without Langfuse wrapping."""
    intent = classify_intent(user_message)
    yield StreamEvent(kind="intent", intent=intent)
    yield StreamEvent(kind="phase", phase="Checking request guardrails")

    init = init or build_initial_state(user_message, history)

    last_values: dict | None = None
    answer_started = False
    routed = False
    seen_tools: set[str] = set()

    for mode, chunk in get_graph().stream(
        init,
        stream_mode=["updates", "messages", "values"],
        config=graph_config or {},
    ):
        if mode == "values":
            last_values = chunk
            continue

        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") != "exposition":
                continue
            text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            if not text:
                continue
            if not answer_started:
                answer_started = True
                yield StreamEvent(kind="answer_start")
            yield StreamEvent(kind="token", token=text)
            continue

        for node, update in chunk.items():
            if node == "input_guard":
                verdict = update.get("pre_input")
                if verdict and not verdict.passed:
                    continue
                if not routed:
                    routed = True
                    yield StreamEvent(kind="phase", phase=f"Routing · {intent.label}")
                    yield StreamEvent(kind="phase", phase="Consulting EU regulatory data")
            elif node == "agent":
                for m in update.get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for call in m.tool_calls:
                            name = call.get("name") or "tool"
                            if name in seen_tools:
                                continue
                            seen_tools.add(name)
                            label = _tool_label(name)
                            yield StreamEvent(
                                kind="tool_start",
                                tool_name=name,
                                tool_label=label,
                                phase=f"Running · {label}",
                            )
            elif node == "tools":
                for m in update.get("messages", []):
                    if isinstance(m, ToolMessage):
                        yield StreamEvent(
                            kind="tool_end",
                            tool_name=m.name or "tool",
                            tool_label=_tool_label(m.name or "tool"),
                        )
            elif node == "output_verify":
                yield StreamEvent(kind="phase", phase="Verifying response guardrails")

    if last_values is not None:
        result = state_to_result(cast(AgentState, last_values))
    else:
        pre = check_pre_input(user_message)
        guardrails = GuardrailReport(pre_input=pre)
        result = TurnResult(
            answer="The agent did not return a response. Please try again.",
            intent=intent,
            guardrails=guardrails,
            trustworthiness=compute_turn_trustworthiness([], guardrails),
        )

    if result.blocked:
        yield StreamEvent(kind="blocked", result=result)
    yield StreamEvent(kind="done", result=result)


@dataclass
class ReasoningStep:
    """One step in the live reasoning progress display.

    Attributes
    ----------
    label : str
        Human-readable step description.
    state : {"pending", "active", "done"}
        Rendering state for the UI.
    """

    label: str
    state: Literal["pending", "active", "done"] = "pending"

    def to_dict(self) -> dict[str, str]:
        """Serialize the step for JSON or session storage.

        Returns
        -------
        dict[str, str]
            Mapping with ``label`` and ``state`` keys.
        """
        return {"label": self.label, "state": self.state}


def apply_event_to_steps(steps: list[ReasoningStep], event: StreamEvent) -> list[ReasoningStep]:
    """Update reasoning steps in place for live UI rendering.

    Parameters
    ----------
    steps : list[ReasoningStep]
        Current reasoning step list (mutated in place).
    event : StreamEvent
        Latest stream event from ``stream_turn``.

    Returns
    -------
    list[ReasoningStep]
        The same ``steps`` list after mutation.
    """
    for step in steps:
        if step.state == "active":
            step.state = "done"

    if event.kind == "phase" and event.phase:
        label = event.phase
        if label.startswith("Routing · "):
            label = "Routing request"
        steps.append(ReasoningStep(label=label, state="active"))
    elif event.kind == "tool_start" and event.tool_label:
        steps.append(ReasoningStep(label=event.tool_label, state="active"))
    elif event.kind in ("answer_start", "token"):
        for step in steps:
            step.state = "done"
    elif event.kind == "done":
        for step in steps:
            if step.state == "active":
                step.state = "done"

    return steps
