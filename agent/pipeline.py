"""Turn orchestration — invoke the LangGraph agent and assemble a TurnResult."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.graph import get_graph
from agent.intent import IntentResult
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from agent.tool_trace import ToolTraceEntry
from guardrails.types import GuardrailReport
from observability.langfuse_integration import langfuse_enabled
from observability.turn_log import log_turn_summary
from observability.turn_trace import run_compliance_turn


@dataclass
class TurnResult:
    """Artifacts produced by one completed user turn.

    Attributes
    ----------
    answer : str
        Final user-facing response text.
    intent : IntentResult
        Rule-based intent classification for the turn.
    guardrails : GuardrailReport
        Verdicts from all three symbolic guardrail stages.
    tool_trace : list[ToolTraceEntry]
        Structured record of tool calls executed during the turn.
    blocked : bool
        ``True`` when pre-input guardrails prevented the LLM from running.
    raw_messages : list[BaseMessage]
        Full LangChain message history after the turn.
    """

    answer: str
    intent: IntentResult
    guardrails: GuardrailReport
    tool_trace: list[ToolTraceEntry] = field(default_factory=list)
    blocked: bool = False
    raw_messages: list[BaseMessage] = field(default_factory=list)


def build_initial_state(
    user_message: str,
    history: list[BaseMessage] | None = None,
    *,
    langfuse_trace_id: str | None = None,
) -> AgentState:
    """Compose the graph input from the system prompt, prior turns, and the new question.

    Parameters
    ----------
    user_message : str
        The user's latest question.
    history : list[BaseMessage] or None, optional
        Prior conversation turns excluding the new question.
    langfuse_trace_id : str or None, optional
        Trace identifier to attach when Langfuse observability is enabled.

    Returns
    -------
    AgentState
        Initial state dict ready for ``graph.invoke``.
    """
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history or [])
    messages.append(HumanMessage(content=user_message))
    state: AgentState = {"messages": messages, "user_message": user_message}
    if langfuse_trace_id:
        state["langfuse_trace_id"] = langfuse_trace_id
    return state


def state_to_result(state: AgentState) -> TurnResult:
    """Assemble a ``TurnResult`` from the graph's final state.

    Parameters
    ----------
    state : AgentState
        Final state returned by ``graph.invoke`` or ``graph.stream``.

    Returns
    -------
    TurnResult
        Structured turn result for the UI or CLI.
    """
    report = GuardrailReport(
        pre_input=state["pre_input"],
        pre_tool=list(state.get("pre_tool") or []),
        post_output=state.get("post_output"),
    )
    return TurnResult(
        answer=state.get("answer", ""),
        intent=state["intent"],
        guardrails=report,
        tool_trace=list(state.get("tool_trace") or []),
        blocked=bool(state.get("blocked", False)),
        raw_messages=list(state.get("messages") or []),
    )


def run_turn(
    user_message: str,
    history: list[BaseMessage] | None = None,
    *,
    session_id: str | None = None,
) -> TurnResult:
    """Run one user turn through the compiled compliance graph.

    Parameters
    ----------
    user_message : str
        The user's question.
    history : list[BaseMessage] or None, optional
        Prior conversation turns.
    session_id : str or None, optional
        Langfuse session identifier when tracing is enabled.

    Returns
    -------
    TurnResult
        Structured result including answer, guardrails, and tool trace.
    """
    def _invoke(
        graph_config: dict,
        _handler: object,
        trace_id: str | None,
    ) -> TurnResult:
        final_state = get_graph().invoke(
            build_initial_state(user_message, history, langfuse_trace_id=trace_id),
            config=graph_config,
        )
        return state_to_result(final_state)

    if langfuse_enabled():
        return run_compliance_turn(
            _invoke,
            session_id=session_id,
            user_message=user_message,
        )

    result = _invoke({}, None, None)
    log_turn_summary(result)
    return result
