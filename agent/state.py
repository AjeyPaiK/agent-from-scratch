"""Shared state for the LangGraph compliance agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent.intent import IntentResult
from agent.tool_trace import ToolTraceEntry
from guardrails.types import GuardrailVerdict


class AgentState(TypedDict, total=False):
    """State threaded through every node of the turn graph.

    ``messages`` accumulates via the standard ``add_messages`` reducer.
    ``pre_tool`` accumulates across ReAct loops so the final report holds every
    tool-argument verdict from the turn.

    Attributes
    ----------
    messages : list[BaseMessage]
        Conversation history including system, human, AI, and tool messages.
    user_message : str
        The latest user question for this turn.
    intent : IntentResult
        Rule-based intent classification result.
    pre_input : GuardrailVerdict
        Stage 1 input guardrail verdict.
    pre_tool : list[GuardrailVerdict]
        Stage 2 tool-argument verdicts accumulated across ReAct loops.
    post_output : GuardrailVerdict or None
        Stage 3 output verification verdict.
    validated_calls : list[dict[str, Any]]
        Tool calls that passed pre-tool validation, awaiting execution.
    tool_trace : list[ToolTraceEntry]
        Structured tool-call trace for display and scoring.
    tool_names : list[str]
        Flat list of tool names used during the turn.
    tool_outputs : list[str]
        Raw JSON tool outputs for guardrail cross-checks.
    langfuse_trace_id : str or None
        Langfuse trace identifier when observability is enabled.
    answer : str
        Final synthesized answer text.
    blocked : bool
        ``True`` when pre-input guardrails blocked the turn.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_message: str
    intent: IntentResult

    pre_input: GuardrailVerdict
    pre_tool: Annotated[list[GuardrailVerdict], operator.add]
    post_output: GuardrailVerdict | None

    validated_calls: list[dict[str, Any]]

    tool_trace: list[ToolTraceEntry]
    tool_names: list[str]
    tool_outputs: list[str]
    langfuse_trace_id: str | None
    answer: str
    blocked: bool
