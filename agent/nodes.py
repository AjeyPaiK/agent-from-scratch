"""Graph nodes for the EU cosmetics compliance agent.

Each symbolic guardrail is its own node, so the ReAct loop and its three
safety stages are visible in the compiled graph::

    intent → input_guard → agent ⇄ (tool_guard → tools) → exposition → output_verify
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from agent.exposition import stream_exposition
from agent.intent import classify_intent
from agent.llm import build_chat_llm
from agent.response import extract_final_answer
from agent.state import AgentState
from agent.tool_args import sanitize_tool_kwargs
from agent.tool_trace import build_tool_trace
from guardrails import check_post_output, check_pre_input, check_pre_tool
from scoring.trace import apply_accuracy_scores
from tools import ALL_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

_NO_ANSWER = (
    "I could not produce a compliance answer for that question. "
    "Please try again or rephrase (e.g. include product type and EU scope)."
)


def intent_node(state: AgentState) -> dict[str, Any]:
    """Classify user intent without invoking an LLM.

    Parameters
    ----------
    state : AgentState
        Current graph state containing ``user_message``.

    Returns
    -------
    dict[str, Any]
        State update with ``intent`` set to an ``IntentResult``.
    """
    return {"intent": classify_intent(state["user_message"])}


def input_guard_node(state: AgentState) -> dict[str, Any]:
    """Run stage 1 — symbolic pre-input guardrail before the LLM sees the input.

    Parameters
    ----------
    state : AgentState
        Current graph state containing ``user_message``.

    Returns
    -------
    dict[str, Any]
        State update with ``pre_input`` verdict; sets ``answer`` and ``blocked``
        when the guard fails.
    """
    verdict = check_pre_input(state["user_message"])
    update: dict[str, Any] = {"pre_input": verdict}
    if not verdict.passed:
        update["answer"] = verdict.message
        update["blocked"] = True
    return update


def agent_node(state: AgentState) -> dict[str, Any]:
    """Run the ReAct reasoning step where the LLM decides whether to call tools.

    Parameters
    ----------
    state : AgentState
        Current graph state containing ``messages``.

    Returns
    -------
    dict[str, Any]
        State update appending the model's ``AIMessage`` response.
    """
    llm = build_chat_llm().bind_tools(ALL_TOOLS)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_guard_node(state: AgentState) -> dict[str, Any]:
    """Run stage 2 — symbolic validation of tool-call arguments before execution.

    Valid calls are queued for the tools node; invalid calls are answered
    immediately with a structured ``guardrail_blocked`` ``ToolMessage`` so every
    ``tool_call_id`` is resolved and the message history stays valid.

    Parameters
    ----------
    state : AgentState
        Current graph state; the last message must be an ``AIMessage`` with
        ``tool_calls``.

    Returns
    -------
    dict[str, Any]
        State update with ``pre_tool`` verdicts, ``validated_calls``, and any
        blocked ``ToolMessage`` entries.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    verdicts = []
    validated: list[dict[str, Any]] = []
    blocked_messages: list[ToolMessage] = []

    for call in tool_calls:
        name = call.get("name") or "tool"
        call_id = call.get("id") or ""
        args = sanitize_tool_kwargs(name, call.get("args") or {})
        verdict = check_pre_tool(name, args)
        verdicts.append(verdict)
        if verdict.passed:
            validated.append({"id": call_id, "name": name, "args": args})
        else:
            payload = {
                "guardrail_blocked": True,
                "stage": "pre_tool",
                "rule_id": verdict.rule_id,
                "message": verdict.message,
            }
            blocked_messages.append(
                ToolMessage(content=json.dumps(payload), name=name, tool_call_id=call_id)
            )

    return {
        "pre_tool": verdicts,
        "validated_calls": validated,
        "messages": blocked_messages,
    }


def tools_node(state: AgentState) -> dict[str, Any]:
    """Execute tool calls that passed the symbolic pre-tool guard.

    Parameters
    ----------
    state : AgentState
        Current graph state containing ``validated_calls``.

    Returns
    -------
    dict[str, Any]
        State update with ``ToolMessage`` results and an empty ``validated_calls``
        list.
    """
    messages: list[ToolMessage] = []
    for call in state.get("validated_calls", []):
        tool = _TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            result: Any = {"error": f"Unknown tool '{call['name']}'."}
        else:
            try:
                result = tool.invoke(call["args"])
            except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
                result = {"error": str(exc)}
        content = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )
        messages.append(ToolMessage(content=content, name=call["name"], tool_call_id=call["id"]))
    return {"messages": messages, "validated_calls": []}


def exposition_node(state: AgentState) -> dict[str, Any]:
    """Synthesize one grounded answer from collected tool outputs.

    Streams internally so answer tokens surface via the graph's message stream;
    falls back to the model's own reply when no tools ran.

    Parameters
    ----------
    state : AgentState
        Current graph state containing ``messages`` and ``user_message``.

    Returns
    -------
    dict[str, Any]
        State update with ``answer``, ``tool_trace``, ``tool_names``, and
        ``tool_outputs``.
    """
    trace, names, outputs = build_tool_trace(state["messages"])
    trace = apply_accuracy_scores(trace)
    answer = "".join(stream_exposition(state["user_message"], trace, outputs)).strip()
    if not answer:
        answer = (extract_final_answer(state["messages"]) or "").strip()
    if not answer:
        answer = _NO_ANSWER
    return {
        "answer": answer,
        "tool_trace": trace,
        "tool_names": names,
        "tool_outputs": outputs,
    }


def output_verify_node(state: AgentState) -> dict[str, Any]:
    """Run stage 3 — symbolic cross-check of the answer against tool outputs.

    Parameters
    ----------
    state : AgentState
        Current graph state containing ``answer``, ``tool_names``, and
        ``tool_outputs``.

    Returns
    -------
    dict[str, Any]
        State update with ``post_output`` verdict; may append a guardrail note
        to ``answer`` when verification fails.
    """
    answer = state.get("answer", "")
    verdict = check_post_output(
        answer,
        tool_names_used=state.get("tool_names", []),
        tool_outputs=state.get("tool_outputs", []),
    )
    update: dict[str, Any] = {"post_output": verdict}
    if not verdict.passed:
        if answer.strip():
            answer = f"{answer}\n\n---\n⚠️ **Guardrail note ({verdict.rule_id}):** {verdict.message}"
        else:
            answer = verdict.message
        update["answer"] = answer
    return update


def route_after_input(state: AgentState) -> str:
    """Route away from the agent when the input guard blocked the request.

    Parameters
    ----------
    state : AgentState
        Current graph state containing ``pre_input``.

    Returns
    -------
    str
        ``"agent"`` when input passed, ``"end"`` when blocked.
    """
    verdict = state.get("pre_input")
    return "agent" if verdict and verdict.passed else "end"


def route_after_agent(state: AgentState) -> str:
    """Route to tools or exposition based on whether the model requested tool calls.

    Parameters
    ----------
    state : AgentState
        Current graph state; the last message must be an ``AIMessage``.

    Returns
    -------
    str
        ``"tool_guard"`` when tool calls are present, otherwise ``"exposition"``.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_guard"
    return "exposition"
