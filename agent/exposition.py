"""Exposition stage — compose one grounded answer from collected tool outputs.

After the ReAct loop has run its tools, this stage makes a single LLM synthesis
call whose only job is to turn the collected tool outputs into a succinct,
citation-faithful answer. It never decides tool calls and is grounded strictly
in what the tools returned, so the post-output guardrails can verify it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.llm import build_chat_llm
from agent.tool_trace import ToolTraceEntry
from tools.messages import annex_absence_from_output

EXPOSITION_SYSTEM_PROMPT = """You are an EU cosmetics regulatory writer.

You are given the user's question and the raw outputs of compliance tools that
already ran. Compose a single, succinct answer that directly resolves the
question.

You MUST:
1. Use ONLY facts present in the tool outputs below. Never introduce annex
   numbers, legal statuses, or concentration limits that are not present.
2. Do not just cite the annex reference — explain in plain language WHAT the
   rule actually says. When the tool outputs include conditions (maximum
   concentrations, the product types they apply to, warnings, or usage
   restrictions), state them concretely so the user understands whether and how
   they can use the ingredient. Cite the annex reference alongside the detail
   (e.g. "restricted to 3% in rinse-off hair products").
3. After each claim, cite the annex reference (e.g. "Annex III, entry 98").
4. Begin directly with the conclusion (e.g. "Retinol is restricted…"). Never
   open with "Based on the tool output" or similar meta phrases.

If the tool outputs do not contain enough information to answer, say so plainly
rather than guessing.

Respond in plain prose only. Don't output JSON or function-call templates.
"""


def direct_annex_absence_answer(tool_trace: list[ToolTraceEntry]) -> str | None:
    """Return a canonical message when every tool reports annex absence.

    Skips LLM synthesis when all tool outputs agree on the same not-found
    annex-absence wording.

    Parameters
    ----------
    tool_trace : list[ToolTraceEntry]
        Executed tool calls from the current turn.

    Returns
    -------
    str or None
        Canonical annex-absence message, or ``None`` when synthesis is required.
    """
    if not tool_trace or any(entry.blocked for entry in tool_trace):
        return None
    messages: list[str] = []
    for entry in tool_trace:
        message = annex_absence_from_output(entry.output)
        if message is None:
            return None
        messages.append(message)
    unique = set(messages)
    return messages[0] if len(unique) == 1 else None


def render_tool_context(tool_trace: list[ToolTraceEntry]) -> str:
    """Flatten the tool trace into a labelled, model-readable evidence block.

    Parameters
    ----------
    tool_trace : list[ToolTraceEntry]
        Executed tool calls to include in the synthesis prompt.

    Returns
    -------
    str
        Formatted multi-block string for the exposition LLM.
    """
    blocks: list[str] = []
    for entry in tool_trace:
        try:
            output = json.dumps(entry.output, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            output = str(entry.output)
        args = ", ".join(f"{k}={v!r}" for k, v in (entry.args or {}).items())
        blocks.append(
            f"[Tool {entry.step}: {entry.label} ({entry.name})]\n"
            f"Arguments: {args or '(none)'}\n"
            f"Output:\n{output}"
        )
    return "\n\n".join(blocks)


def _build_messages(user_message: str, tool_trace: list[ToolTraceEntry]) -> list[BaseMessage]:
    """Build LangChain messages for the exposition LLM call."""
    context = render_tool_context(tool_trace)
    return [
        SystemMessage(content=EXPOSITION_SYSTEM_PROMPT),
        HumanMessage(content=f"User question:\n{user_message}\n\nTool outputs:\n{context}"),
    ]


def stream_exposition(
    user_message: str,
    tool_trace: list[ToolTraceEntry],
    tool_outputs: list[str],
) -> Iterator[str]:
    """Stream the synthesis answer token-by-token.

    Parameters
    ----------
    user_message : str
        The user's original question.
    tool_trace : list[ToolTraceEntry]
        Structured tool-call trace for context rendering.
    tool_outputs : list[str]
        Raw tool output strings; when empty, nothing is yielded.

    Yields
    ------
    str
        Answer token fragments. Yields nothing when no tools ran.
    """
    if not tool_outputs:
        return

    direct = direct_annex_absence_answer(tool_trace)
    if direct:
        yield direct
        return

    for chunk in build_chat_llm().stream(_build_messages(user_message, tool_trace)):
        content = chunk.content
        text = content if isinstance(content, str) else str(content or "")
        if text:
            yield text
