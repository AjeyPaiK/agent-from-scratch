"""Helpers for cleaning model-visible assistant text."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage

_LEAK_MARKERS = (
    "please provide the tool call response",
    "please respond with a json object",
    '"type": "function"',
    "note: the output will be used to answer",
)

_LEADING_TOOL_OUTPUT_RE = re.compile(
    r"^\s*(?:based on|according to|from)\s+(?:the\s+)?tool(?:s)?\s+"
    r"(?:output|outputs|results?|data|response),?\s*",
    re.I,
)

_TOOL_SCAFFOLD_RE = re.compile(
    r"("
    r"i will call the|i'll call the|to answer your question,?\s+i will call|"
    r"with the following arguments|\"name\"\s*:\s*\"lookup_|"
    r"\"parameters\"\s*:\s*\{|function with the following"
    r")",
    re.I,
)


def is_tool_scaffolding(text: str) -> bool:
    """Detect text that describes a tool invocation rather than a compliance answer.

    Parameters
    ----------
    text : str
        Candidate assistant message content.

    Returns
    -------
    bool
        ``True`` when the text looks like tool-call scaffolding or JSON templates.
    """
    if not text or not str(text).strip():
        return True
    t = str(text).strip()
    if _TOOL_SCAFFOLD_RE.search(t):
        return True
    if t.startswith("{") and ("lookup_" in t or '"parameters"' in t):
        return True
    return False


def sanitize_model_answer(text: str) -> str:
    """Strip tool-call template leakage from model-visible text.

    Parameters
    ----------
    text : str
        Raw assistant message content.

    Returns
    -------
    str
        Cleaned text with leak markers and leading tool-output phrases removed.
    """
    if not text or not isinstance(text, str):
        return text if isinstance(text, str) else str(text)

    cut_at: int | None = None
    lower = text.lower()
    for marker in _LEAK_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and (cut_at is None or idx < cut_at):
            cut_at = idx

    if cut_at is not None:
        text = text[:cut_at]

    text = _LEADING_TOOL_OUTPUT_RE.sub("", text)
    text = _LEADING_TOOL_OUTPUT_RE.sub("", text)

    trimmed = text.rstrip()
    if trimmed.endswith("}"):
        lines = trimmed.splitlines()
        if lines and lines[-1].strip() == "}":
            text = "\n".join(lines[:-1]).rstrip()

    return text.rstrip()


def extract_final_answer(messages: list[BaseMessage]) -> str | None:
    """Return the last AI message that is a real answer, not tool-call scaffolding.

    Parameters
    ----------
    messages : list[BaseMessage]
        Conversation message history.

    Returns
    -------
    str or None
        Sanitized answer text, or ``None`` when no suitable message is found.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if msg.tool_calls:
            continue
        if is_tool_scaffolding(content):
            continue
        cleaned = sanitize_model_answer(content)
        if cleaned.strip():
            return cleaned
    return None
