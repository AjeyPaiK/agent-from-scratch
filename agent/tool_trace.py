"""Structured tool-call trace for UI and CLI display."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

TOOL_LABELS: dict[str, str] = {
    "lookup_ingredient_regulation": "Ingredient lookup",
    "check_concentration_compliance": "Concentration check",
    "get_labelling_marketing_rules": "Labelling rules",
}


@dataclass
class ToolTraceEntry:
    """One executed or blocked tool call in a turn.

    Attributes
    ----------
    step : int
        1-based execution order within the turn.
    name : str
        Internal tool name.
    label : str
        Human-readable tool label.
    args : dict[str, Any]
        Sanitized arguments passed to the tool.
    output : Any
        Parsed tool output or guardrail block payload.
    blocked : bool
        ``True`` when pre-tool guardrails blocked execution.
    rule_id : str or None
        Guardrail rule identifier when ``blocked`` is ``True``.
    accuracy : float or None
        Oracle accuracy score when scoring is available.
    accuracy_mismatches : list[str]
        Field-level mismatches reported by the oracle.
    oracle_version : str or None
        Snapshot identifier used for scoring.
    """

    step: int
    name: str
    label: str
    args: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    blocked: bool = False
    rule_id: str | None = None
    accuracy: float | None = None
    accuracy_mismatches: list[str] = field(default_factory=list)
    oracle_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Dataclass fields as a JSON-serializable mapping.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolTraceEntry:
        """Deserialize an entry from a plain dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Mapping produced by ``to_dict``.

        Returns
        -------
        ToolTraceEntry
            Reconstructed trace entry.
        """
        return cls(**data)


def _parse_output(raw: str) -> tuple[Any, bool, str | None]:
    """Parse a raw tool message string into output, blocked flag, and rule id."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, False, None

    if isinstance(parsed, dict) and parsed.get("guardrail_blocked"):
        return parsed, True, parsed.get("rule_id")
    return parsed, False, None


def build_tool_trace(
    messages: list[BaseMessage],
) -> tuple[list[ToolTraceEntry], list[str], list[str]]:
    """Extract a structured tool trace plus flat lists for guardrail checks.

    Parameters
    ----------
    messages : list[BaseMessage]
        Full conversation history including AI tool calls and tool responses.

    Returns
    -------
    entries : list[ToolTraceEntry]
        Structured trace entries in execution order.
    tool_names : list[str]
        Flat list of tool names in execution order.
    tool_outputs : list[str]
        Raw JSON strings of tool outputs for post-output verification.
    """
    pending_args: dict[str, dict[str, Any]] = {}
    entries: list[ToolTraceEntry] = []
    tool_names: list[str] = []
    tool_outputs: list[str] = []
    step = 0

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                call_id = call.get("id")
                if not call_id:
                    continue
                pending_args[call_id] = {
                    "name": call["name"],
                    "args": call.get("args") or {},
                }

        if not isinstance(msg, ToolMessage):
            continue

        step += 1
        pending = pending_args.get(msg.tool_call_id or "", {})
        name = msg.name or pending.get("name") or "tool"
        args = pending.get("args") or {}
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)

        tool_names.append(name)
        tool_outputs.append(raw)

        output, blocked, rule_id = _parse_output(raw)
        entries.append(
            ToolTraceEntry(
                step=step,
                name=name,
                label=TOOL_LABELS.get(name, name.replace("_", " ").title()),
                args=args,
                output=output,
                blocked=blocked,
                rule_id=rule_id,
            )
        )

    return entries, tool_names, tool_outputs
