"""Composite turn trustworthiness score (assignment §2.1).

Combines per-tool oracle accuracy with symbolic guardrail integrity into one
turn-scoped score in ``[0.0, 1.0]``. See ``docs/SCORING.md`` for the full spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.tool_trace import ToolTraceEntry
from guardrails.types import GuardrailReport


@dataclass
class TurnTrustworthiness:
    """Composite trustworthiness for one agent turn.

    Attributes
    ----------
    value : float
        Weighted composite in ``[0.0, 1.0]``.
    tool_accuracy : float
        Tool-output component before weighting.
    guardrail_integrity : float
        Guardrail component before weighting.
    tool_weight : float
        Weight applied to ``tool_accuracy`` (default ``0.6``).
    guardrail_weight : float
        Weight applied to ``guardrail_integrity`` (default ``0.4``).
    """

    value: float
    tool_accuracy: float
    guardrail_integrity: float
    tool_weight: float = 0.6
    guardrail_weight: float = 0.4


def _tool_accuracy_component(tool_trace: list[ToolTraceEntry]) -> float:
    """Mean oracle accuracy for executed tools, or defaults when none ran."""
    executed = [entry for entry in tool_trace if not entry.blocked]
    scored = [entry for entry in executed if entry.accuracy is not None]
    if scored:
        accuracies = [entry.accuracy for entry in scored if entry.accuracy is not None]
        return sum(accuracies) / len(accuracies)
    if not executed:
        return 1.0
    return 0.0


def _guardrail_integrity(guardrails: GuardrailReport, *, blocked: bool) -> float:
    """Score how well symbolic guardrails protected the turn."""
    if blocked and not guardrails.pre_input.passed:
        return 1.0

    parts: list[float] = [1.0 if guardrails.pre_input.passed else 0.0]
    if guardrails.pre_tool:
        passed = sum(1.0 for verdict in guardrails.pre_tool if verdict.passed)
        parts.append(passed / len(guardrails.pre_tool))
    if guardrails.post_output is not None:
        parts.append(1.0 if guardrails.post_output.passed else 0.0)
    return sum(parts) / len(parts)


def compute_turn_trustworthiness(
    tool_trace: list[ToolTraceEntry],
    guardrails: GuardrailReport,
    *,
    blocked: bool = False,
    tool_weight: float = 0.6,
    guardrail_weight: float = 0.4,
) -> TurnTrustworthiness:
    """Compute the composite trustworthiness score for one turn.

    Parameters
    ----------
    tool_trace : list[ToolTraceEntry]
        Tool calls from the current turn only (already oracle-scored).
    guardrails : GuardrailReport
        Guardrail verdicts for the current turn only.
    blocked : bool, optional
        Whether pre-input guardrails blocked the turn.
    tool_weight : float, optional
        Weight for the tool-accuracy component.
    guardrail_weight : float, optional
        Weight for the guardrail-integrity component.

    Returns
    -------
    TurnTrustworthiness
        Composite score and component breakdown.
    """
    tool_accuracy = _tool_accuracy_component(tool_trace)
    guardrail_integrity = _guardrail_integrity(guardrails, blocked=blocked)
    value = min(
        1.0,
        max(0.0, tool_weight * tool_accuracy + guardrail_weight * guardrail_integrity),
    )
    return TurnTrustworthiness(
        value=value,
        tool_accuracy=tool_accuracy,
        guardrail_integrity=guardrail_integrity,
        tool_weight=tool_weight,
        guardrail_weight=guardrail_weight,
    )
