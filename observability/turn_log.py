"""Structured local logging when Langfuse is off or for CLI debugging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config.settings import AGENT_LOG_TURNS
from observability.langfuse_integration import langfuse_enabled

if TYPE_CHECKING:
    from agent.pipeline import TurnResult

logger = logging.getLogger("eu_cosmetics.agent")


def _ensure_logger_configured() -> None:
    """Attach a stderr handler when no logging is configured yet."""
    if logger.handlers or logging.root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def log_turn_summary(
    result: TurnResult,
    *,
    trace_id: str | None = None,
) -> None:
    """Emit a one-line structured summary for a completed turn.

    Parameters
    ----------
    result : TurnResult
        Completed turn artifacts.
    trace_id : str or None, optional
        Langfuse trace id when tracing is active.
    """
    _ensure_logger_configured()
    tool_names = [entry.name for entry in result.tool_trace]
    scored = [e for e in result.tool_trace if e.accuracy is not None]
    avg_accuracy: float | None = None
    if scored:
        accuracies = [e.accuracy for e in scored if e.accuracy is not None]
        avg_accuracy = sum(accuracies) / len(accuracies)

    payload: dict[str, Any] = {
        "intent": result.intent.primary_intent,
        "intent_confidence": result.intent.confidence,
        "blocked": result.blocked,
        "guardrails_passed": result.guardrails.all_passed,
        "tools": tool_names,
        "tool_count": len(tool_names),
        "answer_chars": len(result.answer),
    }
    if avg_accuracy is not None:
        payload["tool_accuracy_avg"] = round(avg_accuracy, 4)
    if result.trustworthiness is not None:
        payload["turn_trustworthiness"] = round(result.trustworthiness.value, 4)
    if trace_id:
        payload["langfuse_trace_id"] = trace_id

    message = (
        f"turn intent={payload['intent']} blocked={payload['blocked']} "
        f"guardrails={'pass' if payload['guardrails_passed'] else 'fail'} "
        f"tools={payload['tool_count']}"
    )
    if result.trustworthiness is not None:
        message += f" trustworthiness={payload['turn_trustworthiness']}"
    if langfuse_enabled() and not AGENT_LOG_TURNS:
        logger.debug(message, extra=payload)
    else:
        logger.info(message, extra=payload)
