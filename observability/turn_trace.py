"""Shared Langfuse turn wrapper for invoke and stream paths."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from observability.langfuse_integration import (
    finalize_turn_observability,
    langfuse_enabled,
    langfuse_turn_context,
)

T = TypeVar("T")

COMPLIANCE_TURN_NAME = "compliance-turn"
DEFAULT_TURN_TAGS = ["eu-cosmetics-agent"]

TurnRunner = Callable[[dict[str, Any], Any, str | None], T]


def run_compliance_turn(
    run_fn: TurnRunner[T],
    *,
    session_id: str | None = None,
    user_message: str | None = None,
    tags: list[str] | None = None,
) -> T:
    """Run a synchronous turn inside a Langfuse ``@observe`` span when enabled.

    Parameters
    ----------
    run_fn : callable
        Receives ``(graph_config, handler, trace_id)`` and returns the turn outcome.
    session_id : str or None, optional
        Langfuse session id.
    user_message : str or None, optional
        User question for trace input metadata.
    tags : list[str] or None, optional
        Langfuse tags; defaults to ``DEFAULT_TURN_TAGS``.

    Returns
    -------
    T
        Whatever ``run_fn`` returns.
    """
    if not langfuse_enabled():
        return run_fn({}, None, None)

    from langfuse import observe

    turn_tags = tags if tags is not None else DEFAULT_TURN_TAGS

    @observe(name=COMPLIANCE_TURN_NAME)
    def _wrapped() -> T:
        with langfuse_turn_context(session_id=session_id, tags=turn_tags) as (
            config,
            handler,
            trace_id,
        ):
            outcome = run_fn(config, handler, trace_id)
            _maybe_finalize_turn(outcome, handler=handler, trace_id=trace_id, user_message=user_message)
            return outcome

    return _wrapped()


def stream_compliance_turn(
    stream_fn: TurnRunner[Iterator[T]],
    *,
    session_id: str | None = None,
    user_message: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[T]:
    """Stream events inside a Langfuse ``@observe`` span when enabled.

    Parameters
    ----------
    stream_fn : callable
        Receives ``(graph_config, handler, trace_id)`` and yields stream events.
    session_id : str or None, optional
        Langfuse session id.
    user_message : str or None, optional
        User question for trace input metadata.
    tags : list[str] or None, optional
        Langfuse tags; defaults to ``DEFAULT_TURN_TAGS``.

    Yields
    ------
    T
        Events from ``stream_fn``.
    """
    if not langfuse_enabled():
        yield from stream_fn({}, None, None)
        return

    from langfuse import observe

    turn_tags = tags if tags is not None else DEFAULT_TURN_TAGS

    @observe(name=COMPLIANCE_TURN_NAME)
    def _wrapped() -> Iterator[T]:
        with langfuse_turn_context(session_id=session_id, tags=turn_tags) as (
            config,
            handler,
            trace_id,
        ):
            turn_result: object | None = None
            for event in stream_fn(config, handler, trace_id):
                result = getattr(event, "result", None)
                if getattr(event, "kind", None) == "done" and result is not None:
                    turn_result = result
                yield event
            if turn_result is not None:
                finalize_turn_observability(
                    turn_result,
                    handler=handler,
                    trace_id=trace_id,
                    user_message=user_message,
                )

    yield from _wrapped()


def _maybe_finalize_turn(
    outcome: object,
    *,
    handler: CallbackHandler | None,
    trace_id: str | None,
    user_message: str | None,
) -> None:
    """Finalize observability when ``outcome`` looks like a ``TurnResult``."""
    if not (
        hasattr(outcome, "tool_trace")
        and hasattr(outcome, "guardrails")
        and hasattr(outcome, "answer")
        and hasattr(outcome, "intent")
    ):
        return
    finalize_turn_observability(
        outcome,  # type: ignore[arg-type]
        handler=handler,
        trace_id=trace_id,
        user_message=user_message,
    )
