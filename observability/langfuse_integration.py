"""Langfuse tracing and tool-accuracy score export.

Integrates LangGraph agent turns with Langfuse traces, scores, and session
metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from agent.tool_trace import ToolTraceEntry
from config.settings import LANGFUSE_ENABLED

if TYPE_CHECKING:
    from langfuse.langchain import CallbackHandler

    from guardrails.types import GuardrailReport


def langfuse_enabled() -> bool:
    """Return whether Langfuse export is enabled in settings.

    Returns
    -------
    bool
        ``True`` when ``LANGFUSE_ENABLED`` is set in application settings.
    """
    return LANGFUSE_ENABLED


def capture_turn_trace_id() -> str | None:
    """Capture the active Langfuse trace id at the start of a traced turn.

    Returns
    -------
    str or None
        Current trace id as a string, or ``None`` when Langfuse is disabled or
        no trace is active.

    Notes
    -----
    No-op when Langfuse is disabled; does not create a new trace.
    """
    if not langfuse_enabled():
        return None

    from langfuse import get_client

    trace_id = get_client().get_current_trace_id()
    return str(trace_id) if trace_id else None


def resolve_export_trace_id(
    handler: CallbackHandler | None = None,
    fallback_trace_id: str | None = None,
) -> str | None:
    """Pick the Langfuse trace id to attach scores to after a turn completes.

    Parameters
    ----------
    handler : CallbackHandler or None, optional
        LangChain callback handler from the turn, by default ``None``.
    fallback_trace_id : str or None, optional
        Trace id captured at turn start when handler id is unavailable,
        by default ``None``.

    Returns
    -------
    str or None
        Resolved trace id, or ``None`` when none can be determined.

    Notes
    -----
    Resolution order: ``handler.last_trace_id``, then ``fallback_trace_id``,
    then the current client trace id.
    """
    if handler is not None:
        handler_trace_id = getattr(handler, "last_trace_id", None)
        if handler_trace_id:
            return str(handler_trace_id)

    if fallback_trace_id:
        return fallback_trace_id

    from langfuse import get_client

    current = get_client().get_current_trace_id()
    return str(current) if current else None


def build_graph_config(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> tuple[dict[str, Any], CallbackHandler | None]:
    """Return LangGraph invoke config plus a CallbackHandler when Langfuse is enabled.

    Parameters
    ----------
    session_id : str or None, optional
        Langfuse session id for grouping turns, by default ``None``.
    user_id : str or None, optional
        Langfuse user id, by default ``None``.
    tags : list[str] or None, optional
        Langfuse tags for the trace, by default ``None``.

    Returns
    -------
    config : dict[str, Any]
        LangGraph invoke config with callbacks and optional metadata.
    handler : CallbackHandler or None
        Langfuse callback handler when enabled; otherwise ``None``.

    Notes
    -----
    When Langfuse is disabled, returns ``({}, None)``.
    """
    if not langfuse_enabled():
        return {}, None

    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler()
    metadata: dict[str, Any] = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if tags:
        metadata["langfuse_tags"] = tags

    config: dict[str, Any] = {"callbacks": [handler]}
    if metadata:
        config["metadata"] = metadata
    return config, handler


SCORE_NAME_BY_TOOL: dict[str, str] = {
    "lookup_ingredient_regulation": "tool_accuracy_lookup",
    "check_concentration_compliance": "tool_accuracy_concentration",
}


def _score_name_for_tool(tool_name: str) -> str:
    """Map a tool name to its Langfuse score name.

    Parameters
    ----------
    tool_name : str
        Registered tool name.

    Returns
    -------
    str
        Score name from ``SCORE_NAME_BY_TOOL``, or ``tool_accuracy_{tool_name}``.
    """
    return SCORE_NAME_BY_TOOL.get(tool_name, f"tool_accuracy_{tool_name}")


def export_tool_accuracy_scores(
    tool_trace: list[ToolTraceEntry],
    trace_id: str | None = None,
) -> None:
    """Push oracle accuracy scores to Langfuse for one completed turn.

    Parameters
    ----------
    tool_trace : list[ToolTraceEntry]
        Tool trace entries for the turn, optionally with ``accuracy`` populated.
    trace_id : str or None, optional
        Langfuse trace id to attach scores to, by default ``None``.

    Returns
    -------
    None

    Notes
    -----
    No-op when Langfuse is disabled or ``trace_id`` is missing. Creates per-tool
    scores, a turn average when scorable entries exist, or a status score when
    tools ran but none were scorable. Calls ``client.flush()`` before returning.
    """
    if not langfuse_enabled() or not trace_id:
        return

    from langfuse import get_client

    client = get_client()
    scored = [entry for entry in tool_trace if entry.accuracy is not None]

    if scored:
        for entry in scored:
            accuracy = entry.accuracy
            assert accuracy is not None
            comment_parts: list[str] = []
            if entry.accuracy_mismatches:
                comment_parts.append("mismatches: " + ", ".join(entry.accuracy_mismatches))
            if entry.oracle_version:
                comment_parts.append(f"oracle: {entry.oracle_version}")
            client.create_score(
                name=_score_name_for_tool(entry.name),
                value=accuracy,
                trace_id=str(trace_id),
                score_id=f"{trace_id}:tool_accuracy:{entry.name}",
                data_type="NUMERIC",
                comment="; ".join(comment_parts) or None,
            )

        accuracies = [entry.accuracy for entry in scored if entry.accuracy is not None]
        client.create_score(
            name="turn_tool_accuracy_avg",
            value=sum(accuracies) / len(accuracies),
            trace_id=str(trace_id),
            score_id=f"{trace_id}:turn_tool_accuracy_avg",
            data_type="NUMERIC",
            comment=f"{len(scored)} scored tool call(s)",
        )
    elif tool_trace:
        tool_names = ", ".join(sorted({entry.name for entry in tool_trace}))
        client.create_score(
            name="tool_accuracy_status",
            value=0.0,
            trace_id=str(trace_id),
            score_id=f"{trace_id}:tool_accuracy_status",
            data_type="NUMERIC",
            comment=f"No scorable tool output; tools used: {tool_names or 'unknown'}",
        )

    client.flush()


def export_guardrail_scores(
    guardrails: GuardrailReport,
    trace_id: str | None = None,
) -> None:
    """Push pass/fail scores for all three symbolic guardrail stages (assignment §2.2).

    Parameters
    ----------
    guardrails : GuardrailReport
        Composite guardrail verdict for one turn.
    trace_id : str or None, optional
        Langfuse trace id to attach scores to, by default ``None``.

    Returns
    -------
    None

    Notes
    -----
    Exports ``guardrail_pre_input``, ``guardrail_pre_tool``,
    ``guardrail_post_output`` (when present), and ``guardrails_all_passed``.
    Pre-tool stage scores ``1.0`` with comment ``"no tool calls"`` when no
    pre-tool checks ran. Calls ``client.flush()`` before returning.
    """
    if not langfuse_enabled() or not trace_id:
        return

    from langfuse import get_client

    client = get_client()
    tid = str(trace_id)

    pre = guardrails.pre_input
    client.create_score(
        name="guardrail_pre_input",
        value=1.0 if pre.passed else 0.0,
        trace_id=tid,
        score_id=f"{tid}:guardrail_pre_input",
        data_type="NUMERIC",
        comment=f"{pre.rule_id}: {pre.message}",
    )

    if guardrails.pre_tool:
        pre_tool_passed = all(v.passed for v in guardrails.pre_tool)
        failed = [v.rule_id for v in guardrails.pre_tool if not v.passed]
        client.create_score(
            name="guardrail_pre_tool",
            value=1.0 if pre_tool_passed else 0.0,
            trace_id=tid,
            score_id=f"{tid}:guardrail_pre_tool",
            data_type="NUMERIC",
            comment="ok" if pre_tool_passed else f"failed: {', '.join(failed)}",
        )
    else:
        client.create_score(
            name="guardrail_pre_tool",
            value=1.0,
            trace_id=tid,
            score_id=f"{tid}:guardrail_pre_tool",
            data_type="NUMERIC",
            comment="no tool calls",
        )

    post = guardrails.post_output
    if post is not None:
        client.create_score(
            name="guardrail_post_output",
            value=1.0 if post.passed else 0.0,
            trace_id=tid,
            score_id=f"{tid}:guardrail_post_output",
            data_type="NUMERIC",
            comment=f"{post.rule_id}: {post.message}",
        )

    stages_passed = guardrails.all_passed
    client.create_score(
        name="guardrails_all_passed",
        value=1.0 if stages_passed else 0.0,
        trace_id=tid,
        score_id=f"{tid}:guardrails_all_passed",
        data_type="NUMERIC",
        comment="composite guardrail pass for turn",
    )

    client.flush()


_METADATA_VALUE_MAX = 200


def _truncate_metadata_value(value: str) -> str:
    if len(value) <= _METADATA_VALUE_MAX:
        return value
    return value[: _METADATA_VALUE_MAX - 1] + "…"


def build_turn_metadata(
    *,
    intent: str,
    intent_confidence: str,
    blocked: bool,
    guardrails_passed: bool,
    tool_names: list[str],
    answer_chars: int,
    avg_tool_accuracy: float | None = None,
) -> dict[str, str]:
    """Build Langfuse-safe string metadata for a completed turn.

    Parameters
    ----------
    intent : str
        Primary intent id.
    intent_confidence : str
        Intent classifier confidence label.
    blocked : bool
        Whether pre-input guardrails blocked the turn.
    guardrails_passed : bool
        Whether all guardrail stages passed.
    tool_names : list[str]
        Tools invoked during the turn.
    answer_chars : int
        Length of the final answer string.
    avg_tool_accuracy : float or None, optional
        Mean oracle accuracy when any tool was scored.

    Returns
    -------
    dict[str, str]
        Metadata keys with values capped at 200 characters (Langfuse limit).
    """
    metadata: dict[str, str] = {
        "intent": intent,
        "intent_confidence": intent_confidence,
        "blocked": str(blocked).lower(),
        "guardrails_passed": str(guardrails_passed).lower(),
        "tool_count": str(len(tool_names)),
        "tools_used": _truncate_metadata_value(", ".join(sorted(set(tool_names))) or "none"),
        "answer_chars": str(answer_chars),
    }
    if avg_tool_accuracy is not None:
        metadata["tool_accuracy_avg"] = f"{avg_tool_accuracy:.4f}"
    return metadata


def enrich_compliance_turn_span(
    *,
    user_message: str | None,
    answer: str,
    blocked: bool,
    metadata: dict[str, str],
) -> None:
    """Attach turn I/O and metadata to the active Langfuse span.

    Parameters
    ----------
    user_message : str or None
        User question text.
    answer : str
        Final answer text.
    blocked : bool
        Whether the turn was blocked before the LLM ran.
    metadata : dict[str, str]
        Turn summary metadata (string values only).

    Notes
    -----
    No-op when Langfuse is disabled.
    """
    if not langfuse_enabled():
        return

    from langfuse import get_client

    span_input: dict[str, Any] | None = None
    if user_message:
        span_input = {"user_message": user_message[:4000]}

    get_client().update_current_span(
        input=span_input,
        output={"answer": answer[:4000], "blocked": blocked},
        metadata=metadata,
    )


def finalize_turn_observability(
    result: Any,
    *,
    handler: CallbackHandler | None = None,
    trace_id: str | None = None,
    user_message: str | None = None,
) -> None:
    """Export scores, enrich the trace span, and log a local summary.

    Parameters
    ----------
    result
        Completed turn with ``tool_trace``, ``guardrails``, ``intent``, ``answer``,
        and ``blocked`` attributes.
    handler : CallbackHandler or None, optional
        LangChain callback handler from the turn.
    trace_id : str or None, optional
        Trace id captured at turn start.
    user_message : str or None, optional
        Original user question for trace input.
    """
    from observability.turn_log import log_turn_summary

    resolved_trace_id = resolve_export_trace_id(handler, trace_id)
    export_turn_scores(
        tool_trace=list(result.tool_trace),
        guardrails=result.guardrails,
        trace_id=resolved_trace_id,
    )

    scored = [entry for entry in result.tool_trace if entry.accuracy is not None]
    avg_accuracy: float | None = None
    if scored:
        accuracies = [entry.accuracy for entry in scored if entry.accuracy is not None]
        avg_accuracy = sum(accuracies) / len(accuracies)

    tool_names = [entry.name for entry in result.tool_trace]
    metadata = build_turn_metadata(
        intent=result.intent.primary_intent,
        intent_confidence=result.intent.confidence,
        blocked=bool(result.blocked),
        guardrails_passed=bool(result.guardrails.all_passed),
        tool_names=tool_names,
        answer_chars=len(result.answer),
        avg_tool_accuracy=avg_accuracy,
    )
    enrich_compliance_turn_span(
        user_message=user_message,
        answer=result.answer,
        blocked=bool(result.blocked),
        metadata=metadata,
    )
    log_turn_summary(result, trace_id=resolved_trace_id)


def export_turn_scores(
    *,
    tool_trace: list[ToolTraceEntry],
    guardrails: GuardrailReport,
    trace_id: str | None = None,
) -> None:
    """Export tool accuracy and guardrail scores for one turn.

    Parameters
    ----------
    tool_trace : list[ToolTraceEntry]
        Tool trace entries for the turn.
    guardrails : GuardrailReport
        Composite guardrail verdict for the turn.
    trace_id : str or None, optional
        Langfuse trace id to attach scores to, by default ``None``.

    Returns
    -------
    None

    Notes
    -----
    Delegates to ``export_tool_accuracy_scores`` and ``export_guardrail_scores``.
    """
    export_tool_accuracy_scores(tool_trace, trace_id)
    export_guardrail_scores(guardrails, trace_id)


@contextmanager
def langfuse_turn_context(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[tuple[dict[str, Any], CallbackHandler | None, str | None]]:
    """Propagate trace attributes and yield LangGraph config for one turn.

    Parameters
    ----------
    session_id : str or None, optional
        Langfuse session id, by default ``None``.
    user_id : str or None, optional
        Langfuse user id, by default ``None``.
    tags : list[str] or None, optional
        Langfuse tags, by default ``None``.

    Yields
    ------
    config : dict[str, Any]
        LangGraph invoke config for the turn.
    handler : CallbackHandler or None
        Langfuse callback handler when enabled.
    trace_id : str or None
        Trace id captured at context entry.

    Notes
    -----
    When Langfuse is disabled, yields ``({}, None, None)``. When attribute
    propagation is needed, wraps the yield in ``propagate_attributes``.
    """
    if not langfuse_enabled():
        yield {}, None, None
        return

    from langfuse import propagate_attributes

    trace_id = capture_turn_trace_id()
    config, handler = build_graph_config(session_id=session_id, user_id=user_id, tags=tags)
    attrs: dict[str, Any] = {}
    if session_id:
        attrs["session_id"] = session_id
    if user_id:
        attrs["user_id"] = user_id
    if tags:
        attrs["tags"] = tags

    if attrs:
        with propagate_attributes(**attrs):
            yield config, handler, trace_id
    else:
        yield config, handler, trace_id
