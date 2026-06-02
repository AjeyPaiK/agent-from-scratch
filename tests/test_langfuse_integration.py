"""Tests for Langfuse score export.

Notes
-----
Langfuse client calls are mocked so score export behavior can be verified
without a live observability backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.tool_trace import ToolTraceEntry
from guardrails.types import GuardrailReport, GuardrailVerdict
from observability.langfuse_integration import (
    build_turn_metadata,
    capture_turn_trace_id,
    enrich_compliance_turn_span,
    export_guardrail_scores,
    export_tool_accuracy_scores,
    finalize_turn_observability,
    resolve_export_trace_id,
)


@patch("observability.langfuse_integration.langfuse_enabled", return_value=True)
@patch("langfuse.get_client")
def test_export_tool_accuracy_scores(mock_get_client, _enabled):
    """Export per-tool and turn-average accuracy scores to Langfuse.

    Parameters
    ----------
    mock_get_client : MagicMock
        Patched ``langfuse.get_client`` returning a mock Langfuse client.
    _enabled : MagicMock
        Patched ``langfuse_enabled`` returning ``True`` (unused beyond setup).

    Notes
    -----
    A scored lookup entry should produce two ``create_score`` calls and one
    ``flush`` call on the Langfuse client.
    """
    client = MagicMock()
    mock_get_client.return_value = client

    trace = [
        ToolTraceEntry(
            step=1,
            name="lookup_ingredient_regulation",
            label="Ingredient lookup",
            args={"inci_name": "Phenoxyethanol"},
            output={"found": True},
            accuracy=0.95,
            accuracy_mismatches=["match_type"],
            oracle_version="default",
        )
    ]

    export_tool_accuracy_scores(trace, "trace-123")

    assert client.create_score.call_count == 2
    client.create_score.assert_any_call(
        name="tool_accuracy_lookup",
        value=0.95,
        trace_id="trace-123",
        score_id="trace-123:tool_accuracy:lookup_ingredient_regulation",
        data_type="NUMERIC",
        comment="mismatches: match_type; oracle: default",
    )
    client.create_score.assert_any_call(
        name="turn_tool_accuracy_avg",
        value=0.95,
        trace_id="trace-123",
        score_id="trace-123:turn_tool_accuracy_avg",
        data_type="NUMERIC",
        comment="1 scored tool call(s)",
    )
    client.flush.assert_called_once()


@patch("observability.langfuse_integration.langfuse_enabled", return_value=True)
@patch("langfuse.get_client")
def test_export_uses_concentration_score_name(mock_get_client, _enabled):
    """Use the concentration-specific score name for concentration tool calls.

    Parameters
    ----------
    mock_get_client : MagicMock
        Patched ``langfuse.get_client`` returning a mock Langfuse client.
    _enabled : MagicMock
        Patched ``langfuse_enabled`` returning ``True`` (unused beyond setup).

    Notes
    -----
    ``check_concentration_compliance`` entries should map to
    ``tool_accuracy_concentration`` rather than the lookup score name.
    """
    client = MagicMock()
    mock_get_client.return_value = client

    trace = [
        ToolTraceEntry(
            step=1,
            name="check_concentration_compliance",
            label="Concentration check",
            args={"inci_name": "Retinol"},
            output={"found": True},
            accuracy=0.9,
        )
    ]

    export_tool_accuracy_scores(trace, "trace-123")

    client.create_score.assert_any_call(
        name="tool_accuracy_concentration",
        value=0.9,
        trace_id="trace-123",
        score_id="trace-123:tool_accuracy:check_concentration_compliance",
        data_type="NUMERIC",
        comment=None,
    )


@patch("observability.langfuse_integration.langfuse_enabled", return_value=True)
@patch("langfuse.get_client")
def test_export_status_score_when_tools_not_scorable(mock_get_client, _enabled):
    """Export a zero status score when no tool output is scorable.

    Parameters
    ----------
    mock_get_client : MagicMock
        Patched ``langfuse.get_client`` returning a mock Langfuse client.
    _enabled : MagicMock
        Patched ``langfuse_enabled`` returning ``True`` (unused beyond setup).

    Notes
    -----
    Tools without oracle scoring (for example, labelling rules) should produce
    a single ``tool_accuracy_status`` score with value 0.0.
    """
    client = MagicMock()
    mock_get_client.return_value = client

    export_tool_accuracy_scores(
        [
            ToolTraceEntry(
                step=1,
                name="get_labelling_marketing_rules",
                label="Labelling rules",
                args={},
                output={},
            )
        ],
        "trace-123",
    )

    client.create_score.assert_called_once_with(
        name="tool_accuracy_status",
        value=0.0,
        trace_id="trace-123",
        score_id="trace-123:tool_accuracy_status",
        data_type="NUMERIC",
        comment="No scorable tool output; tools used: get_labelling_marketing_rules",
    )


@patch("observability.langfuse_integration.langfuse_enabled", return_value=True)
@patch("langfuse.get_client")
def test_export_guardrail_scores(mock_get_client, _enabled):
    """Export per-stage and aggregate guardrail pass/fail scores.

    Parameters
    ----------
    mock_get_client : MagicMock
        Patched ``langfuse.get_client`` returning a mock Langfuse client.
    _enabled : MagicMock
        Patched ``langfuse_enabled`` returning ``True`` (unused beyond setup).

    Notes
    -----
    A report with a failing post-output verdict should still export scores for
    all guardrail stages plus the aggregate ``guardrails_all_passed`` score.
    """
    client = MagicMock()
    mock_get_client.return_value = client

    report = GuardrailReport(
        pre_input=GuardrailVerdict("pre_input", True, "ok", "Input accepted."),
        pre_tool=[GuardrailVerdict("pre_tool", True, "ok", "accepted")],
        post_output=GuardrailVerdict(
            "post_output", False, "hallucinated_annex_citation", "bad cite"
        ),
    )
    export_guardrail_scores(report, "trace-456")

    names = {call.kwargs["name"] for call in client.create_score.call_args_list}
    assert names == {
        "guardrail_pre_input",
        "guardrail_pre_tool",
        "guardrail_post_output",
        "guardrails_all_passed",
    }
    client.flush.assert_called_once()


@patch("observability.langfuse_integration.langfuse_enabled", return_value=False)
def test_export_skipped_when_disabled(_enabled):
    """Skip score export entirely when Langfuse integration is disabled.

    Parameters
    ----------
    _enabled : MagicMock
        Patched ``langfuse_enabled`` returning ``False``.

    Notes
    -----
    No Langfuse client should be contacted when integration is turned off.
    """
    export_tool_accuracy_scores([], "trace-123")


@patch("observability.langfuse_integration.langfuse_enabled", return_value=True)
@patch("langfuse.get_client")
def test_capture_turn_trace_id(mock_get_client, _enabled):
    """Return the current observe-context trace id from the Langfuse client.

    Parameters
    ----------
    mock_get_client : MagicMock
        Patched ``langfuse.get_client`` returning a mock Langfuse client.
    _enabled : MagicMock
        Patched ``langfuse_enabled`` returning ``True`` (unused beyond setup).

    Notes
    -----
    The captured trace id should match ``get_current_trace_id`` on the client.
    """
    client = MagicMock()
    client.get_current_trace_id.return_value = "observe-trace-123"
    mock_get_client.return_value = client

    assert capture_turn_trace_id() == "observe-trace-123"


def test_build_turn_metadata_truncates_long_tool_lists():
    """Cap tools_used metadata at Langfuse's 200-character limit."""
    tools = [f"tool_{i}" for i in range(40)]
    metadata = build_turn_metadata(
        intent="general_compliance",
        intent_confidence="low",
        blocked=False,
        guardrails_passed=True,
        tool_names=tools,
        answer_chars=120,
        avg_tool_accuracy=0.88,
    )
    assert len(metadata["tools_used"]) <= 200
    assert metadata["tool_accuracy_avg"] == "0.8800"


@patch("observability.langfuse_integration.langfuse_enabled", return_value=True)
@patch("langfuse.get_client")
def test_enrich_compliance_turn_span(mock_get_client, _enabled):
    """Attach turn I/O and metadata to the active Langfuse span."""
    client = MagicMock()
    mock_get_client.return_value = client

    enrich_compliance_turn_span(
        user_message="Is phenoxyethanol allowed?",
        answer="Yes, within limits.",
        blocked=False,
        metadata={"intent": "ingredient_allowed"},
    )

    client.update_current_span.assert_called_once()
    kwargs = client.update_current_span.call_args.kwargs
    assert kwargs["metadata"]["intent"] == "ingredient_allowed"
    assert kwargs["output"]["blocked"] is False


@patch("observability.turn_log.log_turn_summary")
@patch("observability.langfuse_integration.enrich_compliance_turn_span")
@patch("observability.langfuse_integration.export_turn_scores")
@patch("observability.langfuse_integration.langfuse_enabled", return_value=True)
@patch("observability.langfuse_integration.resolve_export_trace_id", return_value="trace-789")
def test_finalize_turn_observability(
    mock_resolve,
    _enabled,
    mock_export_scores,
    mock_enrich,
    mock_log,
):
    """Finalize exports scores, enriches the span, and logs locally."""
    from agent.intent import IntentResult
    from agent.pipeline import TurnResult
    from guardrails.types import GuardrailReport, GuardrailVerdict

    result = TurnResult(
        answer="ok",
        intent=IntentResult(
            primary_intent="ingredient_allowed",
            label="Allowed?",
            recommended_tools=["lookup_ingredient_regulation"],
            confidence="high",
        ),
        guardrails=GuardrailReport(
            pre_input=GuardrailVerdict("pre_input", True, "ok", "accepted"),
        ),
        blocked=False,
    )

    finalize_turn_observability(result, user_message="question?", trace_id="trace-789")

    mock_export_scores.assert_called_once()
    mock_enrich.assert_called_once()
    mock_log.assert_called_once_with(result, trace_id="trace-789")


@patch("langfuse.get_client")
def test_resolve_export_trace_id_prefers_handler(mock_get_client):
    """Prefer the callback handler trace id over the observe-context id.

    Parameters
    ----------
    mock_get_client : MagicMock
        Patched ``langfuse.get_client`` whose ``get_current_trace_id`` returns
        ``None``.

    Notes
    -----
    When the Langfuse client has no current trace, the handler's
    ``last_trace_id`` should be used for score export.
    """
    client = MagicMock()
    client.get_current_trace_id.return_value = None
    mock_get_client.return_value = client

    handler = MagicMock()
    handler.last_trace_id = "handler-trace-456"

    assert resolve_export_trace_id(handler, "observe-trace-123") == "handler-trace-456"
