"""Tests for composite turn trustworthiness scoring."""

from __future__ import annotations

from agent.tool_trace import ToolTraceEntry
from guardrails.types import GuardrailReport, GuardrailVerdict
from scoring.trustworthiness import compute_turn_trustworthiness


def test_trustworthiness_perfect_turn():
    """High tool accuracy and passing guardrails yield a high composite score."""
    trace = [
        ToolTraceEntry(
            step=1,
            name="lookup_ingredient_regulation",
            label="Ingredient lookup",
            args={"inci_name": "Phenoxyethanol"},
            output={"found": True},
            accuracy=1.0,
        )
    ]
    guardrails = GuardrailReport(
        pre_input=GuardrailVerdict("pre_input", True, "ok", "accepted"),
        pre_tool=[GuardrailVerdict("pre_tool", True, "ok", "accepted")],
        post_output=GuardrailVerdict("post_output", True, "ok", "verified"),
    )

    score = compute_turn_trustworthiness(trace, guardrails)

    assert score.tool_accuracy == 1.0
    assert score.guardrail_integrity == 1.0
    assert score.value == 1.0


def test_trustworthiness_blocked_input_scores_guardrail_integrity():
    """Pre-input blocks should still score guardrail integrity at 1.0."""
    guardrails = GuardrailReport(
        pre_input=GuardrailVerdict(
            "pre_input",
            False,
            "non_eu_jurisdiction",
            "EU only",
        ),
    )

    score = compute_turn_trustworthiness([], guardrails, blocked=True)

    assert score.tool_accuracy == 1.0
    assert score.guardrail_integrity == 1.0
    assert score.value == 1.0


def test_trustworthiness_failed_post_output_reduces_score():
    """A failed output verifier should reduce the composite score."""
    trace = [
        ToolTraceEntry(
            step=1,
            name="lookup_ingredient_regulation",
            label="Ingredient lookup",
            args={"inci_name": "Phenoxyethanol"},
            output={"found": True},
            accuracy=1.0,
        )
    ]
    guardrails = GuardrailReport(
        pre_input=GuardrailVerdict("pre_input", True, "ok", "accepted"),
        pre_tool=[GuardrailVerdict("pre_tool", True, "ok", "accepted")],
        post_output=GuardrailVerdict(
            "post_output",
            False,
            "hallucinated_annex_citation",
            "bad cite",
        ),
    )

    score = compute_turn_trustworthiness(trace, guardrails)

    assert score.tool_accuracy == 1.0
    assert score.guardrail_integrity < 1.0
    assert score.value < 1.0


def test_trustworthiness_unscorable_tools_score_zero_tool_component():
    """Executed tools without oracle scores should zero the tool component."""
    trace = [
        ToolTraceEntry(
            step=1,
            name="get_labelling_marketing_rules",
            label="Labelling rules",
            args={"inci_name": "X"},
            output={"found": False},
        )
    ]
    guardrails = GuardrailReport(
        pre_input=GuardrailVerdict("pre_input", True, "ok", "accepted"),
        post_output=GuardrailVerdict("post_output", True, "ok", "verified"),
    )

    score = compute_turn_trustworthiness(trace, guardrails)

    assert score.tool_accuracy == 0.0
    assert score.value == 0.4
