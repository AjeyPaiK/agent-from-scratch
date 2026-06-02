"""Tests for CSV oracle lookup and accuracy scoring.

Notes
-----
Validates lookup oracle expectations, tool-oracle parity, and scoring behavior
for ``lookup_ingredient_regulation`` against the CSV-backed oracle.
"""

from __future__ import annotations

import json

from scoring.compare import compare_lookup_output
from scoring.oracle.lookup import oracle_lookup_ingredient_regulation
from scoring.score_tool_call import score_tool_call
from tools.lookup_ingredient import lookup_ingredient_regulation


def test_oracle_lookup_phenoxyethanol():
    """Return expected annex metadata for phenoxyethanol from the lookup oracle.

    Notes
    -----
    Phenoxyethanol should be found with preservative positive-list status,
    exact match type, and at least one annex entry.
    """
    expected = oracle_lookup_ingredient_regulation("Phenoxyethanol")
    assert expected["found"] is True
    assert expected["overall_status"] == "positive_list_preservative"
    assert expected["match_type"] == "exact"
    assert len(expected["annex_entries"]) >= 1


def test_tool_matches_oracle_on_phenoxyethanol():
    """Match oracle output for a phenoxyethanol ingredient lookup.

    Notes
    -----
    Tool invocation and oracle lookup should agree with a perfect accuracy score
    and no field mismatches.
    """
    args = {"inci_name": "Phenoxyethanol"}
    actual = lookup_ingredient_regulation.invoke(args)
    expected = oracle_lookup_ingredient_regulation(**args)
    score, mismatches = compare_lookup_output(actual, expected)
    assert score == 1.0
    assert mismatches == []


def test_score_tool_call_lookup():
    """Score a lookup tool call with perfect oracle accuracy.

    Notes
    -----
    ``score_tool_call`` should return accuracy 1.0, no mismatches, and the
    default oracle version for phenoxyethanol.
    """
    args = {"inci_name": "Phenoxyethanol"}
    actual = lookup_ingredient_regulation.invoke(args)
    result = score_tool_call("lookup_ingredient_regulation", args, actual)
    assert result is not None
    assert result.accuracy == 1.0
    assert result.mismatches == []
    assert result.oracle_version == "default"


def test_score_tool_call_not_found():
    """Apply partial credit when both tool and oracle report ingredient not found.

    Notes
    -----
    Unknown INCI names should yield ``found=False`` on both sides and a compare
    score of 0.2 with no field mismatches.
    """
    args = {"inci_name": "TotallyFakeIngredientXYZ123"}
    actual = lookup_ingredient_regulation.invoke(args)
    expected = oracle_lookup_ingredient_regulation(**args)
    score, mismatches = compare_lookup_output(actual, expected)
    assert actual["found"] is False
    assert expected["found"] is False
    assert score == 0.2  # found field only
    assert mismatches == []


def test_score_tool_call_blocked_output_skipped():
    """Skip scoring when tool output indicates a guardrail block.

    Notes
    -----
    Outputs containing ``guardrail_blocked`` should cause ``score_tool_call``
    to return ``None``.
    """
    blocked = {"guardrail_blocked": True, "rule_id": "test", "message": "blocked"}
    assert score_tool_call("lookup_ingredient_regulation", {"inci_name": "X"}, blocked) is None


def test_compare_detects_wrong_status():
    """Detect a mismatch when overall_status disagrees with the oracle.

    Notes
    -----
    Forcing ``overall_status`` to ``prohibited`` on an otherwise matching payload
    should reduce the score and report the field in mismatches.
    """
    expected = oracle_lookup_ingredient_regulation("Phenoxyethanol")
    actual = json.loads(json.dumps(expected))
    actual["overall_status"] = "prohibited"
    score, mismatches = compare_lookup_output(actual, expected)
    assert score < 1.0
    assert "overall_status" in mismatches
