"""Tests for labelling-tool oracle scoring."""

from __future__ import annotations

import pytest

from scoring.compare import compare_labelling_output
from scoring.oracle.labelling import oracle_get_labelling_marketing_rules
from scoring.score_tool_call import score_tool_call
from tools.labelling_rules import get_labelling_marketing_rules


def test_oracle_labelling_phenoxyethanol():
    """Labelling oracle should find phenoxyethanol and return obligation lists."""
    expected = oracle_get_labelling_marketing_rules(
        "Phenoxyethanol",
        "leave_on_face_cream",
    )
    assert expected["found"] is True
    assert expected["product_category"] == "leave_on_face_cream"
    assert len(expected["labelling_requirements"]) >= 2
    assert len(expected["marketing_restrictions"]) >= 1


def test_tool_matches_oracle_on_phenoxyethanol():
    """Live tool and CSV oracle should agree on phenoxyethanol labelling rules."""
    args = {
        "inci_name": "Phenoxyethanol",
        "product_category": "leave_on_face_cream",
    }
    actual = get_labelling_marketing_rules.invoke(args)
    expected = oracle_get_labelling_marketing_rules(**args)
    score, mismatches = compare_labelling_output(actual, expected)
    assert score == pytest.approx(1.0)
    assert mismatches == []


def test_score_tool_call_labelling():
    """score_tool_call should return a numeric accuracy for labelling tools."""
    args = {
        "inci_name": "Phenoxyethanol",
        "product_category": "leave_on_face_cream",
    }
    actual = get_labelling_marketing_rules.invoke(args)
    result = score_tool_call("get_labelling_marketing_rules", args, actual)
    assert result is not None
    assert result.accuracy == pytest.approx(1.0)
    assert result.tool == "get_labelling_marketing_rules"
