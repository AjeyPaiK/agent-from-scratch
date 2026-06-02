"""Tests for Tool 2 concentration oracle and accuracy scoring.

Notes
-----
Validates oracle expectations, tool output parity, and scoring behavior for
``check_concentration_compliance`` against the concentration oracle.
"""

from __future__ import annotations

import json

from agent.tool_args import sanitize_tool_kwargs
from scoring.compare import compare_concentration_output
from scoring.oracle.concentration import oracle_check_concentration_compliance
from scoring.score_tool_call import score_tool_call
from tools.check_concentration import check_concentration_compliance


def test_oracle_concentration_phenoxyethanol():
    """Return compliant concentration limits for phenoxyethanol in leave-on cream.

    Notes
    -----
    Phenoxyethanol at 0.8% in ``leave_on_face_cream`` should be found, compliant,
    and limited to a 1.0% maximum.
    """
    expected = oracle_check_concentration_compliance(
        "Phenoxyethanol",
        "leave_on_face_cream",
        concentration_percent=0.8,
    )
    assert expected["found"] is True
    assert expected["compliant"] is True
    assert expected["limits"][0]["max_concentration_percent"] == 1.0


def test_tool_matches_oracle_on_phenoxyethanol():
    """Match oracle output for a compliant phenoxyethanol concentration check.

    Notes
    -----
    Tool invocation and oracle lookup should agree with a perfect accuracy score
    and no field mismatches.
    """
    args = {
        "inci_name": "Phenoxyethanol",
        "product_category": "leave_on_face_cream",
        "concentration_percent": 0.8,
    }
    actual = check_concentration_compliance.invoke(args)
    expected = oracle_check_concentration_compliance(**args)
    score, mismatches = compare_concentration_output(actual, expected)
    assert score == 1.0
    assert mismatches == []


def test_tool_matches_oracle_on_retinol_non_compliant():
    """Match oracle output for a non-compliant retinol concentration check.

    Notes
    -----
    Retinol at 0.5% in leave-on face cream should be flagged non-compliant by
    both the tool and the oracle.
    """
    args = {
        "inci_name": "Retinol",
        "product_category": "leave_on_face_cream",
        "concentration_percent": 0.5,
    }
    actual = check_concentration_compliance.invoke(args)
    expected = oracle_check_concentration_compliance(**args)
    score, mismatches = compare_concentration_output(actual, expected)
    assert actual["compliant"] is False
    assert expected["compliant"] is False
    assert score == 1.0
    assert mismatches == []


def test_score_tool_call_concentration():
    """Score a concentration tool call with perfect oracle accuracy.

    Notes
    -----
    ``score_tool_call`` should return a result with accuracy 1.0 and no
    mismatches for a known compliant ingredient.
    """
    args = {
        "inci_name": "Phenoxyethanol",
        "product_category": "leave_on_face_cream",
        "concentration_percent": 0.8,
    }
    actual = check_concentration_compliance.invoke(args)
    result = score_tool_call("check_concentration_compliance", args, actual)
    assert result is not None
    assert result.accuracy == 1.0
    assert result.mismatches == []


def test_score_tool_call_concentration_string_percent():
    """Score concentration checks after coercing string percent arguments.

    Notes
    -----
    ``sanitize_tool_kwargs`` should normalize string concentration values before
    scoring so accuracy remains perfect.
    """
    args = {
        "inci_name": "Phenoxyethanol",
        "product_category": "leave_on_face_cream",
        "concentration_percent": "0.8",
    }
    actual = check_concentration_compliance.invoke(
        sanitize_tool_kwargs("check_concentration_compliance", args)
    )
    result = score_tool_call("check_concentration_compliance", args, actual)
    assert result is not None
    assert result.accuracy == 1.0


def test_score_tool_call_concentration_not_found():
    """Apply the not-found partial credit score for unknown ingredients.

    Notes
    -----
    Unknown INCI names should yield ``found=False`` and a reduced compare score
    of 0.15 with no field mismatches when both sides agree on not found.
    """
    args = {
        "inci_name": "TotallyFakeIngredientXYZ123",
        "product_category": "leave_on_face_cream",
    }
    actual = check_concentration_compliance.invoke(args)
    expected = oracle_check_concentration_compliance(**args)
    score, mismatches = compare_concentration_output(actual, expected)
    assert actual["found"] is False
    assert score == 0.15
    assert mismatches == []


def test_compare_detects_wrong_compliant():
    """Detect a mismatch when the compliant flag disagrees with the oracle.

    Notes
    -----
    Forcing ``compliant`` to ``False`` on an otherwise matching payload should
    reduce the score and report ``compliant`` in mismatches.
    """
    args = {
        "inci_name": "Phenoxyethanol",
        "product_category": "leave_on_face_cream",
        "concentration_percent": 0.8,
    }
    expected = oracle_check_concentration_compliance(**args)
    actual = json.loads(json.dumps(expected))
    actual["compliant"] = False
    score, mismatches = compare_concentration_output(actual, expected)
    assert score < 1.0
    assert "compliant" in mismatches
