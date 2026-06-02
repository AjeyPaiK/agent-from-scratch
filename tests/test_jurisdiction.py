"""Tests for EU-only jurisdiction detection.

Notes
-----
Covers both the low-level ``detect_non_eu_jurisdiction`` helper and
integration with ``check_pre_input`` for EU-scoped product questions.
"""

import pytest

from guardrails import check_pre_input
from guardrails.jurisdiction import detect_non_eu_jurisdiction


@pytest.mark.parametrize(
    "message,expected",
    [
        ("What is the status of Retinol in the US?", "United States"),
        ("What is the status of Retinol as an ingredient in India?", "India"),
        ("Is niacinamide allowed in China?", "China"),
        ("Retinol rules for the Japanese market", "Japanese"),
        ("Compare EU vs FDA rules for retinol", "non-EU regulatory regimes"),
        ("Is phenoxyethanol allowed in Asia?", "Asia"),
    ],
)
def test_detects_non_eu_jurisdictions(message, expected):
    """Detect explicit non-EU jurisdiction references in user messages.

    Parameters
    ----------
    message : str
        User message referencing a non-EU market or regulatory regime.
    expected : str
        Expected jurisdiction label returned by ``detect_non_eu_jurisdiction``.

    Notes
    -----
    Each parametrized case should resolve to a non-``None`` jurisdiction label.
    """
    assert detect_non_eu_jurisdiction(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "Is phenoxyethanol restricted in the EU?",
        "Is retinol allowed in a leave-on hand lotion?",
        "What is the Annex III limit for retinol?",
        "Can I use retinol in leave_on_face_cream at 0.3%?",
        "Is retinol allowed in Germany?",
        "CosIng status of retinol in France",
        "What is the status of Retinol?",
        "Can I use niacinamide in face cream?",
        "Tell us about retinol compliance",
        "Retinol concentration in leave-on products",
        "Is retinol allowed?",
    ],
)
def test_allows_eu_or_product_scoped_questions(message):
    """Allow EU-scoped and product-focused questions through pre-input guardrails.

    Parameters
    ----------
    message : str
        User message scoped to EU regulation or product compliance without
        explicit non-EU jurisdiction references.

    Notes
    -----
    ``detect_non_eu_jurisdiction`` should return ``None`` and
    ``check_pre_input`` should pass for each case.
    """
    assert detect_non_eu_jurisdiction(message) is None
    assert check_pre_input(message).passed
