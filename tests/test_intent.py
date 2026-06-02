"""Tests for zero-LLM intent classifier.

Notes
-----
The intent classifier is fully symbolic; these tests assert primary intent,
recommended tools, and confidence without model calls.
"""

from agent.intent import classify_intent


def test_concentration_intent():
    """Classify concentration-limit questions and recommend the concentration tool.

    Notes
    -----
    Percentage and product-category phrasing should map to
    ``concentration_limit`` with ``check_concentration_compliance`` recommended.
    """
    result = classify_intent("Is 0.8% phenoxyethanol allowed in leave-on cream?")
    assert result.primary_intent == "concentration_limit"
    assert "check_concentration_compliance" in result.recommended_tools


def test_labelling_intent():
    """Classify labelling and marketing questions and recommend the rules tool.

    Notes
    -----
    INCI and warning phrasing should map to ``labelling_rules`` with
    ``get_labelling_marketing_rules`` recommended.
    """
    result = classify_intent("What INCI list warnings apply to phenoxyethanol?")
    assert result.primary_intent == "labelling_rules"
    assert "get_labelling_marketing_rules" in result.recommended_tools


def test_general_fallback():
    """Fall back to general compliance with low confidence for broad questions.

    Notes
    -----
    Non-specific EU rules questions should use ``general_compliance`` intent
    and report ``low`` confidence.
    """
    result = classify_intent("Tell me about EU cosmetics rules")
    assert result.primary_intent == "general_compliance"
    assert result.confidence == "low"
