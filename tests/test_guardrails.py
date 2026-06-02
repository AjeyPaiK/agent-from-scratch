"""Tests for symbolic guardrails.

Notes
-----
Covers pre-input, pre-tool, and post-output guardrail stages without invoking
an LLM or external services.
"""

from guardrails import check_post_output, check_pre_input, check_pre_tool


def test_pre_input_blocks_medical():
    """Block medical-advice requests at the pre-input stage.

    Notes
    -----
    Diagnostic or treatment questions should fail with rule id
    ``medical_advice``.
    """
    verdict = check_pre_input("Can you diagnose my rash?")
    assert not verdict.passed
    assert verdict.rule_id == "medical_advice"


def test_pre_input_blocks_us_market():
    """Block explicit United States market questions at pre-input.

    Notes
    -----
    Non-EU jurisdiction references should fail with rule id
    ``non_eu_jurisdiction``.
    """
    verdict = check_pre_input("What is the status of Retinol in a hand lotion in the US markets?")
    assert not verdict.passed
    assert verdict.rule_id == "non_eu_jurisdiction"


def test_pre_input_blocks_india():
    """Block explicit India jurisdiction questions at pre-input.

    Notes
    -----
    The rejection message should name the detected jurisdiction (``India``).
    """
    verdict = check_pre_input("What is the status of Retinol as an ingredient in India?")
    assert not verdict.passed
    assert verdict.rule_id == "non_eu_jurisdiction"
    assert "India" in verdict.message


def test_pre_input_accepts_eu_product_question():
    """Allow EU product-scoped compliance questions through pre-input.

    Notes
    -----
    Leave-on product formulation questions without non-EU jurisdiction
    references should pass.
    """
    verdict = check_pre_input("Is retinol allowed in a leave-on hand lotion?")
    assert verdict.passed


def test_pre_input_accepts_compliance_question():
    """Allow general EU regulatory compliance questions through pre-input.

    Notes
    -----
    Direct EU restriction inquiries should pass without jurisdiction blocking.
    """
    verdict = check_pre_input("Is phenoxyethanol restricted in the EU?")
    assert verdict.passed


def test_pre_input_accepts_unknown_inci_without_allowlist():
    """Allow unknown INCI names that are not on a deny list.

    Notes
    -----
    Ingredient lookup intent alone should not trigger pre-input blocking for
    common INCI names such as Glycerin.
    """
    verdict = check_pre_input("Is Glycerin allowed?")
    assert verdict.passed


def test_pre_tool_rejects_bad_product_category():
    """Reject concentration checks with an invalid product category.

    Notes
    -----
    Unknown categories such as ``face_serum`` should fail with rule id
    ``invalid_product_category``.
    """
    verdict = check_pre_tool(
        "check_concentration_compliance",
        {
            "inci_name": "Phenoxyethanol",
            "product_category": "face_serum",
            "concentration_percent": 1.0,
        },
    )
    assert not verdict.passed
    assert verdict.rule_id == "invalid_product_category"


def test_post_output_passes_without_disclaimer():
    """Pass factual compliance answers that omit a boilerplate disclaimer.

    Notes
    -----
    Answers grounded in tool usage should pass even without explicit disclaimer
    language when content is consistent with tool outputs.
    """
    verdict = check_post_output(
        "Phenoxyethanol is restricted to 1% in leave-on products.",
        tool_names_used=["lookup_ingredient_regulation"],
        tool_outputs=["Annex III, entry 46"],
    )
    assert verdict.passed


def test_post_output_passes_with_tools():
    """Pass annex-grounded answers when supporting tool outputs are present.

    Notes
    -----
    Citations aligned with lookup tool output should satisfy post-output checks.
    """
    verdict = check_post_output(
        "Per Annex III, phenoxyethanol is restricted.",
        tool_names_used=["lookup_ingredient_regulation"],
        tool_outputs=["Annex III, entry 46"],
    )
    assert verdict.passed


def test_post_output_detects_contradicts_tool_status():
    """Fail answers that contradict the tool-reported ingredient status.

    Notes
    -----
    Claiming an ingredient is allowed when the tool reports ``prohibited``
    should fail with rule id ``contradicts_tool_status``.
    """
    tool_json = '{"found": true, "overall_status": "prohibited", "annex_entries": []}'
    verdict = check_post_output(
        "Phenoxyethanol is allowed in all cosmetic products.",
        tool_names_used=["lookup_ingredient_regulation"],
        tool_outputs=[tool_json],
    )
    assert not verdict.passed
    assert verdict.rule_id == "contradicts_tool_status"


def test_pre_tool_requires_inci_on_lookup():
    """Require a non-empty INCI name for ingredient lookup tool calls.

    Notes
    -----
    Blank or whitespace-only ``inci_name`` values should fail with rule id
    ``missing_inci``.
    """
    verdict = check_pre_tool("lookup_ingredient_regulation", {"inci_name": " "})
    assert not verdict.passed
    assert verdict.rule_id == "missing_inci"
