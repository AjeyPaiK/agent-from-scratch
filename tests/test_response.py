"""Tests for answer extraction and tool-scaffolding detection.

Notes
-----
Validates ``extract_final_answer``, ``is_tool_scaffolding``, and
``sanitize_model_answer`` using static message fixtures.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.response import extract_final_answer, is_tool_scaffolding, sanitize_model_answer


def test_is_tool_scaffolding_detects_call_narration():
    """Detect model text that narrates an impending tool invocation.

    Notes
    -----
    Strings describing function names, arguments, or JSON tool payloads should
    be classified as scaffolding rather than user-facing answers.
    """
    text = (
        "To answer your question, I will call the 'lookup_ingredient_regulation' "
        "function with the following arguments:\n"
        '{"name": "lookup_ingredient_regulation", "parameters": {"inci_name": "Retinol"}}'
    )
    assert is_tool_scaffolding(text)


def test_is_tool_scaffolding_allows_compliance_answer():
    """Allow substantive compliance answers that are not tool narration.

    Notes
    -----
    Annex-grounded restriction summaries with informational disclaimers should
    not be flagged as scaffolding.
    """
    text = (
        "Retinol is restricted under Annex III, entry 376. "
        "Guidance is informational — verify with official EU legal text."
    )
    assert not is_tool_scaffolding(text)


def test_extract_final_answer_skips_tool_turn():
    """Return the last non-scaffolding AI message after tool execution.

    Notes
    -----
    Intermediate tool-call narration should be skipped in favor of the final
    compliance answer message.
    """
    messages = [
        HumanMessage(content="Is retinol allowed?"),
        AIMessage(
            content="To answer your question, I will call lookup_ingredient_regulation.",
            tool_calls=[],
        ),
        ToolMessage(
            content='{"found": true}', name="lookup_ingredient_regulation", tool_call_id="t1"
        ),
        AIMessage(
            content="Retinol is restricted under Annex III, entry 376. "
            "Guidance is informational — verify with a safety assessor."
        ),
    ]
    assert extract_final_answer(messages) == (
        "Retinol is restricted under Annex III, entry 376. "
        "Guidance is informational — verify with a safety assessor."
    )


def test_sanitize_strips_based_on_tool_output_opener():
    """Remove leading "Based on the tool output" phrasing from answers.

    Notes
    -----
    The substantive compliance statement should remain after sanitization.
    """
    raw = "Based on the tool output, Retinol is restricted under Annex III."
    assert sanitize_model_answer(raw) == "Retinol is restricted under Annex III."


def test_sanitize_strips_json_leak():
    """Strip trailing JSON-response prompts leaked by the model.

    Notes
    -----
    Sanitized output should not retain instructions asking for JSON objects.
    """
    raw = "Allowed.\n\nPlease respond with a JSON object containing the result."
    assert "JSON" not in sanitize_model_answer(raw)
