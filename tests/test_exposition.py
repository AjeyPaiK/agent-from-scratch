"""Tests for exposition direct-answer shortcuts.

Notes
-----
Validates ``direct_annex_absence_answer`` behavior for single lookup misses
versus mixed tool-trace results.
"""

from agent.exposition import direct_annex_absence_answer
from agent.tool_trace import ToolTraceEntry
from tools.messages import annex_absence_message


def test_direct_annex_absence_single_lookup():
    """Return the annex absence message for a lone not-found lookup.

    Notes
    -----
    When exactly one lookup reports ``found=False``, the standardized absence
    message should be returned directly without LLM exposition.
    """
    msg = annex_absence_message("Glycerin")
    trace = [
        ToolTraceEntry(
            step=1,
            name="lookup_ingredient_regulation",
            label="Ingredient lookup",
            args={"inci_name": "Glycerin"},
            output={"found": False, "inci_name": "Glycerin", "message": msg},
        )
    ]
    assert direct_annex_absence_answer(trace) == msg


def test_direct_annex_absence_skipped_when_mixed_results():
    """Skip the direct absence shortcut when the trace has mixed tool outcomes.

    Notes
    -----
    A not-found lookup combined with other tool results should return ``None``
    so the normal exposition path can synthesize an answer.
    """
    msg = annex_absence_message("Glycerin")
    trace = [
        ToolTraceEntry(
            step=1,
            name="lookup_ingredient_regulation",
            label="Ingredient lookup",
            output={"found": False, "message": msg},
        ),
        ToolTraceEntry(
            step=2,
            name="check_concentration_compliance",
            label="Concentration check",
            output={"found": True, "overall_status": "restricted"},
        ),
    ]
    assert direct_annex_absence_answer(trace) is None
