"""Tests for structured tool trace parsing.

Notes
-----
Validates ``build_tool_trace`` pairing of tool-call arguments with tool message
outputs and guardrail-block metadata.
"""

import json

from langchain_core.messages import AIMessage, ToolMessage

from agent.tool_trace import build_tool_trace


def test_build_tool_trace_pairs_args_and_output():
    """Pair tool-call arguments with parsed JSON outputs in trace entries.

    Notes
    -----
    A single lookup call should produce one entry with human-readable label,
    original args, parsed output fields, and parallel name/output lists.
    """
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "lookup_ingredient_regulation",
                    "args": {"inci_name": "Phenoxyethanol"},
                }
            ],
        ),
        ToolMessage(
            content=json.dumps({"found": True, "overall_status": "restricted"}),
            name="lookup_ingredient_regulation",
            tool_call_id="call_1",
        ),
    ]

    entries, names, outputs = build_tool_trace(messages)

    assert len(entries) == 1
    assert entries[0].label == "Ingredient lookup"
    assert entries[0].args == {"inci_name": "Phenoxyethanol"}
    assert entries[0].output["overall_status"] == "restricted"
    assert names == ["lookup_ingredient_regulation"]
    assert outputs[0].startswith("{")


def test_build_tool_trace_marks_guardrail_blocks():
    """Mark trace entries blocked when tool output reports a guardrail failure.

    Notes
    -----
    Guardrail-blocked tool messages should set ``blocked=True`` and propagate
    the ``rule_id`` from the JSON payload.
    """
    messages = [
        ToolMessage(
            content=json.dumps(
                {
                    "guardrail_blocked": True,
                    "rule_id": "invalid_product_category",
                    "message": "Unknown category",
                }
            ),
            name="check_concentration_compliance",
            tool_call_id="call_2",
        ),
    ]

    entries, _, _ = build_tool_trace(messages)

    assert entries[0].blocked is True
    assert entries[0].rule_id == "invalid_product_category"
