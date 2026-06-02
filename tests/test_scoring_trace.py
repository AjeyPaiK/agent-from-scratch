"""Tests for accuracy fields on tool trace entries.

Notes
-----
Validates ``apply_accuracy_scores`` integration with ``build_tool_trace`` and
blocked tool outputs that skip scoring.
"""

import json
from dataclasses import replace

from langchain_core.messages import AIMessage, ToolMessage

from agent.tool_trace import ToolTraceEntry, build_tool_trace
from scoring.trace import apply_accuracy_scores


def test_apply_accuracy_scores_on_lookup():
    """Score a lookup tool trace entry against the live tool oracle output.

    Notes
    -----
    After replacing parsed message output with the real lookup invocation
    result, accuracy should be 1.0 with the default oracle version.
    """
    args = {"inci_name": "Phenoxyethanol"}
    output = {"found": True, "overall_status": "positive_list_preservative", "match_type": "exact"}
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "lookup_ingredient_regulation", "args": args}],
        ),
        ToolMessage(
            content=json.dumps(output),
            name="lookup_ingredient_regulation",
            tool_call_id="call_1",
        ),
    ]
    entries, _, _ = build_tool_trace(messages)
    # build_tool_trace stores parsed output; re-run scoring with real tool output
    from tools.lookup_ingredient import lookup_ingredient_regulation

    real_output = lookup_ingredient_regulation.invoke(args)
    entry = replace(entries[0], output=real_output)
    scored = apply_accuracy_scores([entry])
    assert scored[0].accuracy == 1.0
    assert scored[0].oracle_version == "default"


def test_apply_accuracy_scores_skips_blocked():
    """Leave accuracy unset for guardrail-blocked tool trace entries.

    Notes
    -----
    Entries marked ``blocked`` should not receive oracle accuracy scores.
    """
    entry = ToolTraceEntry(
        step=1,
        name="lookup_ingredient_regulation",
        label="Ingredient lookup",
        output={"guardrail_blocked": True},
        blocked=True,
        rule_id="test",
    )
    scored = apply_accuracy_scores([entry])
    assert scored[0].accuracy is None
