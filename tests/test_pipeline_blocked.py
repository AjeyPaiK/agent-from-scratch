"""Tests for blocked turn handling in the agent pipeline.

Notes
-----
These tests verify that pre-input guardrail failures short-circuit the
pipeline before any LLM invocation occurs.
"""

from agent.pipeline import run_turn


def test_blocked_turn_does_not_invoke_llm(monkeypatch):
    """Return a blocked result without calling the LLM for non-EU questions.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to prevent ``build_chat_llm`` from being invoked.

    Notes
    -----
    Non-EU jurisdiction questions should block with an empty tool trace and
    a guardrail rule id of ``non_eu_jurisdiction``.
    """

    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called when pre-input blocks")

    monkeypatch.setattr("agent.llm.build_chat_llm", boom)

    result = run_turn("What is Retinol allowed at in the US market?")

    assert result.blocked is True
    assert result.tool_trace == []
    assert result.guardrails.pre_input.rule_id == "non_eu_jurisdiction"
    assert "EU Regulation" in result.answer
