"""Tests for streaming turn events without a live LLM.

Notes
-----
Graph and LLM dependencies are monkeypatched so streaming behavior can be
asserted deterministically without network or model calls.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from agent.intent import classify_intent
from agent.response import sanitize_model_answer
from agent.streaming import ReasoningStep, apply_event_to_steps, stream_turn
from guardrails.types import GuardrailVerdict


def _ok(stage: str) -> GuardrailVerdict:
    """Build a passing guardrail verdict for a pipeline stage.

    Parameters
    ----------
    stage : str
        Guardrail stage identifier (for example, ``"pre_input"``).

    Returns
    -------
    GuardrailVerdict
        Verdict marked as passed with rule id ``"ok"`` and an empty message.
    """
    return GuardrailVerdict(stage=stage, passed=True, rule_id="ok", message="")


def test_stream_turn_blocked_before_agent(monkeypatch):
    """Emit a blocked event and skip the LLM when pre-input guardrails fail.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to prevent ``build_chat_llm`` from being invoked.

    Notes
    -----
    A non-EU jurisdiction question should block before the agent graph runs.
    """

    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called when pre-input blocks")

    monkeypatch.setattr("agent.llm.build_chat_llm", boom)

    events = list(stream_turn("What is Retinol allowed at in the US market?"))
    kinds = [e.kind for e in events]

    assert "intent" in kinds
    assert "blocked" in kinds
    assert kinds[-1] == "done"
    assert events[-1].result is not None
    assert events[-1].result.blocked is True
    assert events[-1].result.guardrails.pre_input.rule_id == "non_eu_jurisdiction"


class _FakeGraph:
    """Emulate a compiled graph multi-mode stream for one tool-using turn.

    Parameters
    ----------
    narration : str
        ReAct-loop narration that must not appear in streamed answer tokens.
    answer_tokens : list of str
        Token chunks emitted by the exposition node.
    """

    def __init__(self, narration: str, answer_tokens: list[str]):
        self._narration = narration
        self._answer_tokens = answer_tokens

    def stream(self, _init, stream_mode=None, **kwargs):
        """Yield graph stream events for guardrails, tools, and exposition.

        Parameters
        ----------
        _init : Any
            Initial graph state (unused by this fake implementation).
        stream_mode : Any, optional
            LangGraph stream mode selector (unused by this fake implementation).
        **kwargs : Any
            Additional stream kwargs forwarded by LangGraph (ignored).

        Yields
        ------
        tuple
            ``(mode, payload)`` pairs matching LangGraph multi-mode streaming.
        """
        yield ("updates", {"input_guard": {"pre_input": _ok("pre_input")}})
        yield (
            "updates",
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "lookup_ingredient_regulation",
                                    "args": {"inci_name": "Retinol"},
                                    "id": "t1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }
            },
        )
        # ReAct loop narration must never reach the answer stream.
        yield ("messages", (AIMessageChunk(content=self._narration), {"langgraph_node": "agent"}))
        yield (
            "updates",
            {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content='{"found": true}',
                            name="lookup_ingredient_regulation",
                            tool_call_id="t1",
                        )
                    ]
                }
            },
        )
        for tok in self._answer_tokens:
            yield ("messages", (AIMessageChunk(content=tok), {"langgraph_node": "exposition"}))
        yield ("updates", {"output_verify": {"post_output": _ok("post_output")}})
        yield (
            "values",
            {
                "messages": [HumanMessage(content="q")],
                "user_message": "Is Retinol allowed?",
                "intent": classify_intent("Is Retinol allowed?"),
                "pre_input": _ok("pre_input"),
                "pre_tool": [_ok("pre_tool")],
                "post_output": _ok("post_output"),
                "answer": "".join(self._answer_tokens),
                "tool_trace": [],
                "blocked": False,
            },
        )


def test_stream_turn_yields_tokens(monkeypatch):
    """Stream exposition tokens and assemble the final answer on completion.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to inject ``_FakeGraph`` as the compiled graph.

    Notes
    -----
    Token events should reflect only exposition output, not ReAct narration.
    """
    monkeypatch.setattr(
        "agent.streaming.get_graph",
        lambda: _FakeGraph(narration="loop narration", answer_tokens=["Hello", " world"]),
    )

    events = list(stream_turn("Is Retinol allowed?"))
    tokens = [e.token for e in events if e.kind == "token"]
    assert tokens == ["Hello", " world"]
    assert events[-1].kind == "done"
    assert events[-1].result.answer == "Hello world"


def test_stream_emits_only_exposition_tokens(monkeypatch):
    """Exclude ReAct-loop narration from streamed answer tokens.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to inject ``_FakeGraph`` with loop narration.

    Notes
    -----
    Only the exposition node may stream answer tokens; agent-node narration
    must never appear in the token stream.
    """
    monkeypatch.setattr(
        "agent.streaming.get_graph",
        lambda: _FakeGraph(
            narration="loop narration that must not stream",
            answer_tokens=["Clean answer."],
        ),
    )

    tokens = [e.token for e in stream_turn("Is Retinol allowed?") if e.kind == "token"]
    assert tokens == ["Clean answer."]
    assert "narration" not in "".join(tokens)


def test_sanitize_model_answer_strips_tool_leakage():
    """Remove tool-call scaffolding and JSON leakage from model answers.

    Notes
    -----
    Compliance text should remain while trailing tool-invocation prompts
    and stray JSON delimiters are stripped.
    """
    raw = (
        "Retinol is restricted.\n\n"
        "Please note that guidance is informational.\n\n"
        "}\n\n"
        "Please provide the tool call response for further assistance."
    )
    cleaned = sanitize_model_answer(raw)
    assert "Retinol is restricted." in cleaned
    assert "Please provide the tool call" not in cleaned
    assert cleaned.endswith("informational.")


def test_apply_event_to_steps_marks_active_and_done():
    """Transition reasoning steps through active and done states by event kind.

    Notes
    -----
    Intent events do not create steps; phase events activate a step; answer
    start and token events mark prior steps as done.
    """
    from agent.streaming import StreamEvent

    steps: list[ReasoningStep] = []
    steps = apply_event_to_steps(steps, StreamEvent(kind="intent", intent=classify_intent("x")))
    assert steps == []
    steps = apply_event_to_steps(steps, StreamEvent(kind="phase", phase="Checking guardrails"))
    assert steps[-1].state == "active"
    steps = apply_event_to_steps(steps, StreamEvent(kind="answer_start"))
    assert steps[0].state == "done"
    steps = apply_event_to_steps(steps, StreamEvent(kind="token", token="Hi"))
    assert all(s.state == "done" for s in steps)
