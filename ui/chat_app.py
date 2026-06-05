"""EU cosmetics compliance chat — streaming ReAct agent UI."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.intent import classify_intent  # noqa: E402
from agent.pipeline import TurnResult  # noqa: E402
from agent.streaming import ReasoningStep, apply_event_to_steps, stream_turn  # noqa: E402
from config.settings import OLLAMA_MODEL  # noqa: E402
from ui.blocked_view import BLOCKED_CSS, render_blocked_response, render_blocked_turn  # noqa: E402
from ui.chat_theme import CHAT_THEME_CSS  # noqa: E402
from ui.intent_view import INTENT_CSS, intent_badge_html, intent_to_dict  # noqa: E402
from ui.reasoning_view import REASONING_CSS, reasoning_html  # noqa: E402

st.set_page_config(page_title="CosIng Agent", page_icon="◆", layout="centered")

st.markdown(
    f"<style>{CHAT_THEME_CSS}{REASONING_CSS}{BLOCKED_CSS}{INTENT_CSS}</style>",
    unsafe_allow_html=True,
)


def _render_header() -> None:
    """Render the page title and model subtitle.

    Returns
    -------
    None
    """
    st.markdown(
        f"""
        <div class="chat-header">
          <h1>CosIng Agent</h1>
          <p>EU ingredient compliance · {OLLAMA_MODEL}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_welcome() -> None:
    """Render the empty-state welcome card.

    Returns
    -------
    None
    """
    st.markdown(
        """
        <div class="welcome-card">
          Ask about INCI ingredients, concentration limits, and labelling
          under Regulation (EC) No 1223/2009. Answers cite CosIng annex data.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _paint_answer(slot, text: str, *, streaming: bool = False) -> None:
    """Write assistant answer text to a Streamlit placeholder.

    Parameters
    ----------
    slot : streamlit.empty
        Placeholder container for the answer body.
    text : str
        Markdown answer text to display.
    streaming : bool, optional
        When ``True``, append a block cursor for in-progress streaming,
        by default ``False``.

    Returns
    -------
    None

    Notes
    -----
    Typography is controlled by chat-message CSS in ``CHAT_THEME_CSS``.
    """
    body = text or " "
    if streaming:
        slot.markdown(body + " ▌")
    else:
        slot.markdown(body)


def _history_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """Convert session message dicts to LangChain chat history.

    Parameters
    ----------
    messages : list[dict]
        Session messages with ``role``, ``content``, and optional ``raw_answer``.

    Returns
    -------
    list[BaseMessage]
        LangChain messages for prior turns only (user and assistant).

    Notes
    -----
    Assistant entries use ``raw_answer`` when present so disclaimers stripped
    for display do not pollute model context.
    """
    history: list[BaseMessage] = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m.get("raw_answer"):
            history.append(AIMessage(content=m["raw_answer"]))
    return history


def _langfuse_session_id() -> str:
    """Return a stable Langfuse session id for this Streamlit session.

    Returns
    -------
    str
        UUID stored in ``st.session_state.langfuse_session_id``.

    Notes
    -----
    Creates and persists the id on first access within the session.
    """
    if "langfuse_session_id" not in st.session_state:
        st.session_state.langfuse_session_id = str(uuid.uuid4())
    session_id = st.session_state.langfuse_session_id
    return str(session_id)


def _consume_stream(prompt: str, history: list[BaseMessage]) -> TurnResult:
    """Stream one agent turn, updating reasoning and answer placeholders.

    Parameters
    ----------
    prompt : str
        Current user message text.
    history : list[BaseMessage]
        Prior conversation for the agent pipeline.

    Returns
    -------
    TurnResult
        Final turn outcome including answer, intent, and guardrails.

    Notes
    -----
    Reasoning steps render in ``reasoning_slot`` until ``answer_start``; answer
    tokens stream in ``answer_slot`` with throttled UI updates. Blocked turns
    call ``render_blocked_response`` before returning.
    """
    reasoning_slot = st.empty()
    answer_slot = st.empty()

    steps: list[ReasoningStep] = []
    answer_parts: list[str] = []
    result: TurnResult | None = None
    last_ui = 0.0
    throttle_s = 0.04
    answer_visible = False

    for event in stream_turn(prompt, history=history, session_id=_langfuse_session_id()):
        steps = apply_event_to_steps(steps, event)

        if event.kind == "blocked" and event.result:
            result = event.result

        if event.kind == "answer_start":
            reasoning_slot.empty()
            answer_visible = True

        if event.kind == "token" and event.token:
            answer_visible = True
            answer_parts.append(event.token)
            now = time.monotonic()
            if now - last_ui >= throttle_s:
                _paint_answer(answer_slot, "".join(answer_parts), streaming=True)
                last_ui = now

        if event.kind in ("intent", "phase", "tool_start", "tool_end") and not answer_visible:
            reasoning_slot.markdown(reasoning_html(steps), unsafe_allow_html=True)
            answer_slot.empty()

        if event.kind == "done" and event.result:
            result = event.result

    if result is None:
        from guardrails import check_pre_input
        from guardrails.types import GuardrailReport

        pre = check_pre_input(prompt)
        guardrails = GuardrailReport(pre_input=pre)
        from scoring.trustworthiness import compute_turn_trustworthiness

        result = TurnResult(
            answer="".join(answer_parts) or "No response.",
            intent=classify_intent(prompt),
            guardrails=guardrails,
            trustworthiness=compute_turn_trustworthiness([], guardrails),
        )

    reasoning_slot.empty()
    answer_slot.empty()

    if result.blocked:
        render_blocked_response(result)

    return result


def _last_turn_needs_reply() -> bool:
    """Return whether the latest session message is an unanswered user turn.

    Returns
    -------
    bool
        ``True`` when messages exist and the last entry has role ``user``.
    """
    msgs = st.session_state.messages
    return bool(msgs) and msgs[-1]["role"] == "user"


_render_header()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    _render_welcome()

for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        if message.get("blocked"):
            render_blocked_turn(
                rule_id=message.get("rule_id", "pre_input"),
                message=message["content"],
            )
        elif role == "assistant":
            st.markdown(message["content"] or " ")
        else:
            st.markdown(message["content"])
            if message.get("intent"):
                st.markdown(intent_badge_html(message["intent"]), unsafe_allow_html=True)

if prompt := st.chat_input("Ask about an ingredient, concentration, or labelling…"):
    intent = classify_intent(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "intent": intent_to_dict(intent)},
    )
    st.rerun()

if _last_turn_needs_reply():
    last_user = st.session_state.messages[-1]
    with st.chat_message("assistant"):
        history = _history_to_langchain(st.session_state.messages[:-1])
        result = _consume_stream(last_user["content"], history)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "raw_answer": None if result.blocked else result.answer.split("\n\n---\n⚠️")[0],
            "blocked": result.blocked,
            "rule_id": result.guardrails.pre_input.rule_id if result.blocked else None,
        },
    )
    st.rerun()
