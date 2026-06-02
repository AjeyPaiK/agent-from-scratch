"""UI for pre-input guardrail blocks — distinct from normal agent answers.

Notes
-----
``BLOCKED_CSS`` styles refusal cards from ``render_blocked_turn`` and
``render_blocked_response``.
"""

from __future__ import annotations

import html

from agent.pipeline import TurnResult

BLOCKED_CSS = """
.blocked-card {
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin: 0.25rem 0 0.5rem 0;
  border-left: 4px solid #dc2626;
}
.blocked-title {
  font-weight: 600;
  font-size: 0.95rem;
  margin: 0 0 0.5rem 0;
}
.blocked-body { font-size: 0.92rem; margin: 0 0 0.5rem 0; line-height: 1.5; }
.blocked-hint {
  font-size: 0.82rem;
  margin: 0;
  opacity: 0.85;
}
.stApp:not([data-theme="dark"]) .blocked-card {
  background: #fef2f2;
  border-color: #dc2626;
  color: #991b1b;
}
.stApp[data-theme="dark"] .blocked-card {
  background: #450a0a;
  border-color: #f87171;
  color: #fecaca;
}
"""


def render_blocked_turn(*, rule_id: str, message: str) -> None:
    """Show a clear refusal when pre-input guardrails block the turn.

    Parameters
    ----------
    rule_id : str
        Guardrail rule identifier that triggered the block.
    message : str
        User-facing explanation of why the request was blocked.

    Returns
    -------
    None

    Notes
    -----
    Renders a Streamlit markdown card via ``st.markdown`` with
    ``unsafe_allow_html=True``.
    """
    import streamlit as st

    st.markdown(
        f"""
        <div class="blocked-card">
          <p class="blocked-title">Request blocked · {html.escape(rule_id)}</p>
          <p class="blocked-body">{html.escape(message)}</p>
          <p class="blocked-hint">The agent did not run — no tools were called.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_blocked_response(result: TurnResult) -> None:
    """Render a blocked-turn card from a pipeline ``TurnResult``.

    Parameters
    ----------
    result : TurnResult
        Turn outcome whose ``guardrails.pre_input`` verdict caused the block.

    Returns
    -------
    None

    Notes
    -----
    Delegates to ``render_blocked_turn`` using the pre-input guardrail verdict.
    """
    verdict = result.guardrails.pre_input
    render_blocked_turn(rule_id=verdict.rule_id, message=verdict.message)
