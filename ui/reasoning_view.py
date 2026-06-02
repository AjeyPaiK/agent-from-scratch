"""Inline reasoning steps — no box, tick when done.

Notes
-----
``REASONING_CSS`` styles the HTML produced by ``reasoning_html``. Inject it
once via ``st.markdown(..., unsafe_allow_html=True)`` in the chat app.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

from agent.streaming import ReasoningStep

REASONING_CSS = """
.reasoning-inline {
  margin: 0 0 0.75rem 0;
  padding: 0;
  font-family: 'Source Sans 3', sans-serif;
  font-size: 0.76rem;
  line-height: 1.5;
}

.reasoning-step {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.12rem 0;
  color: #a8a29e;
  transition: opacity 0.3s ease;
}

.reasoning-step.done {
  opacity: 0.5;
}
.reasoning-step.active {
  opacity: 1;
  color: #78716c;
}
.stApp[data-theme="dark"] .reasoning-step.active { color: #d6d3d1; }

.reasoning-mark {
  flex-shrink: 0;
  width: 0.95rem;
  text-align: center;
  font-size: 0.72rem;
  line-height: 1;
  color: #a8a29e;
}
.reasoning-step.done .reasoning-mark { color: #78716c; }
.stApp[data-theme="dark"] .reasoning-step.done .reasoning-mark { color: #a8a29e; }

.reasoning-step.active .reasoning-mark {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #a8a29e;
  margin: 0 0.14rem;
  animation: reasoning-pulse 1.4s ease-in-out infinite;
}

.reasoning-step.active .reasoning-label {
  background: linear-gradient(90deg, #78716c 0%, #d6d3d1 50%, #78716c 100%);
  background-size: 200% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: reasoning-shimmer 2.2s ease-in-out infinite;
}
.stApp[data-theme="dark"] .reasoning-step.active .reasoning-label {
  background: linear-gradient(90deg, #d6d3d1 0%, #57534e 50%, #d6d3d1 100%);
  background-size: 200% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.reasoning-faded .reasoning-step.done { opacity: 0.35; }

@keyframes reasoning-shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}
@keyframes reasoning-pulse {
  0%, 100% { opacity: 0.55; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.2); }
}
"""


def _escape(text: str) -> str:
    """HTML-escape text for safe embedding in markup.

    Parameters
    ----------
    text : str
        Raw step label or other user-facing string.

    Returns
    -------
    str
        Escaped string safe for HTML attribute and body insertion.
    """
    return html.escape(text, quote=True)


def reasoning_html(
    steps: Sequence[ReasoningStep | dict[str, Any]],
    *,
    faded: bool = False,
) -> str:
    """Build inline HTML for agent reasoning steps.

    Parameters
    ----------
    steps : sequence of ReasoningStep or dict
        Reasoning steps to render. Dict items are coerced to ``ReasoningStep``.
    faded : bool, optional
        When ``True``, apply reduced opacity to completed steps, by default
        ``False``.

    Returns
    -------
    str
        HTML fragment for one reasoning block, or an empty string when
        ``steps`` is empty.

    Notes
    -----
    Step states ``pending``, ``active``, and ``done`` control the mark icon
    (dot, pulse, or check). Unknown states are treated as ``done``.
    """
    if not steps:
        return ""

    normalized: list[ReasoningStep] = [
        s if isinstance(s, ReasoningStep) else ReasoningStep(**s) for s in steps
    ]

    rows = []
    for step in normalized:
        state = step.state if step.state in ("pending", "active", "done") else "done"
        if state == "done":
            mark = '<span class="reasoning-mark">✓</span>'
        elif state == "active":
            mark = '<span class="reasoning-mark"></span>'
        else:
            mark = '<span class="reasoning-mark">·</span>'
        rows.append(
            f'<div class="reasoning-step {state}">'
            f"{mark}"
            f'<span class="reasoning-label">{_escape(step.label)}</span>'
            f"</div>"
        )

    fade_class = " reasoning-faded" if faded else ""
    return f'<div class="reasoning-inline{fade_class}">{"".join(rows)}</div>'
