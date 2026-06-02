"""Intent classification badge — shown alongside user messages.

Notes
-----
``INTENT_CSS`` styles badges from ``intent_badge_html``. Serialize intents
with ``intent_to_dict`` for session state; restore with ``intent_from_dict``.
"""

from __future__ import annotations

import html
from typing import Any

from agent.intent import IntentResult

INTENT_CSS = """
.intent-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  margin-top: 0.45rem;
  padding: 0.22rem 0.55rem 0.22rem 0.45rem;
  border-radius: 999px;
  font-family: 'Source Sans 3', sans-serif;
  font-size: 0.72rem;
  font-weight: 500;
  line-height: 1.2;
  letter-spacing: 0.01em;
  max-width: 100%;
}
.stApp:not([data-theme="dark"]) .intent-badge {
  background: rgba(120, 113, 108, 0.08);
  border: 1px solid rgba(120, 113, 108, 0.18);
  color: #57534e;
}
.stApp[data-theme="dark"] .intent-badge {
  background: rgba(214, 211, 209, 0.06);
  border: 1px solid rgba(214, 211, 209, 0.14);
  color: #d6d3d1;
}

.intent-dot {
  flex-shrink: 0;
  width: 6px;
  height: 6px;
  border-radius: 50%;
}
.intent-badge[data-confidence="high"] .intent-dot { background: #16a34a; }
.intent-badge[data-confidence="medium"] .intent-dot { background: #d97706; }
.intent-badge[data-confidence="low"] .intent-dot { background: #a8a29e; }

.intent-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.intent-confidence {
  flex-shrink: 0;
  font-size: 0.66rem;
  font-weight: 400;
  opacity: 0.55;
  text-transform: capitalize;
}
"""


def intent_to_dict(intent: IntentResult) -> dict[str, Any]:
    """Serialize an intent result for Streamlit session state.

    Parameters
    ----------
    intent : IntentResult
        Classified intent from the current user message.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dict with ``primary_intent``, ``label``,
        ``confidence``, and ``matched_rules``.
    """
    return {
        "primary_intent": intent.primary_intent,
        "label": intent.label,
        "confidence": intent.confidence,
        "matched_rules": list(intent.matched_rules),
    }


def intent_from_dict(data: dict[str, Any]) -> IntentResult:
    """Restore an intent result from session-state dict data.

    Parameters
    ----------
    data : dict[str, Any]
        Dict produced by ``intent_to_dict`` or equivalent persisted shape.

    Returns
    -------
    IntentResult
        Reconstructed intent with recommended tools from ``INTENT_TOOLS``.

    Notes
    -----
    Missing fields fall back to ``general_compliance`` intent defaults.
    """
    from agent.intent import INTENT_TOOLS

    primary = data.get("primary_intent", "general_compliance")
    return IntentResult(
        primary_intent=primary,
        label=data.get("label", primary.replace("_", " ").title()),
        recommended_tools=INTENT_TOOLS.get(primary, INTENT_TOOLS["general_compliance"]),
        matched_rules=list(data.get("matched_rules") or []),
        confidence=data.get("confidence", "low"),
    )


def intent_badge_html(intent: IntentResult | dict[str, Any]) -> str:
    """Build HTML for an intent classification badge.

    Parameters
    ----------
    intent : IntentResult or dict[str, Any]
        Intent to display. Dict values are converted via ``intent_from_dict``.

    Returns
    -------
    str
        HTML fragment with confidence-colored dot, label, and confidence text.

    Notes
    -----
    Unknown confidence values are displayed as ``low``.
    """
    if isinstance(intent, dict):
        intent = intent_from_dict(intent)

    confidence = intent.confidence if intent.confidence in ("high", "medium", "low") else "low"
    label = html.escape(intent.label, quote=True)

    return (
        f'<div class="intent-badge" data-confidence="{confidence}" '
        f'title="{label} · {confidence} confidence">'
        f'<span class="intent-dot"></span>'
        f'<span class="intent-label">{label}</span>'
        f'<span class="intent-confidence">{confidence}</span>'
        f"</div>"
    )
