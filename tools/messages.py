"""Canonical user-facing messages for tool outputs."""

from __future__ import annotations

from typing import Any

ANNEX_ABSENCE_MARKER = "does not appear in any annex of Regulation (EC) No 1223/2009"


def annex_absence_message(inci_name: str) -> str:
    """Build the standard message when an INCI name is absent from annexes II–VI.

    Parameters
    ----------
    inci_name : str
        INCI name queried by the user or tool.

    Returns
    -------
    str
        CPSR-oriented explanation that the ingredient has no annex-specific
        restrictions but still requires safety assessment.
    """
    name = (inci_name or "This ingredient").strip()
    return (
        f"{name} does not appear in any annex of Regulation (EC) No 1223/2009 "
        "and is therefore not subject to specific restrictions or conditions of use. "
        "As with all cosmetic ingredients, its use must be justified by a qualified "
        "safety assessor within the Cosmetic Product Safety Report (CPSR)."
    )


def annex_absence_from_output(output: Any) -> str | None:
    """Extract the annex-absence message from a tool output dict, if present.

    Parameters
    ----------
    output : Any
        Raw tool return value (typically a ``dict``).

    Returns
    -------
    str or None
        The annex-absence ``message`` when ``found`` is false and the text matches
        the canonical marker; otherwise ``None``.
    """
    if not isinstance(output, dict) or output.get("found") is not False:
        return None
    message = output.get("message")
    if isinstance(message, str) and ANNEX_ABSENCE_MARKER in message:
        return message.strip()
    return None
