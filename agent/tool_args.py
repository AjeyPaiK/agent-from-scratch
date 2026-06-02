"""Normalize LLM tool arguments before Pydantic validation."""

from __future__ import annotations

from typing import Any


def _is_empty(value: Any) -> bool:
    """Return whether a tool argument value should be treated as absent."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


def sanitize_tool_kwargs(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Coerce common malformed optional arguments from tool-calling LLMs.

    Parameters
    ----------
    tool_name : str
        Name of the tool being invoked.
    kwargs : dict[str, Any]
        Raw keyword arguments emitted by the model.

    Returns
    -------
    dict[str, Any]
        Cleaned arguments with empty or invalid optional fields removed or coerced.
    """
    cleaned = dict(kwargs)

    for key in ("cas_number",):
        val = cleaned.get(key)
        if _is_empty(val) or not isinstance(val, str):
            cleaned.pop(key, None)

    if tool_name == "check_concentration_compliance":
        val = cleaned.get("concentration_percent")
        if _is_empty(val):
            cleaned.pop("concentration_percent", None)
        elif isinstance(val, str):
            try:
                cleaned["concentration_percent"] = float(val.replace(",", ".").strip().rstrip("%"))
            except ValueError:
                cleaned.pop("concentration_percent", None)

    if tool_name in {"get_labelling_marketing_rules"}:
        val = cleaned.get("concentration_percent")
        if _is_empty(val) or isinstance(val, dict):
            cleaned.pop("concentration_percent", None)

    return cleaned
