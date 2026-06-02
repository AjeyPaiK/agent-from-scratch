"""Tests for LLM tool argument sanitization.

Notes
-----
Validates ``sanitize_tool_kwargs`` cleanup of optional ``cas_number`` fields
passed to ``lookup_ingredient_regulation``.
"""

from agent.tool_args import sanitize_tool_kwargs


def test_cas_number_empty_dict_removed():
    """Remove an empty dict ``cas_number`` from lookup tool arguments.

    Notes
    -----
    Only ``inci_name`` should remain when ``cas_number`` is an empty mapping.
    """
    cleaned = sanitize_tool_kwargs(
        "lookup_ingredient_regulation",
        {"inci_name": "Retinol", "cas_number": {}},
    )
    assert cleaned == {"inci_name": "Retinol"}


def test_cas_number_empty_string_removed():
    """Remove a blank string ``cas_number`` from lookup tool arguments.

    Notes
    -----
    Empty-string CAS values should be dropped entirely from sanitized kwargs.
    """
    cleaned = sanitize_tool_kwargs(
        "lookup_ingredient_regulation",
        {"inci_name": "Retinol", "cas_number": ""},
    )
    assert "cas_number" not in cleaned


def test_cas_number_valid_string_kept():
    """Preserve a valid CAS number string in sanitized lookup arguments.

    Notes
    -----
    Well-formed CAS strings such as ``68-26-8`` should pass through unchanged.
    """
    cleaned = sanitize_tool_kwargs(
        "lookup_ingredient_regulation",
        {"inci_name": "Retinol", "cas_number": "68-26-8"},
    )
    assert cleaned["cas_number"] == "68-26-8"
