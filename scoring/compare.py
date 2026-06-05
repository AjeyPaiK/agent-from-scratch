"""Compare tool output to oracle expected output.

Weighted field-by-field comparison for lookup and concentration tool outputs.
"""

from __future__ import annotations

from typing import Any

MATCH_CONFIDENCE = {
    "exact": 1.0,
    "cas": 1.0,
    "identified": 0.85,
    "fuzzy": 0.7,
}


def _entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    """Build a stable key for one annex entry in lookup output.

    Parameters
    ----------
    entry : dict[str, Any]
        Single annex entry from lookup tool or oracle output.

    Returns
    -------
    tuple[str, str]
        ``(annex_reference, legal_status)`` or a fallback key derived from
        ``annex_reference`` and ``legal_status`` fields.
    """
    ref = entry.get("annex_reference") or ""
    parts = ref.replace("Annex ", "").split(", entry ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return ref, entry.get("legal_status") or ""


def _jaccard(left: set[Any], right: set[Any]) -> float:
    """Compute Jaccard similarity between two sets.

    Parameters
    ----------
    left : set
        First set of hashable elements.
    right : set
        Second set of hashable elements.

    Returns
    -------
    float
        Similarity in ``[0.0, 1.0]``. Both empty sets yield ``1.0``; one empty
        set yields ``0.0``.
    """
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _conditions_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    """Return whether condition lists match field-for-field.

    Parameters
    ----------
    actual : list[dict[str, Any]]
        Condition dicts from tool output.
    expected : list[dict[str, Any]]
        Condition dicts from oracle output.

    Returns
    -------
    bool
        ``True`` when lengths and compared string fields are equal after
        stripping whitespace.
    """
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected, strict=False):
        for key in (
            "applies_to",
            "max_concentration",
            "conditions_of_use_and_warnings",
            "other_conditions",
        ):
            if (left.get(key) or "").strip() != (right.get(key) or "").strip():
                return False
    return True


def _limit_key(entry: dict[str, Any]) -> tuple[str, str]:
    """Build a stable key for one concentration limit row.

    Parameters
    ----------
    entry : dict[str, Any]
        Single limit entry from concentration tool or oracle output.

    Returns
    -------
    tuple[str, str]
        ``(annex_reference, product_type_body_parts)`` after stripping.
    """
    return (
        (entry.get("annex_reference") or "").strip(),
        (entry.get("product_type_body_parts") or "").strip(),
    )


def _limit_fields_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return whether concentration limit fields match between actual and expected.

    Parameters
    ----------
    actual : dict[str, Any]
        Limit entry from tool output.
    expected : dict[str, Any]
        Limit entry from oracle output.

    Returns
    -------
    bool
        ``True`` when all compared limit fields are equal.
    """
    for key in (
        "max_concentration_text",
        "max_concentration_percent",
        "other_conditions",
        "compliant",
    ):
        if actual.get(key) != expected.get(key):
            return False
    return True


def compare_concentration_output(
    actual: dict[str, Any], expected: dict[str, Any]
) -> tuple[float, list[str]]:
    """Score ``check_concentration_compliance`` output against the CSV oracle.

    Parameters
    ----------
    actual : dict[str, Any]
        Tool output from ``check_concentration_compliance``.
    expected : dict[str, Any]
        Oracle expected output for the same arguments.

    Returns
    -------
    accuracy : float
        Weighted similarity score in ``[0.0, 1.0]``.
    mismatches : list[str]
        Names of fields or limit keys that differ.

    Notes
    -----
    Weights: ``found`` 0.15, ``product_category`` 0.10, ``limits`` 0.45,
    ``compliant`` 0.30. An ``error`` key in ``actual`` yields score ``0.0``.
    """
    mismatches: list[str] = []
    weights = {
        "found": 0.15,
        "product_category": 0.10,
        "limits": 0.45,
        "compliant": 0.30,
    }
    score = 0.0

    if actual.get("error"):
        return 0.0, ["error"]

    if bool(actual.get("found")) != bool(expected.get("found")):
        mismatches.append("found")
    else:
        score += weights["found"]

    if not expected.get("found"):
        return score, mismatches

    if (actual.get("product_category") or "").strip() != (
        expected.get("product_category") or ""
    ).strip():
        if expected.get("product_category") or actual.get("product_category"):
            mismatches.append("product_category")
    elif expected.get("product_category"):
        score += weights["product_category"]

    expected_limits = expected.get("limits") or []
    actual_limits = actual.get("limits") or []

    if not expected_limits and not actual_limits:
        limits_score = 1.0
        if (actual.get("message") or "").strip() != (expected.get("message") or "").strip():
            mismatches.append("message")
            limits_score = 0.5
    else:
        expected_keys = {_limit_key(entry) for entry in expected_limits}
        actual_keys = {_limit_key(entry) for entry in actual_limits}
        limits_score = _jaccard(actual_keys, expected_keys)
        if limits_score < 1.0:
            mismatches.append("limits")

        expected_by_key = {_limit_key(entry): entry for entry in expected_limits}
        field_matches = 0
        field_total = max(len(expected_by_key), 1)
        for key, expected_entry in expected_by_key.items():
            actual_entry = next(
                (entry for entry in actual_limits if _limit_key(entry) == key), None
            )
            if actual_entry is None:
                continue
            if _limit_fields_match(actual_entry, expected_entry):
                field_matches += 1
            else:
                mismatches.append(f"limit_fields:{key}")
        limits_score = (limits_score + (field_matches / field_total)) / 2

    score += weights["limits"] * limits_score

    expected_compliant = expected.get("compliant")
    if expected_compliant is not None:
        if actual.get("compliant") != expected_compliant:
            mismatches.append("compliant")
        else:
            score += weights["compliant"]
    elif actual.get("compliant") is None:
        score += weights["compliant"]

    return min(1.0, score), mismatches


def compare_lookup_output(
    actual: dict[str, Any], expected: dict[str, Any]
) -> tuple[float, list[str]]:
    """Score ``lookup_ingredient_regulation`` output against the CSV oracle.

    Parameters
    ----------
    actual : dict[str, Any]
        Tool output from ``lookup_ingredient_regulation``.
    expected : dict[str, Any]
        Oracle expected output for the same arguments.

    Returns
    -------
    accuracy : float
        Weighted similarity score in ``[0.0, 1.0]``, scaled by match-type
        confidence when applicable.
    mismatches : list[str]
        Names of fields or entry keys that differ.

    Notes
    -----
    Weights: ``found`` 0.20, ``overall_status`` 0.25, ``annex_entries`` 0.30,
    ``entry_fields`` 0.20, ``match_type`` 0.05. Match-type confidence is
    defined in ``MATCH_CONFIDENCE``.
    """
    mismatches: list[str] = []
    weights = {
        "found": 0.20,
        "overall_status": 0.25,
        "annex_entries": 0.30,
        "entry_fields": 0.20,
        "match_type": 0.05,
    }
    score = 0.0

    if actual.get("error"):
        return 0.0, ["error"]

    if bool(actual.get("found")) != bool(expected.get("found")):
        mismatches.append("found")
    else:
        score += weights["found"]

    if not expected.get("found"):
        return score, mismatches

    if actual.get("overall_status") != expected.get("overall_status"):
        mismatches.append("overall_status")
    else:
        score += weights["overall_status"]

    actual_entries = actual.get("annex_entries") or []
    expected_entries = expected.get("annex_entries") or []
    actual_keys = {_entry_key(entry) for entry in actual_entries}
    expected_keys = {_entry_key(entry) for entry in expected_entries}
    entry_set_score = _jaccard(actual_keys, expected_keys)
    score += weights["annex_entries"] * entry_set_score
    if entry_set_score < 1.0:
        mismatches.append("annex_entries")

    expected_by_key = {_entry_key(entry): entry for entry in expected_entries}
    field_matches = 0
    field_total = max(len(expected_by_key), 1)
    for key, expected_entry in expected_by_key.items():
        actual_entry = next((entry for entry in actual_entries if _entry_key(entry) == key), None)
        if actual_entry is None:
            continue
        entry_ok = True
        if actual_entry.get("legal_status") != expected_entry.get("legal_status"):
            mismatches.append(f"legal_status:{key}")
            entry_ok = False
        if (actual_entry.get("substance_name") or "").strip() != (
            expected_entry.get("substance_name") or ""
        ).strip():
            mismatches.append(f"substance_name:{key}")
            entry_ok = False
        if not _conditions_match(
            actual_entry.get("conditions") or [], expected_entry.get("conditions") or []
        ):
            mismatches.append(f"conditions:{key}")
            entry_ok = False
        if entry_ok:
            field_matches += 1
    score += weights["entry_fields"] * (field_matches / field_total)

    if actual.get("match_type") != expected.get("match_type"):
        mismatches.append("match_type")
    else:
        score += weights["match_type"]

    confidence = MATCH_CONFIDENCE.get(str(actual.get("match_type")), 1.0)
    return min(1.0, score * confidence), mismatches


def _normalized_strings(items: list[Any] | None) -> set[str]:
    """Return a set of stripped, lowercased strings from a list field."""
    return {str(item).strip().lower() for item in (items or []) if str(item).strip()}


def compare_labelling_output(
    actual: dict[str, Any], expected: dict[str, Any]
) -> tuple[float, list[str]]:
    """Score ``get_labelling_marketing_rules`` output against the CSV oracle.

    Parameters
    ----------
    actual : dict[str, Any]
        Tool output from ``get_labelling_marketing_rules``.
    expected : dict[str, Any]
        Oracle expected output for the same arguments.

    Returns
    -------
    accuracy : float
        Weighted similarity score in ``[0.0, 1.0]``.
    mismatches : list[str]
        Names of fields that differ.

    Notes
    -----
    Weights: ``found`` 0.15, ``inci_name`` 0.10, ``product_category`` 0.10,
    ``labelling_requirements`` 0.35, ``marketing_restrictions`` 0.20,
    ``references`` 0.10. List fields use Jaccard similarity on normalized
    strings. An ``error`` key in ``actual`` yields score ``0.0``.
    """
    mismatches: list[str] = []
    weights = {
        "found": 0.15,
        "inci_name": 0.10,
        "product_category": 0.10,
        "labelling_requirements": 0.35,
        "marketing_restrictions": 0.20,
        "references": 0.10,
    }
    score = 0.0

    if actual.get("error"):
        return 0.0, ["error"]

    if bool(actual.get("found")) != bool(expected.get("found")):
        mismatches.append("found")
    else:
        score += weights["found"]

    if not expected.get("found"):
        return score, mismatches

    if (actual.get("inci_name") or "").strip() != (expected.get("inci_name") or "").strip():
        mismatches.append("inci_name")
    else:
        score += weights["inci_name"]

    if (actual.get("product_category") or "").strip() != (
        expected.get("product_category") or ""
    ).strip():
        if expected.get("product_category") or actual.get("product_category"):
            mismatches.append("product_category")
    elif expected.get("product_category"):
        score += weights["product_category"]

    for field in ("labelling_requirements", "marketing_restrictions", "references"):
        actual_set = _normalized_strings(actual.get(field))
        expected_set = _normalized_strings(expected.get(field))
        field_score = _jaccard(actual_set, expected_set)
        score += weights[field] * field_score
        if field_score < 1.0:
            mismatches.append(field)

    return min(1.0, score), mismatches
