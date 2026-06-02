"""Tool 2: Check concentration limits against Annex III/V/VI restrictions."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from data.cosing_api import search_ingredient
from data.product_types import (
    extract_first_percent,
    get_product_type,
    restriction_matches_product,
)
from tools.messages import annex_absence_message


@tool
def check_concentration_compliance(
    inci_name: str,
    product_category: str,
    concentration_percent: float | None = None,
) -> dict[str, Any]:
    """Check EU maximum allowed concentration for an ingredient in a product category.

    Use when the user asks about limits, maximum %, or whether a stated
    concentration complies.

    Parameters
    ----------
    inci_name : str
        INCI name of the ingredient.
    product_category : str
        Product type id (e.g. ``leave_on_face_cream``, ``rinse_off_shampoo``).
    concentration_percent : float or None, optional
        User-stated concentration to validate against annex limits.

    Returns
    -------
    dict[str, Any]
        Tool payload with applicable limits, per-row compliance flags, and an
        overall ``compliant`` value when a concentration was supplied.
    """
    result = search_ingredient(inci_name)
    if not result.found:
        return {"found": False, "inci_name": inci_name, "message": annex_absence_message(inci_name)}

    product = get_product_type(product_category)
    if not product:
        return {
            "found": False,
            "message": f"Unknown product category '{product_category}'.",
            "hint": "Use ids like leave_on_face_cream, rinse_off_shampoo, sunscreen.",
        }

    restrictions = [
        m
        for m in result.matches
        if m.max_concentration or m.product_type_body_parts or m.wording_of_conditions
    ]
    applicable = [
        m
        for m in restrictions
        if restriction_matches_product(m.product_type_body_parts, product.annex_category)
    ]
    if not applicable:
        applicable = restrictions

    if not applicable:
        return {
            "found": True,
            "inci_name": result.matched_name or inci_name,
            "product_category": product.id,
            "message": "No numeric restriction rows found for this ingredient in the annexes.",
            "compliant": None,
        }

    limits = []
    for match in applicable:
        max_pct = extract_first_percent(match.max_concentration)
        entry = {
            "annex_reference": match.annex_reference,
            "product_type_body_parts": match.product_type_body_parts,
            "max_concentration_text": match.max_concentration,
            "max_concentration_percent": max_pct,
            "other_conditions": match.other_conditions,
        }
        if concentration_percent is not None and max_pct is not None:
            entry["compliant"] = concentration_percent <= max_pct
        else:
            entry["compliant"] = None
        limits.append(entry)

    overall_compliant = None
    if concentration_percent is not None:
        checked = [e["compliant"] for e in limits if e["compliant"] is not None]
        if checked:
            overall_compliant = all(checked)

    return {
        "found": True,
        "inci_name": result.matched_name or inci_name,
        "product_category": product.id,
        "product_label": product.label,
        "user_concentration_percent": concentration_percent,
        "compliant": overall_compliant,
        "limits": limits,
        "note": "Complex multi-tier limits may require manual review of annex text.",
    }
