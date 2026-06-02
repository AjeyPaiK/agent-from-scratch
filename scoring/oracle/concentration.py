"""Oracle for ``check_concentration_compliance`` — built from pinned annex CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.product_types import extract_first_percent, get_product_type, restriction_matches_product
from scoring.oracle.csv_reader import OracleRowMatch, search_annex_rows


def _is_restriction_row(match: OracleRowMatch) -> bool:
    """Return whether an annex row carries concentration restriction data.

    Parameters
    ----------
    match : OracleRowMatch
        Matched annex CSV row.

    Returns
    -------
    bool
        ``True`` when max concentration, product type, or wording fields are
        present.
    """
    return bool(
        match.max_concentration or match.product_type_body_parts or match.wording_of_conditions
    )


def oracle_check_concentration_compliance(
    inci_name: str,
    product_category: str,
    concentration_percent: float | None = None,
    *,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Expected output for ``check_concentration_compliance`` from pinned CSV data.

    Parameters
    ----------
    inci_name : str
        INCI or glossary name to evaluate.
    product_category : str
        Product category id (for example ``leave_on_face_cream``).
    concentration_percent : float or None, optional
        User-supplied concentration to test against limits, by default ``None``.
    snapshot_dir : Path or None, optional
        Directory containing pinned annex CSV snapshots, by default ``None``.

    Returns
    -------
    dict[str, Any]
        Oracle output mirroring the concentration tool schema: ``found``,
        ``inci_name``, ``product_category``, optional ``limits``, ``compliant``,
        and error or hint messages when lookup fails.

    Notes
    -----
    When no product-specific restriction rows apply, all restriction rows are
    considered. ``compliant`` is ``None`` when concentration or numeric limits
    are unavailable.
    """
    result = search_annex_rows(inci_name, snapshot_dir=snapshot_dir)
    if not result.found:
        return {"found": False, "inci_name": inci_name, "message": "Ingredient not found."}

    product = get_product_type(product_category)
    if not product:
        return {
            "found": False,
            "message": f"Unknown product category '{product_category}'.",
            "hint": "Use ids like leave_on_face_cream, rinse_off_shampoo, sunscreen.",
        }

    restrictions = [match for match in result.matches if _is_restriction_row(match)]
    applicable = [
        match
        for match in restrictions
        if restriction_matches_product(match.product_type_body_parts, product.annex_category)
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

    limits: list[dict[str, Any]] = []
    for match in applicable:
        max_pct = extract_first_percent(match.max_concentration)
        entry: dict[str, Any] = {
            "annex_reference": f"Annex {match.annex}, entry {match.entry_number}",
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
        checked = [entry["compliant"] for entry in limits if entry["compliant"] is not None]
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
    }
