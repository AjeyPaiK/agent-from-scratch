"""Tool 1: Look up ingredient regulatory status via the live EU CosIng API."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.tools import tool

from data.cosing_api import AnnexMatch, search_ingredient
from data.legal_status import LEGAL_SUMMARY
from tools.messages import annex_absence_message


def _condition_payload(match: AnnexMatch) -> dict[str, Any]:
    """Build a dict of populated restriction-condition fields for one annex match.

    Parameters
    ----------
    match : AnnexMatch
        Parsed annex row for the ingredient.

    Returns
    -------
    dict[str, Any]
        Subset of condition keys with non-empty, human-meaningful values.
    """
    payload = {
        "applies_to": match.product_type_body_parts,
        "max_concentration": match.max_concentration,
        "conditions_of_use_and_warnings": match.wording_of_conditions,
        "other_conditions": match.other_conditions,
    }
    return {k: v for k, v in payload.items() if v and str(v).strip()}


@tool
def lookup_ingredient_regulation(inci_name: str, cas_number: str | None = None) -> dict[str, Any]:
    """Look up an INCI ingredient against EU Annex II–VI rules via the live CosIng API.

    Use first when the user asks whether an ingredient is allowed, prohibited, or
    restricted.

    Parameters
    ----------
    inci_name : str
        INCI name (e.g. Phenoxyethanol, Retinol).
    cas_number : str or None, optional
        CAS number for disambiguation when multiple substances share a name.

    Returns
    -------
    dict[str, Any]
        Tool payload with ``found``, status summaries, annex entries, and source
        metadata; or a not-found message when the ingredient is absent from annexes.
    """
    result = search_ingredient(inci_name, cas_number)
    if not result.found:
        return {
            "found": False,
            "inci_name": inci_name,
            "message": annex_absence_message(inci_name),
        }

    conditions_by_entry: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in result.matches:
        payload = _condition_payload(match)
        if payload:
            conditions_by_entry[(match.annex, match.entry_number)].append(payload)

    seen: set[tuple[str, str]] = set()
    statuses: list[dict[str, Any]] = []
    for match in result.matches:
        key = (match.annex, match.entry_number)
        if key in seen:
            continue
        seen.add(key)
        entry: dict[str, Any] = {
            "legal_status": match.legal_status,
            "summary": LEGAL_SUMMARY.get(match.legal_status, match.legal_status),
            "annex_reference": match.annex_reference,
            "substance_name": match.substance_name,
            "regulation": match.regulation,
        }
        conditions = conditions_by_entry.get(key)
        if conditions:
            entry["conditions"] = conditions
        statuses.append(entry)

    if any(m.legal_status == "prohibited" for m in result.matches):
        overall = "prohibited"
    elif any(m.legal_status == "restricted" for m in result.matches):
        overall = "restricted"
    else:
        overall = result.matches[0].legal_status

    return {
        "found": True,
        "inci_name": result.matched_name or inci_name,
        "match_type": result.match_type,
        "overall_status": overall,
        "annex_entries": statuses,
        "source": "Live EU CosIng API (Regulation EC 1223/2009, annexes II–VI)",
    }
