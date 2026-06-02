"""Oracle for ``lookup_ingredient_regulation`` — built from pinned annex CSVs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from data.legal_status import LEGAL_SUMMARY
from scoring.oracle.csv_reader import OracleRowMatch, search_annex_rows
from tools.messages import annex_absence_message


def _condition_payload(match: OracleRowMatch) -> dict[str, Any]:
    """Build a non-empty condition dict from one annex row match.

    Parameters
    ----------
    match : OracleRowMatch
        Matched annex CSV row.

    Returns
    -------
    dict[str, Any]
        Condition fields with non-blank values only.
    """
    payload = {
        "applies_to": match.product_type_body_parts,
        "max_concentration": match.max_concentration,
        "conditions_of_use_and_warnings": match.wording_of_conditions,
        "other_conditions": match.other_conditions,
    }
    return {key: value for key, value in payload.items() if value and str(value).strip()}


def oracle_lookup_ingredient_regulation(
    inci_name: str,
    cas_number: str | None = None,
    *,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Expected output for ``lookup_ingredient_regulation`` from pinned CSV data.

    Parameters
    ----------
    inci_name : str
        INCI or glossary name to look up.
    cas_number : str or None, optional
        Optional CAS number for disambiguation, by default ``None``.
    snapshot_dir : Path or None, optional
        Directory containing pinned annex CSV snapshots, by default ``None``.

    Returns
    -------
    dict[str, Any]
        Oracle output mirroring the lookup tool schema: ``found``, ``inci_name``,
        optional ``message``, ``match_type``, ``overall_status``, and
        ``annex_entries`` when matches exist.

    Notes
    -----
    ``overall_status`` is ``prohibited`` if any match is prohibited,
    ``restricted`` if any match is restricted, otherwise the first match's
    legal status.
    """
    result = search_annex_rows(inci_name, cas_number, snapshot_dir=snapshot_dir)
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
            "annex_reference": f"Annex {match.annex}, entry {match.entry_number}",
            "substance_name": match.substance_name,
            "regulation": match.regulation,
        }
        conditions = conditions_by_entry.get(key)
        if conditions:
            entry["conditions"] = conditions
        statuses.append(entry)

    if any(match.legal_status == "prohibited" for match in result.matches):
        overall = "prohibited"
    elif any(match.legal_status == "restricted" for match in result.matches):
        overall = "restricted"
    else:
        overall = result.matches[0].legal_status

    return {
        "found": True,
        "inci_name": result.matched_name or inci_name,
        "match_type": result.match_type,
        "overall_status": overall,
        "annex_entries": statuses,
    }
