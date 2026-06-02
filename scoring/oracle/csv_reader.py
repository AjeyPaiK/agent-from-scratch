"""Read pinned annex CSV snapshots for oracle scoring.

Independent of ``data.cosing_api`` — same source files, separate parser and matcher.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from config.settings import ANNEX_SNAPSHOT_DIR

ANNEX_LEGAL_STATUS: dict[str, str] = {
    "II": "prohibited",
    "III": "restricted",
    "IV": "positive_list_colorant",
    "V": "positive_list_preservative",
    "VI": "positive_list_uv_filter",
}

_GLOSSARY_KEYS = (
    "Name of Common Ingredients Glossary",
    "Colour index Number / Name of Common Ingredients Glossary",
)
_SUBSTANCE_KEYS = ("Chemical name / INN", "Chemical name")
_IDENTIFIED_KEYS = ("Identified INGREDIENTS or substances e.g.",)


@dataclass
class OracleRowMatch:
    """One annex CSV row matched to an ingredient query.

    Attributes
    ----------
    annex : str
        Annex identifier (for example ``"III"``).
    entry_number : str
        Reference number within the annex.
    legal_status : str
        Derived legal status for the annex (for example ``"restricted"``).
    substance_name : str or None
        Chemical or INN name from the row.
    regulation : str or None
        Regulation citation from the row.
    product_type_body_parts : str or None
        Product type and body-part restriction text.
    max_concentration : str or None
        Maximum concentration text from the row.
    other_conditions : str or None
        Other restriction conditions.
    wording_of_conditions : str or None
        Wording of conditions of use and warnings.
    match_type : str
        Match quality: ``exact``, ``cas``, ``identified``, or ``fuzzy``.
    """

    annex: str
    entry_number: str
    legal_status: str
    substance_name: str | None
    regulation: str | None
    product_type_body_parts: str | None
    max_concentration: str | None
    other_conditions: str | None
    wording_of_conditions: str | None
    match_type: str


@dataclass
class OracleSearchResult:
    """Aggregated result of searching annex snapshots for one ingredient.

    Attributes
    ----------
    found : bool
        Whether at least one annex row matched.
    matched_name : str or None
        Glossary or substance name from the best match.
    match_type : str or None
        Best match type among all matches.
    matches : list[OracleRowMatch]
        All matching annex rows.
    """

    found: bool
    matched_name: str | None
    match_type: str | None
    matches: list[OracleRowMatch]


def _normalize(name: str) -> str:
    """Normalize an ingredient name for case-insensitive comparison.

    Parameters
    ----------
    name : str
        Raw ingredient or glossary name.

    Returns
    -------
    str
        Uppercase string with collapsed internal whitespace.
    """
    return " ".join((name or "").split()).upper()


def _split_multi(value: str | None) -> list[str]:
    """Split a CSV cell containing multiple delimited names.

    Parameters
    ----------
    value : str or None
        Cell text that may contain ``/``, ``;``, or ``,`` separators.

    Returns
    -------
    list[str]
        Unique trimmed name parts in first-seen order.
    """
    if not value:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\s*/\s*|\s*;\s*|\s*,\s*", value):
        name = part.strip()
        key = name.upper()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _clean(value: str | None) -> str | None:
    """Strip whitespace and convert blank strings to ``None``.

    Parameters
    ----------
    value : str or None
        Raw cell or field value.

    Returns
    -------
    str or None
        Stripped non-empty string, or ``None`` when blank.
    """
    if value is None:
        return None
    text = value.strip()
    return text or None


def _first(record: dict[str, str], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among candidate CSV column keys.

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    keys : tuple[str, ...]
        Column header names to try in order.

    Returns
    -------
    str or None
        Cleaned value from the first matching key, or ``None``.
    """
    for key in keys:
        if key in record:
            return _clean(record.get(key))
    return None


def _parse_annex_csv(text: str, annex: str) -> list[dict[str, str]]:
    """Parse annex CSV text into row dicts keyed by column headers.

    Parameters
    ----------
    text : str
        Full CSV file contents.
    annex : str
        Annex identifier (used for context; not stored on rows).

    Returns
    -------
    list[dict[str, str]]
        Data rows with header keys plus ``_reference_number``.

    Notes
    -----
    Returns an empty list when no ``Reference Number`` header row is found.
    """
    rows = list(csv.reader(StringIO(text)))
    try:
        header_idx = next(i for i, row in enumerate(rows) if row and row[0] == "Reference Number")
    except StopIteration:
        return []
    headers = rows[header_idx]

    records: list[dict[str, str]] = []
    for row in rows[header_idx + 1 :]:
        if not row or not row[0].strip() or not row[0].strip()[0].isdigit():
            continue
        record = {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
        record["_reference_number"] = row[0].strip()
        records.append(record)
    return records


def load_annex_records(annex: str, snapshot_dir: Path | None = None) -> list[dict[str, str]]:
    """Load one annex from a pinned on-disk snapshot.

    Parameters
    ----------
    annex : str
        Annex identifier (for example ``"III"``).
    snapshot_dir : Path or None, optional
        Snapshot root directory. When ``None``, uses ``ANNEX_SNAPSHOT_DIR``,
        by default ``None``.

    Returns
    -------
    list[dict[str, str]]
        Parsed rows from ``annex_{annex}.csv``.

    Notes
    -----
    Raises ``FileNotFoundError`` when the snapshot file is missing.
    """
    root = snapshot_dir or ANNEX_SNAPSHOT_DIR
    path = root / f"annex_{annex}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing annex snapshot: {path}")
    return _parse_annex_csv(path.read_text(encoding="utf-8", errors="replace"), annex)


def _precise_match(
    record: dict[str, str], normalized_query: str, cas_number: str | None
) -> str | None:
    """Return precise match type for one record, or ``None`` if no match.

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    normalized_query : str
        Uppercase normalized INCI or glossary query.
    cas_number : str or None
        Optional CAS number for CAS-based matching.

    Returns
    -------
    str or None
        One of ``"exact"``, ``"cas"``, ``"identified"``, or ``None``.
    """
    glossary_names = {_normalize(name) for name in _split_multi(_first(record, _GLOSSARY_KEYS))}
    if normalized_query in glossary_names:
        return "exact"

    if cas_number:
        cas_values = {value.strip() for value in _split_multi(record.get("CAS Number"))}
        if cas_number.strip() in cas_values:
            return "cas"

    identified = {_normalize(name) for name in _split_multi(_first(record, _IDENTIFIED_KEYS))}
    if normalized_query in identified:
        return "identified"
    return None


def _fuzzy_match(record: dict[str, str], normalized_query: str) -> bool:
    """Return whether the query appears as a whole word in glossary names.

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    normalized_query : str
        Uppercase normalized query; queries shorter than four characters never match.

    Returns
    -------
    bool
        ``True`` when a word-boundary match is found in glossary names.
    """
    if len(normalized_query) < 4:
        return False
    pattern = re.compile(rf"\b{re.escape(normalized_query)}\b")
    return any(
        pattern.search(_normalize(name)) for name in _split_multi(_first(record, _GLOSSARY_KEYS))
    )


def _record_to_match(record: dict[str, str], annex: str, match_type: str) -> OracleRowMatch:
    """Convert a parsed CSV record into an ``OracleRowMatch``.

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    annex : str
        Annex identifier.
    match_type : str
        Match quality label for this row.

    Returns
    -------
    OracleRowMatch
        Structured match object for oracle scoring.
    """
    entry = record["_reference_number"]
    return OracleRowMatch(
        annex=annex,
        entry_number=entry,
        legal_status=ANNEX_LEGAL_STATUS.get(annex, "listed"),
        substance_name=_first(record, _SUBSTANCE_KEYS),
        regulation=_clean(record.get("Regulation")),
        product_type_body_parts=_clean(record.get("Product Type, body parts")),
        max_concentration=_clean(record.get("Maximum concentration in ready for use preparation")),
        other_conditions=_clean(record.get("Other")),
        wording_of_conditions=_clean(record.get("Wording of conditions of use and warnings")),
        match_type=match_type,
    )


def search_annex_rows(
    inci_name: str,
    cas_number: str | None = None,
    annexes: tuple[str, ...] = ("II", "III", "IV", "V", "VI"),
    snapshot_dir: Path | None = None,
) -> OracleSearchResult:
    """Find annex rows for one ingredient using the oracle CSV matcher.

    Parameters
    ----------
    inci_name : str
        INCI or glossary name to search for.
    cas_number : str or None, optional
        Optional CAS number for precise matching, by default ``None``.
    annexes : tuple[str, ...], optional
        Annex ids to search, by default ``("II", "III", "IV", "V", "VI")``.
    snapshot_dir : Path or None, optional
        Snapshot root directory, by default ``None``.

    Returns
    -------
    OracleSearchResult
        Search outcome with matched rows and best match type.

    Notes
    -----
    Precise matches (exact, cas, identified) take precedence over fuzzy matches.
    Queries shorter than two normalized characters return ``found=False``.
    """
    normalized = _normalize(inci_name)
    if len(normalized) < 2:
        return OracleSearchResult(found=False, matched_name=None, match_type=None, matches=[])

    precise: list[OracleRowMatch] = []
    fuzzy: list[OracleRowMatch] = []
    precise_name: str | None = None
    fuzzy_name: str | None = None

    for annex in annexes:
        records = load_annex_records(annex, snapshot_dir)
        for record in records:
            match_type = _precise_match(record, normalized, cas_number)
            if match_type:
                precise.append(_record_to_match(record, annex, match_type))
                if precise_name is None:
                    precise_name = _first(record, _GLOSSARY_KEYS) or _first(record, _SUBSTANCE_KEYS)
            elif _fuzzy_match(record, normalized):
                fuzzy.append(_record_to_match(record, annex, "fuzzy"))
                if fuzzy_name is None:
                    fuzzy_name = _first(record, _GLOSSARY_KEYS)

    matches = precise or fuzzy
    matched_name = precise_name if precise else fuzzy_name
    rank = {"exact": 0, "cas": 1, "identified": 2, "fuzzy": 3}
    best_type = min((match.match_type for match in matches), key=lambda t: rank[t], default=None)

    return OracleSearchResult(
        found=bool(matches),
        matched_name=matched_name,
        match_type=best_type,
        matches=matches,
    )
