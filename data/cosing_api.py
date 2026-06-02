"""Live client for the official EU CosIng API.

This is the agent's one **external API** dependency. It queries the European
Commission CosIng service at runtime (the public ``/annexes/{annex}/export-csv``
endpoint), parses the official annex export, and searches it for a single
ingredient.

Notes
-----
Self-contained: no dependency on any local SQLite pipeline, so the rest of
the data layer can be removed without touching this file.

Resilient for live demos: each annex export is cached on disk with a TTL.
A cached copy is also used as a fallback if the network call fails, so a
flaky connection during a presentation never breaks the agent.
"""

from __future__ import annotations

import csv
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from config.settings import COSING_API_BASE, PROJECT_ROOT
from data.annex_snapshots import sync_annex_snapshot_from_cache

# Annexes that carry a regulatory legal status, mapped to that status.
ANNEX_LEGAL_STATUS: dict[str, str] = {
    "II": "prohibited",
    "III": "restricted",
    "IV": "positive_list_colorant",
    "V": "positive_list_preservative",
    "VI": "positive_list_uv_filter",
}

CACHE_DIR = PROJECT_ROOT / "data" / ".cosing_cache"
CACHE_TTL_SECONDS = int(os.getenv("COSING_CACHE_TTL", str(24 * 60 * 60)))
REQUEST_TIMEOUT = int(os.getenv("COSING_API_TIMEOUT", "30"))

# Column-name variants seen across the annex exports.
_GLOSSARY_KEYS = (
    "Name of Common Ingredients Glossary",
    "Colour index Number / Name of Common Ingredients Glossary",
)
_SUBSTANCE_KEYS = ("Chemical name / INN", "Chemical name")
_IDENTIFIED_KEYS = ("Identified INGREDIENTS or substances e.g.",)


@dataclass
class AnnexMatch:
    """One regulatory annex row matched to an ingredient query.

    Attributes
    ----------
    annex : str
        Annex identifier (``II``–``VI``).
    entry_number : str
        Official reference number within the annex.
    legal_status : str
        Normalized status (e.g. ``prohibited``, ``restricted``).
    annex_reference : str
        Human-readable reference (e.g. ``Annex III, entry 45``).
    substance_name : str or None
        Chemical or INN name from the export row.
    inci_glossary : str or None
        INCI / Common Ingredients Glossary name from the row.
    regulation : str or None
        Citing regulation text when present in the export.
    product_type_body_parts : str or None
        Field of application (product type, body parts).
    max_concentration : str or None
        Maximum concentration text from the annex.
    other_conditions : str or None
        Other restriction conditions from the annex.
    wording_of_conditions : str or None
        Wording of conditions of use and warnings.
    match_type : str
        How the row matched: ``exact``, ``cas``, ``identified``, or ``fuzzy``.
    """

    annex: str
    entry_number: str
    legal_status: str
    annex_reference: str
    substance_name: str | None = None
    inci_glossary: str | None = None
    regulation: str | None = None
    product_type_body_parts: str | None = None
    max_concentration: str | None = None
    other_conditions: str | None = None
    wording_of_conditions: str | None = None
    match_type: str = "exact"  # exact | cas | identified | fuzzy


@dataclass
class IngredientSearchResult:
    """Aggregated search outcome across annex exports.

    Attributes
    ----------
    found : bool
        Whether at least one annex row matched the query.
    query : str
        Original INCI or substance name submitted.
    matched_name : str or None
        Best display name from the matched row(s).
    match_type : str or None
        Strongest match tier among hits (``exact``, ``cas``, etc.).
    matches : list[AnnexMatch]
        All annex rows matched for this query.
    """

    found: bool
    query: str
    matched_name: str | None = None
    match_type: str | None = None
    matches: list[AnnexMatch] = field(default_factory=list)


def _normalize(name: str) -> str:
    """Normalize an ingredient name for case-insensitive comparison.

    Parameters
    ----------
    name : str
        Raw INCI or substance name.

    Returns
    -------
    str
        Uppercased string with collapsed internal whitespace.
    """
    return " ".join((name or "").split()).upper()


def _split_multi(value: str | None) -> list[str]:
    """Split a multi-value annex cell into distinct names.

    Parameters
    ----------
    value : str or None
        Cell text (e.g. ``Retinol; Retinyl Acetate``).

    Returns
    -------
    list[str]
        Deduplicated name parts in encounter order.
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
    """Strip whitespace and map empty strings to ``None``.

    Parameters
    ----------
    value : str or None
        Raw cell value.

    Returns
    -------
    str or None
        Stripped text, or ``None`` if empty.
    """
    if value is None:
        return None
    text = value.strip()
    return text or None


def _first(record: dict[str, str], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among alternative column names.

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    keys : tuple[str, ...]
        Column header names to try in order.

    Returns
    -------
    str or None
        Cleaned cell value, or ``None`` if no key matches or all are empty.
    """
    for key in keys:
        if key in record:
            return _clean(record.get(key))
    return None


def _cache_path(annex: str) -> Path:
    """Return the on-disk cache path for one annex export.

    Parameters
    ----------
    annex : str
        Annex identifier (``II``–``VI``).

    Returns
    -------
    pathlib.Path
        Path to the cached CSV file under ``CACHE_DIR``.
    """
    return CACHE_DIR / f"annex_{annex}.csv"


def _is_fresh(path: Path) -> bool:
    """Check whether a cache file is younger than ``CACHE_TTL_SECONDS``.

    Parameters
    ----------
    path : pathlib.Path
        Cached annex CSV path.

    Returns
    -------
    bool
        ``True`` if the file exists and is within the TTL window.
    """
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def fetch_annex_csv(annex: str) -> str:
    """Fetch the raw CSV text for an annex from the live API (cached on disk).

    Falls back to a stale cache copy if the network call fails.

    Parameters
    ----------
    annex : str
        Annex identifier (``II``–``VI``).

    Returns
    -------
    str
        Full CSV export text for the annex.

    Raises
    ------
    RuntimeError
        If the API response is unexpectedly short or indicates missing credentials.
    Exception
        Re-raised when no cache exists and the download fails.
    """
    cache = _cache_path(annex)
    if _is_fresh(cache):
        return cache.read_text(encoding="utf-8", errors="replace")

    url = f"{COSING_API_BASE}/annexes/{annex}/export-csv"
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
            text: str = resp.read().decode("utf-8", errors="replace")
        if len(text) < 500 or "Missing Credentials" in text[:200]:
            raise RuntimeError(f"Unexpected CosIng response for annex {annex}.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
        sync_annex_snapshot_from_cache(annex, cache)
        return text
    except Exception:
        if cache.exists():  # stale-but-usable fallback keeps the agent live
            return cache.read_text(encoding="utf-8", errors="replace")
        raise


def _parse_annex(text: str, annex: str) -> list[dict[str, str]]:
    """Parse annex CSV text into a list of row dicts.

    Parameters
    ----------
    text : str
        Raw CSV export from CosIng.
    annex : str
        Annex identifier (used only for consistency with callers).

    Returns
    -------
    list[dict[str, str]]
        Rows keyed by column header, each with ``_reference_number`` set.
    """
    rows = list(csv.reader(StringIO(text)))
    try:
        header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Reference Number")
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


def _precise_match(
    record: dict[str, str], normalized_query: str, cas_number: str | None
) -> str | None:
    """Return a precise match type (exact name / CAS / identified), else ``None``.

    Precise matching only ever compares against whole, split-out names — never a
    substring of a chemical name — so e.g. "Glycerin" never matches the chemical
    "Nitroglycerin".

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    normalized_query : str
        Uppercased, whitespace-normalized query name.
    cas_number : str or None
        Optional CAS number for disambiguation.

    Returns
    -------
    str or None
        Match tier (``exact``, ``cas``, ``identified``) or ``None`` if no precise hit.
    """
    glossary_names = {_normalize(n) for n in _split_multi(_first(record, _GLOSSARY_KEYS))}
    if normalized_query in glossary_names:
        return "exact"

    if cas_number:
        cas_values = {c.strip() for c in _split_multi(record.get("CAS Number"))}
        if cas_number.strip() in cas_values:
            return "cas"

    identified = {_normalize(n) for n in _split_multi(_first(record, _IDENTIFIED_KEYS))}
    if normalized_query in identified:
        return "identified"
    return None


def _fuzzy_match(record: dict[str, str], normalized_query: str) -> bool:
    """Match query as a whole word within INCI glossary names (fallback tier).

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    normalized_query : str
        Uppercased, whitespace-normalized query name.

    Returns
    -------
    bool
        ``True`` if a word-boundary match is found and the query is at least four
        characters long.
    """
    if len(normalized_query) < 4:
        return False
    pattern = re.compile(rf"\b{re.escape(normalized_query)}\b")
    return any(pattern.search(_normalize(n)) for n in _split_multi(_first(record, _GLOSSARY_KEYS)))


def _record_to_match(record: dict[str, str], annex: str, match_type: str) -> AnnexMatch:
    """Map one parsed annex row to an ``AnnexMatch`` instance.

    Parameters
    ----------
    record : dict[str, str]
        Parsed annex CSV row.
    annex : str
        Annex identifier (``II``–``VI``).
    match_type : str
        How this row matched the query.

    Returns
    -------
    AnnexMatch
        Structured match with legal status and restriction fields populated.
    """
    entry = record["_reference_number"]
    return AnnexMatch(
        annex=annex,
        entry_number=entry,
        legal_status=ANNEX_LEGAL_STATUS.get(annex, "listed"),
        annex_reference=f"Annex {annex}, entry {entry}",
        substance_name=_first(record, _SUBSTANCE_KEYS),
        inci_glossary=_first(record, _GLOSSARY_KEYS),
        regulation=_clean(record.get("Regulation")),
        product_type_body_parts=_clean(record.get("Product Type, body parts")),
        max_concentration=_clean(record.get("Maximum concentration in ready for use preparation")),
        other_conditions=_clean(record.get("Other")),
        wording_of_conditions=_clean(record.get("Wording of conditions of use and warnings")),
        match_type=match_type,
    )


def search_ingredient(
    inci_name: str,
    cas_number: str | None = None,
    annexes: tuple[str, ...] = ("II", "III", "IV", "V", "VI"),
) -> IngredientSearchResult:
    """Search live CosIng annex exports for one ingredient across annexes.

    Parameters
    ----------
    inci_name : str
        INCI or substance name to look up.
    cas_number : str or None, optional
        Optional CAS number for precise disambiguation.
    annexes : tuple[str, ...], optional
        Annex identifiers to search (default: II through VI).

    Returns
    -------
    IngredientSearchResult
        Aggregated matches, preferring precise hits over fuzzy fallbacks.
    """
    normalized = _normalize(inci_name)
    if len(normalized) < 2:
        return IngredientSearchResult(found=False, query=inci_name)

    precise: list[AnnexMatch] = []
    fuzzy: list[AnnexMatch] = []
    precise_name: str | None = None
    fuzzy_name: str | None = None

    for annex in annexes:
        try:
            records = _parse_annex(fetch_annex_csv(annex), annex)
        except Exception:
            continue
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

    # Prefer precise matches; only fall back to fuzzy when nothing precise hit.
    matches = precise or fuzzy
    matched_name = precise_name if precise else fuzzy_name
    rank = {"exact": 0, "cas": 1, "identified": 2, "fuzzy": 3}
    best_type = min((m.match_type for m in matches), key=lambda t: rank[t], default=None)

    return IngredientSearchResult(
        found=bool(matches),
        query=inci_name,
        matched_name=matched_name,
        match_type=best_type,
        matches=matches,
    )
