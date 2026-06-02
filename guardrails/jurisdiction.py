"""Detect whether a query is scoped to EU jurisdiction only.

Pattern-based checks for non-EU regulators, regions, adjective scopes,
and geographic phrases. EU member states and cosmetic product contexts
are allowlisted.

Notes
-----
Used by the pre-input guardrail to block non-EU jurisdiction queries
before the LLM runs.
"""

from __future__ import annotations

import re

# Explicit EU scope — user clearly asks about EU rules.
EU_SCOPE_MARKERS = re.compile(
    r"\b("
    r"EU|E\.U\.|European Union|EEA|"
    r"Regulation \(EC\) No 1223|1223/2009|"
    r"CosIng|Annex (?:II|III|IV|V|VI)"
    r")\b",
    re.I,
)

# Non-EU regulatory bodies → always out of scope.
NON_EU_REGULATORS = re.compile(
    r"\b(FDA|CFDA|NMPA|MHLW|PMDA|TGA|Health Canada)\b",
    re.I,
)

# Non-EU regions (broader than country lists).
NON_EU_REGIONS = re.compile(
    r"\b("
    r"Asia|Africa|Latin America|South America|North America|"
    r"Middle East|APAC|Australasia|Sub-Saharan Africa"
    r")\b",
    re.I,
)

# Adjective + market/regulation phrasing, e.g. "American market", "Indian regulations".
NON_EU_ADJECTIVE_SCOPE = re.compile(
    r"\b("
    r"American|US|U\.S\.|Indian|Chinese|Japanese|British|Canadian|Australian|"
    r"Brazilian|Mexican|Korean|Russian|Turkish|Thai|Singaporean|Malaysian|"
    r"Indonesian|Nigerian|South African|Saudi|Emirati|Vietnamese|Filipino|"
    r"Pakistani|Bangladeshi|Taiwanese|Hong Kong"
    r")\s+(market|markets|regulations?|rules?|law|jurisdiction|cosmetics?)\b",
    re.I,
)

# "in India", "in leave-on face cream", etc. — greedy place capture so hyphens stay inside product names.
JURISDICTION_IN_PHRASE = re.compile(
    r"\b(?:in|across|within|under|sold in|market(?:s)? in|regulated in|approved in)\s+"
    r"(?:the\s+)?(?P<place>[A-Za-z][A-Za-z\s\-']{0,80})"
    r"(?:\s+(?:market|markets|jurisdiction|regulations?|rules?|law|cosmetics?))?"
    r"(?=\s|$|[.,?!])",
    re.I,
)

JURISDICTION_FOR_MARKET = re.compile(
    r"\bfor\s+(?:the\s+)?(?P<place>[A-Za-z][A-Za-z\s\-']{0,48}?)\s+"
    r"(?:market|markets|jurisdiction|regulations?|rules?|law|cosmetics?)\b",
    re.I,
)

# Common abbreviations / aliases mapped to canonical non-EU labels.
NON_EU_ALIASES: dict[str, str] = {
    "us": "United States",
    "u.s.": "United States",
    "u.s": "United States",
    "usa": "United States",
    "united states": "United States",
    "america": "United States",
    "american": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "india": "India",
    "indian": "India",
    "china": "China",
    "chinese": "China",
    "japan": "Japan",
    "japanese": "Japan",
    "canada": "Canada",
    "canadian": "Canada",
    "australia": "Australia",
    "australian": "Australia",
    "brazil": "Brazil",
    "mexico": "Mexico",
    "korea": "South Korea",
    "south korea": "South Korea",
    "singapore": "Singapore",
    "switzerland": "Switzerland",
    "norway": "Norway",
    "turkey": "Turkey",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
}

# EU member states + explicit EU geographic aliases (lowercase keys).
EU_JURISDICTIONS: set[str] = {
    "eu",
    "e.u.",
    "european union",
    "eea",
    "europe",
    "european",
    "eurozone",
    "austria",
    "belgium",
    "bulgaria",
    "croatia",
    "cyprus",
    "czech republic",
    "czechia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "greece",
    "hungary",
    "ireland",
    "italy",
    "latvia",
    "lithuania",
    "luxembourg",
    "malta",
    "netherlands",
    "the netherlands",
    "poland",
    "portugal",
    "romania",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
}

# Cosmetic product contexts — "in hand lotion" is not a country reference.
PRODUCT_CONTEXT_TERMS: tuple[str, ...] = (
    "leave",
    "leave-on",
    "leave on",
    "rinse-off",
    "rinse off",
    "hand lotion",
    "body lotion",
    "face cream",
    "face serum",
    "eye cream",
    "lip balm",
    "shampoo",
    "cleanser",
    "sunscreen",
    "deodorant",
    "hair dye",
    "oral care",
    "cosmetic",
    "cosmetics",
    "product",
    "products",
    "formulation",
    "cream",
    "lotion",
    "serum",
    "gel",
    "balm",
    "makeup",
    "lipstick",
    "mascara",
    "foundation",
    "toner",
    "moisturizer",
    "moisturiser",
    "leave_on_face_cream",
    "leave_on_body_lotion",
    "rinse_off_shampoo",
    "rinse_off_cleanser",
)

BLOCK_MESSAGE = (
    "This agent covers EU Regulation (EC) No 1223/2009 only. "
    "It cannot answer questions about {place} or other non-EU markets. "
    "Rephrase your question for the EU (e.g. “Is retinol allowed in a leave-on body lotion in the EU?”)."
)


def _normalize_place(raw: str) -> str:
    """Normalize a captured geographic phrase for lookup.

    Parameters
    ----------
    raw : str
        Raw place string from a regex capture group.

    Returns
    -------
    str
        Lowercased, trimmed place with leading articles and trailing
        punctuation removed.
    """
    place = raw.strip().lower()
    place = re.sub(r"^(a|an|the)\s+", "", place)
    return place.strip(" .,?!\"'")


def _is_product_context(place: str) -> bool:
    """Return whether a place string refers to a product, not geography.

    Parameters
    ----------
    place : str
        Normalized place string.

    Returns
    -------
    bool
        ``True`` when ``place`` matches cosmetic product terms or contains
        numeric concentration context; ``False`` otherwise.
    """
    if not place:
        return True
    if any(term in place for term in PRODUCT_CONTEXT_TERMS):
        return True
    # Percent / concentration context: "in 0.8% serum"
    if re.search(r"\d", place):
        return True
    return False


def _is_eu_jurisdiction(place: str) -> bool:
    """Return whether a place string refers to the EU or a member state.

    Parameters
    ----------
    place : str
        Normalized place string.

    Returns
    -------
    bool
        ``True`` when ``place`` is an EU jurisdiction alias or contains
        one; ``False`` otherwise.
    """
    if place in EU_JURISDICTIONS:
        return True
    # "german market" → germany
    for eu in EU_JURISDICTIONS:
        if eu in place and len(eu) > 3:
            return True
    return False


def _resolve_non_eu(place: str, *, allow_unknown: bool = True) -> str | None:
    """Map a normalized place to a display name when it is non-EU.

    Parameters
    ----------
    place : str
        Normalized place string.
    allow_unknown : bool, optional
        When ``True``, treat unknown lowercase geography strings as non-EU
        (EU member states are allowlisted separately). Default is ``True``.

    Returns
    -------
    str or None
        Canonical display name for a non-EU place; ``None`` when ``place`` is
        empty, a product context, EU jurisdiction, or unrecognized.
    """
    if not place or _is_product_context(place):
        return None
    if _is_eu_jurisdiction(place):
        return None

    if place in NON_EU_ALIASES:
        return NON_EU_ALIASES[place]

    # Unknown geography after "in …" defaults to non-EU (EU member states are allowlisted).
    if allow_unknown and re.fullmatch(r"[a-z][a-z\s\-']*", place) and len(place) >= 3:
        return place.title()

    return None


def detect_non_eu_jurisdiction(text: str) -> str | None:
    """Detect non-EU jurisdiction references in user text.

    Parameters
    ----------
    text : str
        User message to scan for regulators, regions, adjective scopes,
        and geographic phrases.

    Returns
    -------
    str or None
        Human-readable non-EU place or regime label when detected;
        ``None`` when no non-EU jurisdiction is found.
    """
    if NON_EU_REGULATORS.search(text):
        return "non-EU regulatory regimes"

    if NON_EU_REGIONS.search(text):
        match = NON_EU_REGIONS.search(text)
        return match.group(0) if match else "non-EU regions"

    adj = NON_EU_ADJECTIVE_SCOPE.search(text)
    if adj:
        return adj.group(1)

    for match in JURISDICTION_IN_PHRASE.finditer(text):
        place = _normalize_place(match.group("place"))
        resolved = _resolve_non_eu(place, allow_unknown=False)
        if resolved:
            return resolved

    for match in JURISDICTION_FOR_MARKET.finditer(text):
        place = _normalize_place(match.group("place"))
        resolved = _resolve_non_eu(place, allow_unknown=False)
        if resolved:
            return resolved

    for alias, label in NON_EU_ALIASES.items():
        if re.search(rf"\bfor\s+(?:the\s+)?{re.escape(alias)}\b", text, re.I):
            return label
        if len(alias) >= 3 and re.search(
            rf"\b{re.escape(alias)}\s+(?:market|markets|jurisdiction|regulations?|rules?|law)\b",
            text,
            re.I,
        ):
            return label
        if re.search(rf"\b(?:in|under)\s+(?:the\s+)?{re.escape(alias)}\b", text, re.I):
            return label

    return None
