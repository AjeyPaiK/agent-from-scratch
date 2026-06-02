"""Static product-type vocabulary and restriction-matching helpers.

Replaces the SQLite-seeded ``product_types`` table and the small pure helpers
that lived in ``data/eu/queries.py``, so the agent tools depend only on the
live CosIng API client and this static config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProductType:
    """Agent-facing product category for annex restriction matching.

    Attributes
    ----------
    id : str
        Stable identifier passed to tools (e.g. ``leave_on_face_cream``).
    label : str
        Human-readable product description.
    annex_category : str
        Broad category for matching annex fields: ``leave_on``, ``rinse_off``,
        ``hair_dye``, or ``oral``.
    requires_annex : str or None
        When set, indicates a product must comply with a specific annex (e.g. ``VI`` for sunscreen).
    """

    id: str
    label: str
    annex_category: str  # leave_on | rinse_off | hair_dye | oral
    requires_annex: str | None = None


PRODUCT_TYPES: dict[str, ProductType] = {
    "leave_on_face_cream": ProductType("leave_on_face_cream", "Leave-on face cream", "leave_on"),
    "leave_on_body_lotion": ProductType("leave_on_body_lotion", "Leave-on body lotion", "leave_on"),
    "rinse_off_shampoo": ProductType("rinse_off_shampoo", "Rinse-off shampoo", "rinse_off"),
    "rinse_off_cleanser": ProductType(
        "rinse_off_cleanser", "Rinse-off facial cleanser", "rinse_off"
    ),
    "sunscreen": ProductType(
        "sunscreen", "Sunscreen / sun protection product", "leave_on", requires_annex="VI"
    ),
    "hair_dye": ProductType("hair_dye", "Hair dye product", "hair_dye"),
    "oral_care": ProductType("oral_care", "Oral care product", "oral"),
    "deodorant": ProductType("deodorant", "Deodorant / antiperspirant", "leave_on"),
}

# Field-of-application vocabulary used to match annex restriction rows to a
# product category (e.g. an Annex III "rinse-off" condition vs a leave-on query).
CATEGORY_HINTS: dict[str, list[str]] = {
    "leave_on": ["leave-on", "leave on", "body and hand", "face"],
    "rinse_off": ["rinse-off", "rinse off"],
    "oral": ["oral"],
    "hair_dye": ["hair dye", "hair"],
}


def get_product_type(product_type_id: str | None) -> ProductType | None:
    """Resolve a product type by id or case-insensitive label.

    Parameters
    ----------
    product_type_id : str or None
        Tool ``product_category`` id or display label.

    Returns
    -------
    ProductType or None
        Matching ``ProductType``, or ``None`` if unknown or empty input.
    """
    if not product_type_id:
        return None
    key = product_type_id.strip()
    if key in PRODUCT_TYPES:
        return PRODUCT_TYPES[key]
    for product in PRODUCT_TYPES.values():
        if product.label.lower() == key.lower():
            return product
    return None


def restriction_matches_product(product_type_body_parts: str | None, annex_category: str) -> bool:
    """Return whether an annex restriction's field-of-application covers this category.

    Parameters
    ----------
    product_type_body_parts : str or None
        Annex ``Product Type, body parts`` cell text.
    annex_category : str
        Product's broad category (``leave_on``, ``rinse_off``, etc.).

    Returns
    -------
    bool
        ``True`` if the restriction applies to this product category.
    """
    text = (product_type_body_parts or "").lower()
    if not text or "all cosmetic" in text:
        return True
    hints = CATEGORY_HINTS.get(annex_category, [annex_category.replace("_", " ")])
    return any(hint in text for hint in hints)


def extract_first_percent(value: str | None) -> float | None:
    """Extract the first percentage value from a free-text concentration limit.

    Parameters
    ----------
    value : str or None
        Annex maximum-concentration text.

    Returns
    -------
    float or None
        Parsed percentage, or ``None`` if no ``%`` pattern is found.
    """
    if not value:
        return None
    match = re.search(r"(\d+[.,]?\d*)\s*%", value.replace(",", "."))
    return float(match.group(1)) if match else None
