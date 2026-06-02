"""Shared legal-status labels for ingredient lookup tools and scoring oracles.

Attributes
----------
LEGAL_SUMMARY : dict[str, str]
    Maps normalized ``legal_status`` codes to short human-readable summaries.
"""

from __future__ import annotations

LEGAL_SUMMARY: dict[str, str] = {
    "prohibited": "Prohibited in EU cosmetic products (Annex II).",
    "restricted": "Restricted — allowed only under Annex III conditions.",
    "positive_list_preservative": "Listed preservative (Annex V) — allowed under listed conditions.",
    "positive_list_colorant": "Listed colorant (Annex IV) — allowed under listed conditions.",
    "positive_list_uv_filter": "Listed UV filter (Annex VI) — allowed under listed conditions.",
}
