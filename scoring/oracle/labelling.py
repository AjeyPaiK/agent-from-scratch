"""Oracle for ``get_labelling_marketing_rules`` — built from pinned annex CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.product_types import get_product_type
from scoring.oracle.csv_reader import search_annex_rows
from tools.messages import annex_absence_message


def oracle_get_labelling_marketing_rules(
    inci_name: str,
    product_category: str,
    concentration_percent: float | None = None,
    *,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Expected output for ``get_labelling_marketing_rules`` from pinned CSV data.

    Mirrors the live-tool logic but reads annex rows from the snapshot oracles
    instead of the CosIng API.
    """
    result = search_annex_rows(inci_name, snapshot_dir=snapshot_dir)
    if not result.found:
        return {"found": False, "inci_name": inci_name, "message": annex_absence_message(inci_name)}

    product = get_product_type(product_category)

    labelling: list[str] = [
        "List ingredients on the packaging in descending order of weight (Art. 19(1)(g), Regulation 1223/2009)."
    ]
    marketing_restrictions: list[str] = []
    references: list[str] = ["Regulation (EC) No 1223/2009, Art. 19"]

    seen: set[tuple[str, str]] = set()
    for match in result.matches:
        key = (match.annex, match.entry_number)
        if key in seen:
            continue
        seen.add(key)

        annex_reference = f"Annex {match.annex}, entry {match.entry_number}"
        if match.legal_status == "prohibited":
            marketing_restrictions.append(
                f"Substance is prohibited ({annex_reference}) — must not be placed on the EU market."
            )
        if "preservative" in match.legal_status:
            labelling.append(
                f"Preservative must appear by INCI name on the ingredient list ({annex_reference})."
            )
        if "uv_filter" in match.legal_status:
            labelling.append(
                f"UV filter must be from Annex VI positive list ({annex_reference}). "
                "SPF claims require compliant filter system and testing."
            )
            references.append("Annex VI")

        if match.wording_of_conditions:
            labelling.append(match.wording_of_conditions.strip())
            references.append(annex_reference)
        if match.other_conditions:
            labelling.append(match.other_conditions.strip())

    if product and product.annex_category == "leave_on":
        labelling.append(
            "Fragrance allergens above 0.001% in leave-on products must be individually labelled (Annex III)."
        )

    marketing_restrictions.append(
        "Do not claim medicinal effects (treatment, cure, prevention of disease) — Art. 2(1)(a)."
    )

    return {
        "found": True,
        "inci_name": result.matched_name or inci_name,
        "product_category": product.id if product else product_category,
        "concentration_percent": concentration_percent,
        "labelling_requirements": labelling,
        "marketing_restrictions": marketing_restrictions,
        "references": sorted(set(references)),
    }
