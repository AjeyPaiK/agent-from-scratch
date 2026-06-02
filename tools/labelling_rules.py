"""Tool 3: Labelling and marketing obligations."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from data.cosing_api import search_ingredient
from data.product_types import get_product_type
from tools.messages import annex_absence_message


@tool
def get_labelling_marketing_rules(
    inci_name: str,
    product_category: str,
    concentration_percent: float | None = None,
) -> dict[str, Any]:
    """Return EU labelling requirements and marketing restrictions for an ingredient.

    Use for INCI list obligations, warnings, allergen labels, or packaging rules.

    Parameters
    ----------
    inci_name : str
        INCI name of the ingredient.
    product_category : str
        Product type id (e.g. ``leave_on_face_cream``).
    concentration_percent : float or None, optional
        Stated concentration for contextual labelling guidance.

    Returns
    -------
    dict[str, Any]
        Tool payload with labelling requirements, marketing restrictions, and
        regulatory references.
    """
    result = search_ingredient(inci_name)
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

        if match.legal_status == "prohibited":
            marketing_restrictions.append(
                f"Substance is prohibited ({match.annex_reference}) — must not be placed on the EU market."
            )
        if "preservative" in match.legal_status:
            labelling.append(
                f"Preservative must appear by INCI name on the ingredient list ({match.annex_reference})."
            )
        if "uv_filter" in match.legal_status:
            labelling.append(
                f"UV filter must be from Annex VI positive list ({match.annex_reference}). "
                "SPF claims require compliant filter system and testing."
            )
            references.append("Annex VI")

        if match.wording_of_conditions:
            labelling.append(match.wording_of_conditions.strip())
            references.append(match.annex_reference)
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
