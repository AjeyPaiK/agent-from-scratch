"""Oracle functions that read pinned annex CSV snapshots.

Notes
-----
Public API: ``oracle_lookup_ingredient_regulation``,
``oracle_check_concentration_compliance``, ``oracle_get_labelling_marketing_rules``.
"""

from scoring.oracle.concentration import oracle_check_concentration_compliance
from scoring.oracle.labelling import oracle_get_labelling_marketing_rules
from scoring.oracle.lookup import oracle_lookup_ingredient_regulation

__all__ = [
    "oracle_check_concentration_compliance",
    "oracle_get_labelling_marketing_rules",
    "oracle_lookup_ingredient_regulation",
]
