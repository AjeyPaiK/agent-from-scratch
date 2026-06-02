"""EU cosmetics compliance tools for the ReAct agent.

Exports the three LangChain tools and the ``ALL_TOOLS`` registry used by the agent graph.
"""

from tools.check_concentration import check_concentration_compliance
from tools.labelling_rules import get_labelling_marketing_rules
from tools.lookup_ingredient import lookup_ingredient_regulation

ALL_TOOLS = [
    lookup_ingredient_regulation,
    check_concentration_compliance,
    get_labelling_marketing_rules,
]

__all__ = [
    "ALL_TOOLS",
    "lookup_ingredient_regulation",
    "check_concentration_compliance",
    "get_labelling_marketing_rules",
]
