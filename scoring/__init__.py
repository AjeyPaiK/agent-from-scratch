"""Code-based tool output scoring (oracle vs tool output).

Compare agent tool outputs against independent oracle implementations built
from pinned annex CSV snapshots.

Notes
-----
Public API: ``ToolAccuracyScore``, ``TurnTrustworthiness``,
``apply_accuracy_scores``, ``compute_turn_trustworthiness``, ``score_tool_call``.
"""

from scoring.score_tool_call import score_tool_call
from scoring.trace import apply_accuracy_scores
from scoring.trustworthiness import TurnTrustworthiness, compute_turn_trustworthiness
from scoring.types import ToolAccuracyScore

__all__ = [
    "ToolAccuracyScore",
    "TurnTrustworthiness",
    "apply_accuracy_scores",
    "compute_turn_trustworthiness",
    "score_tool_call",
]
