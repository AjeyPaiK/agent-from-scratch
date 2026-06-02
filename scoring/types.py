"""Shared types for code-based tool output scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolAccuracyScore:
    """Accuracy of one tool call's output vs an independent oracle.

    Attributes
    ----------
    tool : str
        Name of the scored tool (for example ``lookup_ingredient_regulation``).
    accuracy : float
        Similarity score in ``[0.0, 1.0]`` between tool output and oracle output.
    mismatches : list[str], optional
        Field names or keys that differ between actual and expected output,
        by default an empty list.
    oracle_version : str, optional
        Identifier for the annex snapshot used to build the oracle,
        by default ``"default"``.
    """

    tool: str
    accuracy: float
    mismatches: list[str] = field(default_factory=list)
    oracle_version: str = "default"
