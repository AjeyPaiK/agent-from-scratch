"""Shared guardrail result types.

Dataclasses representing per-stage verdicts and an aggregated report
for a single agent turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailVerdict:
    """Outcome of a single guardrail check.

    Attributes
    ----------
    stage : str
        Guardrail stage identifier: ``"pre_input"``, ``"pre_tool"``, or
        ``"post_output"``.
    passed : bool
        ``True`` when the check succeeded.
    rule_id : str
        Machine-readable rule identifier (e.g. ``"ok"``, ``"empty_input"``).
    message : str
        Human-readable explanation of the outcome.
    details : dict[str, Any]
        Optional structured context for logging or debugging.
    """

    stage: str  # pre_input | pre_tool | post_output
    passed: bool
    rule_id: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailReport:
    """Aggregated guardrail results for one agent turn.

    Attributes
    ----------
    pre_input : GuardrailVerdict
        Verdict from the input guardrail stage.
    pre_tool : list[GuardrailVerdict]
        Verdicts from tool-argument validation, one per tool call checked.
    post_output : GuardrailVerdict or None
        Verdict from the output guardrail stage, if run.
    """

    pre_input: GuardrailVerdict
    pre_tool: list[GuardrailVerdict] = field(default_factory=list)
    post_output: GuardrailVerdict | None = None

    @property
    def all_passed(self) -> bool:
        """Return whether every recorded stage passed.

        Returns
        -------
        bool
            ``True`` when pre-input, all pre-tool checks, and post-output
            (if present) passed; ``False`` otherwise.
        """
        if not self.pre_input.passed:
            return False
        if any(not v.passed for v in self.pre_tool):
            return False
        if self.post_output and not self.post_output.passed:
            return False
        return True
