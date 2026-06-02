"""Stage 3 output verifier for agent responses.

Cross-check the LLM's final answer against tool results to detect
hallucinations, unsourced regulatory claims, and contradictions with
tool-reported statuses.

Notes
-----
Implements assignment section 1.1 (post-output validation).
"""

from __future__ import annotations

import json
import re
from typing import Any

from guardrails.types import GuardrailVerdict

REGULATORY_ASSERTION = re.compile(
    r"\b(prohibited|not allowed|allowed|maximum|compliant|non-compliant|Annex [IVX]+)\b",
    re.I,
)

ANNEX_CITATION = re.compile(r"Annex\s+[IVX]+,\s*entry\s+\d+", re.I)

ANSWER_PROHIBITED = re.compile(
    r"\b(prohibited|banned|not permitted|not allowed)\b",
    re.I,
)

ANSWER_ALLOWED = re.compile(
    r"\b(allowed|permitted|can be used)\b",
    re.I,
)

TOOL_STATUS_KEYS = ("overall_status", "legal_status")


def _parse_tool_payload(raw: str) -> dict[str, Any] | None:
    """Parse a tool output string as JSON.

    Parameters
    ----------
    raw : str
        Raw tool output, expected to be a JSON object string.

    Returns
    -------
    dict[str, Any] or None
        Parsed dictionary when ``raw`` is valid JSON object text;
        ``None`` on decode failure or non-dict payloads.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _collect_statuses(tool_outputs: list[str]) -> set[str]:
    """Extract normalized legal statuses from tool output payloads.

    Parameters
    ----------
    tool_outputs : list[str]
        Raw JSON strings returned by compliance tools.

    Returns
    -------
    set[str]
        Lowercase status strings from ``overall_status``, ``legal_status``,
        and nested ``annex_entries`` fields.
    """
    statuses: set[str] = set()
    for raw in tool_outputs:
        data = _parse_tool_payload(raw)
        if not data:
            continue
        for key in TOOL_STATUS_KEYS:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                statuses.add(val.strip().lower())
        for entry in data.get("annex_entries") or []:
            if isinstance(entry, dict):
                ls = entry.get("legal_status")
                if isinstance(ls, str) and ls.strip():
                    statuses.add(ls.strip().lower())
    return statuses


def _contradicts_tool_status(answer: str, statuses: set[str]) -> str | None:
    """Detect keyword conflicts between the answer and tool statuses.

    Parameters
    ----------
    answer : str
        Final agent response text.
    statuses : set[str]
        Normalized legal statuses collected from tool outputs.

    Returns
    -------
    str or None
        Short human-readable reason when the answer contradicts tool
        statuses; ``None`` when no conflict is detected.
    """
    if not statuses:
        return None

    says_prohibited = bool(ANSWER_PROHIBITED.search(answer))
    says_allowed = bool(ANSWER_ALLOWED.search(answer))

    tool_has_prohibited = "prohibited" in statuses
    tool_has_restricted = "restricted" in statuses

    if says_prohibited and not tool_has_prohibited and not tool_has_restricted:
        return "Answer asserts prohibition but tools did not return a prohibited/restricted status."

    if says_allowed and tool_has_prohibited:
        return "Answer asserts allowance but tools reported prohibited status."

    if says_allowed and tool_has_restricted:
        return "Answer asserts allowance but tools reported restricted status."

    return None


def check_post_output(
    answer: str,
    *,
    tool_names_used: list[str],
    tool_outputs: list[str],
) -> GuardrailVerdict:
    """Verify the agent's final answer against tool evidence.

    Flags empty responses, unsourced regulatory claims, hallucinated annex
    citations, and answers that contradict tool-reported legal statuses.

    Parameters
    ----------
    answer : str
        Final agent response text.
    tool_names_used : list[str]
        Names of tools invoked during the turn.
    tool_outputs : list[str]
        Raw JSON strings returned by those tools.

    Returns
    -------
    GuardrailVerdict
        Verdict with ``stage="post_output"``. ``passed`` is ``True`` when all
        checks succeed; otherwise ``passed`` is ``False`` with a ``rule_id``
        and user-facing ``message``.
    """
    text = (answer or "").strip()
    if not text:
        return GuardrailVerdict(
            stage="post_output",
            passed=False,
            rule_id="empty_response",
            message="Agent returned an empty response.",
        )

    if REGULATORY_ASSERTION.search(text) and not tool_names_used:
        return GuardrailVerdict(
            stage="post_output",
            passed=False,
            rule_id="unsourced_regulatory_claim",
            message="Regulatory conclusion stated without calling any compliance tools.",
            details={"tools_used": tool_names_used},
        )

    citations = ANNEX_CITATION.findall(text)
    if citations:
        corpus = " ".join(tool_outputs).lower()
        for cite in citations:
            if cite.lower() not in corpus:
                return GuardrailVerdict(
                    stage="post_output",
                    passed=False,
                    rule_id="hallucinated_annex_citation",
                    message=f"Cited '{cite}' but it was not returned by any tool output.",
                    details={"citation": cite},
                )

    statuses = _collect_statuses(tool_outputs)
    if tool_names_used and statuses:
        conflict = _contradicts_tool_status(text, statuses)
        if conflict:
            return GuardrailVerdict(
                stage="post_output",
                passed=False,
                rule_id="contradicts_tool_status",
                message=conflict,
                details={"tool_statuses": sorted(statuses)},
            )

    return GuardrailVerdict(
        stage="post_output",
        passed=True,
        rule_id="ok",
        message="Output checks passed.",
    )
