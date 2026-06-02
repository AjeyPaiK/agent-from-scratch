"""Attach oracle accuracy scores to a tool trace."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent.tool_trace import ToolTraceEntry
from scoring.score_tool_call import score_tool_call


def apply_accuracy_scores(
    entries: list[ToolTraceEntry],
    *,
    snapshot_dir: Path | None = None,
) -> list[ToolTraceEntry]:
    """Score each executed tool call against its oracle when available.

    Parameters
    ----------
    entries : list[ToolTraceEntry]
        Tool trace entries produced during an agent turn.
    snapshot_dir : Path or None, optional
        Directory containing pinned annex CSV snapshots. When ``None``, uses
        the default snapshot from settings, by default ``None``.

    Returns
    -------
    list[ToolTraceEntry]
        A new list of trace entries with ``accuracy``, ``accuracy_mismatches``,
        and ``oracle_version`` populated where an oracle exists.

    Notes
    -----
    Blocked tool calls and entries whose output is not a dict are returned
    unchanged without scoring.
    """
    scored: list[ToolTraceEntry] = []
    for entry in entries:
        if entry.blocked or not isinstance(entry.output, dict):
            scored.append(entry)
            continue
        result = score_tool_call(entry.name, entry.args, entry.output, snapshot_dir=snapshot_dir)
        if result is None:
            scored.append(entry)
            continue
        scored.append(
            replace(
                entry,
                accuracy=result.accuracy,
                accuracy_mismatches=result.mismatches,
                oracle_version=result.oracle_version,
            )
        )
    return scored
