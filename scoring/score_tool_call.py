"""Dispatch oracle scoring for individual tool calls."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.tool_args import sanitize_tool_kwargs
from config.settings import ANNEX_SNAPSHOT_DIR
from scoring.compare import (
    compare_concentration_output,
    compare_labelling_output,
    compare_lookup_output,
)
from scoring.oracle.concentration import oracle_check_concentration_compliance
from scoring.oracle.labelling import oracle_get_labelling_marketing_rules
from scoring.oracle.lookup import oracle_lookup_ingredient_regulation
from scoring.types import ToolAccuracyScore

_COMPARATORS: dict[str, Callable[[dict[str, Any], dict[str, Any]], tuple[float, list[str]]]] = {
    "lookup_ingredient_regulation": compare_lookup_output,
    "check_concentration_compliance": compare_concentration_output,
    "get_labelling_marketing_rules": compare_labelling_output,
}


def _snapshot_version(snapshot_dir: Path | None = None) -> str:
    """Return a version label for the annex snapshot directory.

    Parameters
    ----------
    snapshot_dir : Path or None, optional
        Snapshot root directory. When ``None``, uses ``ANNEX_SNAPSHOT_DIR``,
        by default ``None``.

    Returns
    -------
    str
        The snapshot directory name, used as ``oracle_version`` in scores.

    Notes
    -----
    If ``manifest.json`` exists under the snapshot root, the directory name
    still serves as the version identifier.
    """
    root = snapshot_dir or ANNEX_SNAPSHOT_DIR
    manifest = root / "manifest.json"
    if manifest.exists():
        return root.name
    return root.name


def score_tool_call(
    tool_name: str,
    args: dict[str, Any],
    output: Any,
    *,
    snapshot_dir: Path | None = None,
) -> ToolAccuracyScore | None:
    """Return an accuracy score when an oracle exists for this tool.

    Parameters
    ----------
    tool_name : str
        Registered tool name (for example ``lookup_ingredient_regulation``).
    args : dict[str, Any]
        Arguments passed to the tool call.
    output : Any
        Tool output to compare against the oracle.
    snapshot_dir : Path or None, optional
        Directory containing pinned annex CSV snapshots, by default ``None``.

    Returns
    -------
    ToolAccuracyScore or None
        Accuracy result when the tool is scorable and output is valid; otherwise
        ``None``.

    Notes
    -----
    Returns ``None`` when output is not a dict, the call was guardrail-blocked,
    or no oracle/comparator is registered for ``tool_name``.
    """
    if not isinstance(output, dict) or output.get("guardrail_blocked"):
        return None

    compare = _COMPARATORS.get(tool_name)
    if compare is None:
        return None

    args = sanitize_tool_kwargs(tool_name, dict(args))
    version = _snapshot_version(snapshot_dir)

    if tool_name == "lookup_ingredient_regulation":
        expected = oracle_lookup_ingredient_regulation(
            args.get("inci_name", ""),
            args.get("cas_number"),
            snapshot_dir=snapshot_dir,
        )
    elif tool_name == "check_concentration_compliance":
        expected = oracle_check_concentration_compliance(
            args.get("inci_name", ""),
            args.get("product_category", ""),
            args.get("concentration_percent"),
            snapshot_dir=snapshot_dir,
        )
    elif tool_name == "get_labelling_marketing_rules":
        expected = oracle_get_labelling_marketing_rules(
            args.get("inci_name", ""),
            args.get("product_category", ""),
            args.get("concentration_percent"),
            snapshot_dir=snapshot_dir,
        )
    else:
        return None

    accuracy, mismatches = compare(output, expected)
    return ToolAccuracyScore(
        tool=tool_name,
        accuracy=accuracy,
        mismatches=mismatches,
        oracle_version=version,
    )
