"""Stage 2 tool-argument validator.

Symbolic validation of tool call arguments inside the graph, before
execution. Ensures INCI names, product categories, and concentration
values are well-formed.

Notes
-----
Implements assignment section 1.1 (pre-tool validation).
"""

from __future__ import annotations

import re
from typing import Any

from data.product_types import PRODUCT_TYPES
from guardrails.types import GuardrailVerdict

VALID_PRODUCT_IDS = frozenset(PRODUCT_TYPES)

TOOLS_REQUIRING_PRODUCT = frozenset(
    {"check_concentration_compliance", "get_labelling_marketing_rules"}
)

ALL_TOOL_NAMES = frozenset(
    {
        "lookup_ingredient_regulation",
        "check_concentration_compliance",
        "get_labelling_marketing_rules",
    }
)

INCI_NAME_PATTERN = re.compile(r"^[\w\s\-.',()/&+]+$")
MAX_INCI_LENGTH = 200


def _validate_inci(inci: str, tool_name: str, tool_args: dict[str, Any]) -> GuardrailVerdict | None:
    """Validate an INCI name for length and character set.

    Parameters
    ----------
    inci : str
        Trimmed INCI ingredient name from tool arguments.
    tool_name : str
        Name of the tool being validated.
    tool_args : dict[str, Any]
        Full tool argument payload, included in failure ``details``.

    Returns
    -------
    GuardrailVerdict or None
        A failed ``GuardrailVerdict`` when validation fails; ``None`` when
        the INCI name is acceptable.
    """
    if len(inci) < 2:
        return GuardrailVerdict(
            stage="pre_tool",
            passed=False,
            rule_id="missing_inci",
            message="Tool calls require a valid INCI name (at least 2 characters).",
            details={"tool": tool_name, "args": tool_args},
        )
    if len(inci) > MAX_INCI_LENGTH:
        return GuardrailVerdict(
            stage="pre_tool",
            passed=False,
            rule_id="invalid_inci_format",
            message=f"INCI name exceeds maximum length ({MAX_INCI_LENGTH} characters).",
            details={"tool": tool_name, "args": tool_args},
        )
    if not INCI_NAME_PATTERN.match(inci):
        return GuardrailVerdict(
            stage="pre_tool",
            passed=False,
            rule_id="invalid_inci_format",
            message="INCI name contains invalid characters for a cosmetic ingredient lookup.",
            details={"tool": tool_name, "args": tool_args},
        )
    return None


def check_pre_tool(tool_name: str, tool_args: dict[str, Any]) -> GuardrailVerdict:
    """Validate tool name and arguments before tool execution.

    Checks that the tool is known, the INCI name is present and well-formed,
    required product categories are valid, and concentration values are
    numeric and within 0–100 when supplied.

    Parameters
    ----------
    tool_name : str
        Name of the tool the agent intends to call.
    tool_args : dict[str, Any]
        Argument payload for the tool call.

    Returns
    -------
    GuardrailVerdict
        Verdict with ``stage="pre_tool"``. ``passed`` is ``True`` when all
        checks succeed; otherwise ``passed`` is ``False`` with a ``rule_id``
        and user-facing ``message``.
    """
    if tool_name not in ALL_TOOL_NAMES:
        return GuardrailVerdict(
            stage="pre_tool",
            passed=False,
            rule_id="unknown_tool",
            message=f"Unknown tool '{tool_name}'.",
            details={"tool": tool_name, "args": tool_args},
        )

    inci = (tool_args.get("inci_name") or "").strip()
    inci_verdict = _validate_inci(inci, tool_name, tool_args)
    if inci_verdict is not None:
        return inci_verdict

    if tool_name in TOOLS_REQUIRING_PRODUCT:
        category = (tool_args.get("product_category") or "").strip()
        if category not in VALID_PRODUCT_IDS:
            return GuardrailVerdict(
                stage="pre_tool",
                passed=False,
                rule_id="invalid_product_category",
                message=(
                    f"Unknown product category '{category}'. "
                    f"Use one of: {', '.join(sorted(VALID_PRODUCT_IDS))}."
                ),
                details={"tool": tool_name, "args": tool_args},
            )

    if tool_name == "check_concentration_compliance":
        if "concentration_percent" in tool_args:
            conc = tool_args.get("concentration_percent")
            if conc is None:
                return GuardrailVerdict(
                    stage="pre_tool",
                    passed=False,
                    rule_id="missing_concentration",
                    message="concentration_percent was provided but is empty.",
                    details={"tool": tool_name, "args": tool_args},
                )
            if not isinstance(conc, (int, float)):
                return GuardrailVerdict(
                    stage="pre_tool",
                    passed=False,
                    rule_id="invalid_concentration",
                    message="concentration_percent must be a number.",
                    details={"tool": tool_name, "args": tool_args},
                )
            if conc < 0 or conc > 100:
                return GuardrailVerdict(
                    stage="pre_tool",
                    passed=False,
                    rule_id="invalid_concentration",
                    message="concentration_percent must be between 0 and 100.",
                    details={"tool": tool_name, "args": tool_args},
                )

    return GuardrailVerdict(
        stage="pre_tool",
        passed=True,
        rule_id="ok",
        message=f"Tool call accepted: {tool_name}.",
        details={"tool": tool_name},
    )
