"""Zero-LLM intent classifier using pattern and keyword rules only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INTENT_TOOLS: dict[str, list[str]] = {
    "ingredient_allowed": ["lookup_ingredient_regulation"],
    "concentration_limit": [
        "lookup_ingredient_regulation",
        "check_concentration_compliance",
    ],
    "labelling_rules": [
        "lookup_ingredient_regulation",
        "get_labelling_marketing_rules",
    ],
    "general_compliance": [
        "lookup_ingredient_regulation",
        "check_concentration_compliance",
        "get_labelling_marketing_rules",
    ],
}

INTENT_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "labelling_rules",
        "Labelling and packaging obligations",
        re.compile(
            r"\b(label|labelling|labeling|INCI list|warning|allergen|packaging text)\b",
            re.I,
        ),
    ),
    (
        "concentration_limit",
        "Concentration / maximum percentage",
        re.compile(
            r"(\d+[.,]?\d*\s*%|concentration|maximum|limit|how much|percent|ppm)",
            re.I,
        ),
    ),
    (
        "ingredient_allowed",
        "Ingredient allowed / prohibited status",
        re.compile(
            r"\b(allowed|permitted|prohibited|banned|restricted|can I use|is .+ allowed)\b",
            re.I,
        ),
    ),
]


@dataclass
class IntentResult:
    """Result of rule-based intent classification.

    Attributes
    ----------
    primary_intent : str
        Intent identifier (e.g. ``"concentration_limit"``).
    label : str
        Human-readable intent label for the UI.
    recommended_tools : list[str]
        Tool names suggested for this intent.
    matched_rules : list[str]
        Intent rule ids that matched the message.
    confidence : {"high", "medium", "low"}
        Confidence derived from the number of matched rules.
    """

    primary_intent: str
    label: str
    recommended_tools: list[str]
    matched_rules: list[str] = field(default_factory=list)
    confidence: str = "high"


def classify_intent(user_message: str) -> IntentResult:
    """Classify the user's question into a compliance intent without an LLM.

    Parameters
    ----------
    user_message : str
        Raw user question text.

    Returns
    -------
    IntentResult
        Primary intent, recommended tools, and confidence score.
    """
    text = (user_message or "").strip()
    matched: list[tuple[str, str]] = []

    for intent_id, label, pattern in INTENT_RULES:
        if pattern.search(text):
            matched.append((intent_id, label))

    if not matched:
        return IntentResult(
            primary_intent="general_compliance",
            label="General EU compliance question",
            recommended_tools=INTENT_TOOLS["general_compliance"],
            matched_rules=[],
            confidence="low",
        )

    primary_id, primary_label = matched[0]
    tools: list[str] = []
    for intent_id, _ in matched:
        for tool in INTENT_TOOLS[intent_id]:
            if tool not in tools:
                tools.append(tool)

    return IntentResult(
        primary_intent=primary_id,
        label=primary_label,
        recommended_tools=tools,
        matched_rules=[m[0] for m in matched],
        confidence="high" if len(matched) == 1 else "medium",
    )
