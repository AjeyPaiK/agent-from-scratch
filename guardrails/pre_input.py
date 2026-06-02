"""Stage 1 input guardrail for user messages.

Rule-based classifier that blocks malicious or out-of-scope inputs before
the LLM runs. Covers prompt injection, out-of-scope domains, non-EU
jurisdiction, and off-topic queries.

Notes
-----
Implements assignment section 1.1 (pre-input validation).
"""

from __future__ import annotations

import re

from guardrails.jurisdiction import BLOCK_MESSAGE, detect_non_eu_jurisdiction
from guardrails.topic_scope import detect_off_topic
from guardrails.types import GuardrailVerdict

PROMPT_INJECTION = re.compile(
    r"(ignore (all )?(previous|prior) instructions|you are now|system prompt|jailbreak)",
    re.I,
)

OUT_OF_SCOPE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "medical_advice",
        re.compile(
            r"\b(diagnos(e|is|ing)|prescri(be|ption)|what disease|treat my (rash|eczema|acne))\b",
            re.I,
        ),
        "Medical diagnosis and treatment are out of scope. Ask about cosmetic ingredient compliance.",
    ),
    (
        "formulation_recipe",
        re.compile(
            r"\b(formulate a product for me|full formula recipe|step-by-step formulation)\b",
            re.I,
        ),
        "Full product formulation is out of scope. Ask about a specific ingredient or labelling rule.",
    ),
]


def check_pre_input(user_message: str) -> GuardrailVerdict:
    """Validate a user message before it reaches the LLM.

    Runs prompt-injection detection, out-of-scope pattern matching,
    non-EU jurisdiction checks, and topic-scope scoring.

    Parameters
    ----------
    user_message : str
        Raw user input text.

    Returns
    -------
    GuardrailVerdict
        Verdict with ``stage="pre_input"``. ``passed`` is ``True`` when all
        checks succeed; otherwise ``passed`` is ``False`` with a ``rule_id``
        and user-facing ``message``.
    """
    text = (user_message or "").strip()
    if not text:
        return GuardrailVerdict(
            stage="pre_input",
            passed=False,
            rule_id="empty_input",
            message="Please enter a question about EU cosmetic ingredient compliance.",
        )

    if PROMPT_INJECTION.search(text):
        return GuardrailVerdict(
            stage="pre_input",
            passed=False,
            rule_id="prompt_injection",
            message="Request blocked — please ask a compliance question about EU cosmetics.",
        )

    for rule_id, pattern, message in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(text):
            return GuardrailVerdict(
                stage="pre_input",
                passed=False,
                rule_id=rule_id,
                message=message,
            )

    non_eu = detect_non_eu_jurisdiction(text)
    if non_eu:
        return GuardrailVerdict(
            stage="pre_input",
            passed=False,
            rule_id="non_eu_jurisdiction",
            message=BLOCK_MESSAGE.format(place=non_eu),
            details={"detected_jurisdiction": non_eu},
        )

    off_topic = detect_off_topic(text)
    if off_topic:
        return GuardrailVerdict(
            stage="pre_input",
            passed=False,
            rule_id="off_topic",
            message=off_topic,
        )

    return GuardrailVerdict(
        stage="pre_input",
        passed=True,
        rule_id="ok",
        message="Input accepted.",
    )
