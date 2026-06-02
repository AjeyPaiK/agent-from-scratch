"""Tests for on-topic cosmetics compliance detection.

Notes
-----
Exercises topic-scope scoring, evidence heuristics, and pre-input integration
for on-topic and off-topic user messages.
"""

import pytest

from guardrails.pre_input import check_pre_input
from guardrails.topic_scope import (
    SCORE_THRESHOLD,
    compliance_evidence,
    detect_off_topic,
    has_compliance_evidence,
    is_cosmetics_compliance_topic,
    score_compliance,
)


@pytest.mark.parametrize(
    "message",
    [
        "Is retinol allowed in a leave-on hand lotion?",
        "What is the Annex III limit for phenoxyethanol?",
        "Is phenoxyethanol restricted in the EU?",
        "Is Glycerin allowed?",
        "Can you tell me what's the compliance status of Glycerin?",
    ],
)
def test_on_topic_cosmetics_messages(message):
    """Classify cosmetics compliance questions as on-topic.

    Parameters
    ----------
    message : str
        User message expected to relate to EU cosmetics compliance.

    Notes
    -----
    Both ``is_cosmetics_compliance_topic`` and ``detect_off_topic`` should
    indicate the message is in scope.
    """
    assert is_cosmetics_compliance_topic(message)
    assert detect_off_topic(message) is None


@pytest.mark.parametrize(
    "message",
    [
        "What is a consumer facing production grade webapp library?",
        "How do I deploy a React app to AWS?",
        "What is the capital of France?",
        "Is Chris Martin allowed to smoke?",
        "Is smoking allowed at concerts?",
        "Can you tell me the difference between a dancer and a cosmetic product?",
        "What is the difference between a car and a cosmetic product?",
    ],
)
def test_off_topic_messages(message):
    """Classify unrelated or riddle-style messages as off-topic.

    Parameters
    ----------
    message : str
        User message expected to fall outside cosmetics compliance scope.

    Notes
    -----
    Off-topic messages should fail ``is_cosmetics_compliance_topic`` and return
    a non-``None`` label from ``detect_off_topic``.
    """
    assert not is_cosmetics_compliance_topic(message)
    assert detect_off_topic(message) is not None


def test_glycerin_passes_pre_input():
    """Allow generic Glycerin compliance questions through pre-input guardrails.

    Notes
    -----
    Unknown INCI names with compliance intent should not be blocked at pre-input.
    """
    verdict = check_pre_input("Is Glycerin allowed?")
    assert verdict.passed


def test_glycerin_compliance_status_inquiry():
    """Recognize explicit compliance-status phrasing for Glycerin inquiries.

    Notes
    -----
    The message should contribute ``ingredient_status_inquiry`` evidence and
    pass pre-input checks.
    """
    q = "Can you tell me what's the compliance status of Glycerin?"
    assert "ingredient_status_inquiry" in compliance_evidence(q)
    assert check_pre_input(q).passed


@pytest.mark.parametrize(
    "message",
    [
        "What is the difference between Annex III and Annex V?",
        "How is retinol different from retinal in leave-on products?",
    ],
)
def test_compliance_comparisons_stay_on_topic(message):
    """Treat annex and ingredient comparisons as on-topic compliance questions.

    Parameters
    ----------
    message : str
        Comparison-style message anchored in cosmetics regulatory concepts.

    Notes
    -----
    Regulatory comparisons should remain in scope unlike unanchored riddle prompts.
    """
    assert is_cosmetics_compliance_topic(message)
    assert detect_off_topic(message) is None


def test_dancer_riddle_blocked_before_llm():
    """Block unanchored dancer-vs-cosmetic riddle prompts at pre-input.

    Notes
    -----
    The message should fail with rule id ``off_topic`` and produce no compliance
    evidence tokens.
    """
    verdict = check_pre_input(
        "Can you tell me the difference between a dancer and a cosmetic product?"
    )
    assert not verdict.passed
    assert verdict.rule_id == "off_topic"
    assert not has_compliance_evidence(
        "Can you tell me the difference between a dancer and a cosmetic product?"
    )


def test_bare_cosmetic_mention_has_no_evidence():
    """Assign no compliance evidence to casual cosmetic mentions.

    Notes
    -----
    Generic beauty chatter or definitional cosmetic questions without regulatory
    anchors should not score as compliance topics.
    """
    assert compliance_evidence("I love cosmetics and beauty trends") == []
    assert not is_cosmetics_compliance_topic("What is a cosmetic product?")


def test_general_eu_rules_has_evidence():
    """Detect compliance evidence in broad EU cosmetics regulation questions.

    Notes
    -----
    General annex and regulation inquiries should both produce evidence and pass
    topic classification.
    """
    assert has_compliance_evidence("Tell me about EU cosmetics regulation and annex rules")
    assert is_cosmetics_compliance_topic("Tell me about EU cosmetics regulation and annex rules")


def test_weighted_score_exposes_breakdown():
    """Expose per-signal contribution weights in the compliance score result.

    Notes
    -----
    A Glycerin status inquiry should pass, meet ``SCORE_THRESHOLD``, and show
    a weighted contribution for ``ingredient_status_inquiry``.
    """
    result = score_compliance("Can you tell me what's the compliance status of Glycerin?")
    assert result.passed
    assert result.score >= SCORE_THRESHOLD
    assert "ingredient_status_inquiry" in result.contributions
    assert result.contributions["ingredient_status_inquiry"] == 4.0


def test_comparison_riddle_negative_score():
    """Apply a comparison-without-anchor penalty for riddle-style prompts.

    Notes
    -----
    The dancer-vs-cosmetic riddle should fail scoring and record
    ``penalty:comparison_without_anchor`` in contributions.
    """
    msg = "Can you tell me the difference between a dancer and a cosmetic product?"
    result = score_compliance(msg)
    assert not result.passed
    assert "penalty:comparison_without_anchor" in result.contributions
