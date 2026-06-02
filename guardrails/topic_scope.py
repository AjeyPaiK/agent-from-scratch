"""Detect whether a user message is on-topic for the EU cosmetics agent.

Weighted evidence score using symbolic feature detectors (no LLM).
Features are binary; the total score is the weighted sum minus penalties.
A message passes when its score meets or exceeds the configured threshold.

Notes
-----
Hard vetoes (never scored): empty text, deny-list domains, and
"allowed to {activity}" phrasing without an ingredient signal.

Score threshold is tuned on ``tests/test_topic_scope.py`` exemplars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

AGENT_SCOPE = (
    "EU cosmetic ingredient compliance under Regulation (EC) No 1223/2009 and "
    "Regulation (EU) No 655/2013: INCI annex status (II–VI), concentration limits, "
    "and labelling obligations. Geography: European Union / EEA only."
)

# Pass threshold — tuned on tests/test_topic_scope.py exemplars.
SCORE_THRESHOLD = 2.0

# --- Feature detectors --------------------------------------------------------

NAMED_INGREDIENT = re.compile(
    r"\b("
    r"retinol|retinal|niacinamide|phenoxyethanol|salicylic acid|hyaluronic acid|"
    r"paraben|formaldehyde|tocopherol|ascorbic acid|glycolic acid|lactic acid|"
    r"benzoyl peroxide|zinc oxide|titanium dioxide|fragrance|allantoin|urea"
    r")\b",
    re.I,
)

INGREDIENT_SHAPED = re.compile(
    r"\b[a-z][a-z\-']{2,}(?:ol|one|ate|ide|ene|ium|oin|yl|ic acid)\b",
    re.I,
)

INGREDIENT_STATUS_QUESTION = re.compile(
    r"\b(?:is|are|can i use|may i use)\b.+\b(?:allowed|permitted|prohibited|banned|restricted)\b"
    r"(?:\s+(?:in|for|on|under)\b|\s*[?.!]?\s*$)",
    re.I,
)

INGREDIENT_STATUS_INQUIRY = re.compile(
    r"\b(?:compliance|regulatory|legal)\s+status\s+of\s+"
    r"([A-Za-z][A-Za-z0-9\-']+(?:\s+[A-Za-z][A-Za-z0-9\-']+)?)\b",
    re.I,
)

ANNEX_REGULATORY = re.compile(
    r"\b("
    r"annex|cosing|inci list|1223/2009|regulation \(ec\)|655/2013|"
    r"prohibited list|positive list|restricted substances"
    r")\b",
    re.I,
)

CONCENTRATION_SIGNAL = re.compile(
    r"(\d+[.,]?\d*\s*%|concentration|maximum|limit|how much|percent|ppm)",
    re.I,
)

LABELLING_SIGNAL = re.compile(
    r"\b(label|labelling|labeling|packag(?:e|ing)|warning|allergen|marketing claim|spf)\b",
    re.I,
)

PRODUCT_CONTEXT = re.compile(
    r"\b("
    r"leave-on|leave on|rinse-off|rinse off|"
    r"face cream|body lotion|hand lotion|shampoo|cleanser|sunscreen|deodorant|"
    r"hair dye|oral care|serum|lotion|cream|"
    r"product category|leave_on_|rinse_off_"
    r")\b",
    re.I,
)

REGULATORY_VERB = re.compile(
    r"\b(restricted|prohibited|banned|allowed|permitted|compliant|compliance)\b",
    re.I,
)

EU_COSMETICS_TOPIC = re.compile(
    r"\b(?:eu|e\.u\.|european union|eea)\b.{0,40}\b(?:cosmetic|ingredient|annex|regulation)\b|"
    r"\b(?:cosmetic|ingredient|annex|regulation)\b.{0,40}\b(?:eu|e\.u\.|european union|eea)\b",
    re.I,
)

COMPARISON_QUESTION = re.compile(
    r"\b("
    r"difference between|differences between|"
    r"how (?:is|are)\s+.+\s+different from|"
    r"compare\s+.+\s+(?:to|with|and)\b|"
    r"\bvs\.?\s|\bversus\b"
    r")\b",
    re.I,
)

WEAK_COSMETIC = re.compile(r"\bcosmetics?\b", re.I)

OFF_TOPIC_DOMAIN = re.compile(
    r"\b("
    r"smok(?:e|ing|ed)|cigarette|vap(?:e|ing)|alcohol|beer|wine|"
    r"webapp|website|react|vue|angular|javascript|typescript|python|library|framework|"
    r"deploy|kubernetes|aws|azure|"
    r"capital of|weather|football|soccer|basketball|movie|song|celebrity|"
    r"diagnos(?:e|is)|prescri(?:be|ption)|treat my"
    r")\b",
    re.I,
)

ALLOWED_TO_ACTIVITY = re.compile(r"\ballowed\s+to\s+\w", re.I)

OFF_TOPIC_MESSAGE = (
    "This agent covers EU cosmetic ingredient compliance only — INCI status, "
    "annex limits, and labelling under Regulation (EC) No 1223/2009. "
    "Please ask about an ingredient, product type, or EU regulatory rule."
)

# Feature weights (wᵢ)
FEATURE_WEIGHTS: dict[str, float] = {
    "ingredient_status_question": 4.0,
    "ingredient_status_inquiry": 4.0,
    "ingredient_token": 3.0,
    "annex_regulatory": 3.0,
    "concentration": 2.0,
    "labelling": 2.0,
    "product_and_regulatory_verb": 3.0,
    "eu_cosmetics_topic": 3.0,
}

# Penalties (subtracted when condition holds)
PENALTY_WEIGHTS: dict[str, float] = {
    "comparison_without_anchor": 5.0,
    "weak_cosmetic_only": 2.0,
}


@dataclass(frozen=True)
class ComplianceFeatures:
    """Binary feature vector for one user message.

    Attributes
    ----------
    ingredient_status_question : bool
        Message asks whether an ingredient is allowed or restricted.
    ingredient_status_inquiry : bool
        Message asks for the compliance status of a named ingredient.
    ingredient_token : bool
        Message contains a named or ingredient-shaped token.
    annex_regulatory : bool
        Message references annex lists or EU regulatory instruments.
    concentration : bool
        Message mentions concentration, limits, or percentages.
    labelling : bool
        Message mentions labelling or packaging obligations.
    product_and_regulatory_verb : bool
        Message combines product context with a regulatory verb.
    eu_cosmetics_topic : bool
        Message explicitly links EU geography to cosmetics regulation.
    comparison_question : bool
        Message is a comparison-style question.
    weak_cosmetic : bool
        Message mentions cosmetics without stronger regulatory signals.
    """

    ingredient_status_question: bool = False
    ingredient_status_inquiry: bool = False
    ingredient_token: bool = False
    annex_regulatory: bool = False
    concentration: bool = False
    labelling: bool = False
    product_and_regulatory_verb: bool = False
    eu_cosmetics_topic: bool = False
    comparison_question: bool = False
    weak_cosmetic: bool = False


@dataclass(frozen=True)
class ComplianceScore:
    """Auditable weighted score breakdown for topic classification.

    Attributes
    ----------
    score : float
        Final score after feature weights and penalties.
    threshold : float
        Pass threshold used for the decision.
    passed : bool
        ``True`` when ``score >= threshold``.
    contributions : dict[str, float]
        Per-feature weights and penalty line items.
    """

    score: float
    threshold: float
    passed: bool
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def active_features(self) -> list[str]:
        """Feature names with positive weight contributions.

        Returns
        -------
        list[str]
            Keys from ``contributions`` whose values are greater than zero.
        """
        return [k for k, v in self.contributions.items() if v > 0]


def _has_ingredient_signal(text: str) -> bool:
    """Return whether text contains a named or ingredient-shaped token.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    bool
        ``True`` when a known ingredient or ingredient-shaped token is found.
    """
    return bool(NAMED_INGREDIENT.search(text) or INGREDIENT_SHAPED.search(text))


def _has_ingredient_status_inquiry(text: str) -> bool:
    """Return whether text asks for an ingredient's compliance status.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    bool
        ``True`` when a status inquiry matches and the captured ingredient
        is not from an off-topic domain.
    """
    match = INGREDIENT_STATUS_INQUIRY.search(text)
    if not match:
        return False
    return not OFF_TOPIC_DOMAIN.search(match.group(1))


def _has_regulatory_anchor(features: ComplianceFeatures) -> bool:
    """Return whether features include a strong regulatory anchor.

    Parameters
    ----------
    features : ComplianceFeatures
        Extracted binary feature vector.

    Returns
    -------
    bool
        ``True`` when at least one ingredient or annex regulatory feature
        is active.
    """
    return (
        features.ingredient_status_question
        or features.ingredient_status_inquiry
        or features.ingredient_token
        or features.annex_regulatory
    )


def extract_features(text: str) -> ComplianceFeatures:
    """Extract the binary feature vector from a message.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    ComplianceFeatures
        Binary feature flags detected in ``text``. Returns an empty feature
        vector when ``text`` is blank.
    """
    if not text or not text.strip():
        return ComplianceFeatures()

    has_product = bool(PRODUCT_CONTEXT.search(text))
    has_regulatory = bool(REGULATORY_VERB.search(text))

    return ComplianceFeatures(
        ingredient_status_question=bool(INGREDIENT_STATUS_QUESTION.search(text)),
        ingredient_status_inquiry=_has_ingredient_status_inquiry(text),
        ingredient_token=_has_ingredient_signal(text),
        annex_regulatory=bool(ANNEX_REGULATORY.search(text)),
        concentration=bool(CONCENTRATION_SIGNAL.search(text)),
        labelling=bool(LABELLING_SIGNAL.search(text)),
        product_and_regulatory_verb=has_product and has_regulatory,
        eu_cosmetics_topic=bool(EU_COSMETICS_TOPIC.search(text)),
        comparison_question=bool(COMPARISON_QUESTION.search(text)),
        weak_cosmetic=bool(WEAK_COSMETIC.search(text)),
    )


def score_compliance(text: str) -> ComplianceScore:
    """Compute weighted score and pass/fail against ``SCORE_THRESHOLD``.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    ComplianceScore
        Score breakdown including contributions, threshold, and pass flag.
    """
    features = extract_features(text)
    contributions: dict[str, float] = {}
    total = 0.0

    for name, weight in FEATURE_WEIGHTS.items():
        if getattr(features, name):
            contributions[name] = weight
            total += weight

    if features.comparison_question and not _has_regulatory_anchor(features):
        contributions["penalty:comparison_without_anchor"] = -PENALTY_WEIGHTS[
            "comparison_without_anchor"
        ]
        total -= PENALTY_WEIGHTS["comparison_without_anchor"]

    positive_before_weak = total
    if (
        features.weak_cosmetic
        and positive_before_weak < SCORE_THRESHOLD
        and not _has_regulatory_anchor(features)
    ):
        contributions["penalty:weak_cosmetic_only"] = -PENALTY_WEIGHTS["weak_cosmetic_only"]
        total -= PENALTY_WEIGHTS["weak_cosmetic_only"]

    passed = total >= SCORE_THRESHOLD
    return ComplianceScore(
        score=total,
        threshold=SCORE_THRESHOLD,
        passed=passed,
        contributions=contributions,
    )


def _hard_veto(text: str) -> bool:
    """Return whether a message is unconditionally off-topic.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    bool
        ``True`` for empty text, deny-list domains, or "allowed to {activity}"
        phrasing without an ingredient signal.
    """
    if not text or not text.strip():
        return True
    if OFF_TOPIC_DOMAIN.search(text):
        return True
    if ALLOWED_TO_ACTIVITY.search(text) and not _has_ingredient_signal(text):
        return True
    return False


def compliance_evidence(text: str) -> list[str]:
    """Return feature names with positive weight contributions.

    Backward-compatible debug helper that skips hard-vetoed messages.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    list[str]
        Active positive feature names, or an empty list when vetoed.
    """
    if _hard_veto(text):
        return []
    return score_compliance(text).active_features


def has_compliance_evidence(text: str) -> bool:
    """Return whether the message gives tools something actionable.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    bool
        ``True`` when the message is not hard-vetoed and passes the
        compliance score threshold.
    """
    if _hard_veto(text):
        return False
    return score_compliance(text).passed


def is_cosmetics_compliance_topic(text: str) -> bool:
    """Return whether the message is plausibly about EU cosmetics compliance.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    bool
        ``True`` when ``has_compliance_evidence`` returns ``True``.
    """
    return has_compliance_evidence(text)


def detect_off_topic(text: str) -> str | None:
    """Return a user-facing block message when the query is off-topic.

    Parameters
    ----------
    text : str
        User message text.

    Returns
    -------
    str or None
        ``OFF_TOPIC_MESSAGE`` when the message is vetoed or fails scoring;
        ``None`` when the message is on-topic.
    """
    if _hard_veto(text) or not score_compliance(text).passed:
        return OFF_TOPIC_MESSAGE
    return None
