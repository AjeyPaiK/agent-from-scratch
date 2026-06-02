"""Symbolic guardrails for the EU cosmetics compliance agent.

Re-exports the three guardrail stages (pre-input, pre-tool, post-output),
shared result types, and the agent scope string.

Notes
-----
Implements assignment section 1.1.
"""

from guardrails.post_output import check_post_output
from guardrails.pre_input import check_pre_input
from guardrails.pre_tool import check_pre_tool
from guardrails.topic_scope import AGENT_SCOPE
from guardrails.types import GuardrailReport, GuardrailVerdict

__all__ = [
    "AGENT_SCOPE",
    "GuardrailReport",
    "GuardrailVerdict",
    "check_post_output",
    "check_pre_input",
    "check_pre_tool",
]
