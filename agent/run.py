"""Run the compliance agent from the command line."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.pipeline import run_turn


def main() -> int:
    """Execute one compliance query from ``sys.argv`` and print the result.

    Returns
    -------
    int
        ``0`` on success, ``1`` when usage is incorrect.
    """
    if len(sys.argv) < 2:
        print('Usage: python -m agent.run "Your question here"')
        return 1

    query = " ".join(sys.argv[1:])
    result = run_turn(query)
    if result.blocked:
        print(f"[blocked: {result.guardrails.pre_input.rule_id}]")
        print("[guardrails: pre-input failed — agent not invoked]")
        print()
        print(result.answer)
        return 0

    print(f"[intent: {result.intent.primary_intent}]")
    print(f"[guardrails: {'pass' if result.guardrails.all_passed else 'issues'}]")
    if result.tool_trace:
        print("[tools]")
        for entry in result.tool_trace:
            status = "BLOCKED" if entry.blocked else "ok"
            print(f"  {entry.step}. {entry.label} [{status}]")
            if entry.args:
                print(f"     in:  {entry.args}")
            preview = entry.output
            if isinstance(preview, dict):
                keys = ", ".join(preview.keys())
                print(f"     out: {{{keys}}}")
            else:
                text = str(preview)
                print(f"     out: {text[:120]}{'…' if len(text) > 120 else ''}")
    print()
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
