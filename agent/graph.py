"""Explicit LangGraph StateGraph for the EU cosmetics compliance agent.

The whole turn is one compiled graph. Every symbolic guardrail is a named node,
and the ReAct tool loop is the ``agent ⇄ tools`` cycle::

    START → intent → input_guard ─(blocked)─→ END
                          │
                       (passed)
                          ▼
                        agent ─(tool calls)─→ tool_guard → tools ──┐
                          │                                        │
                          └────────────────◄──────────────────────┘
                          │
                     (no tool calls)
                          ▼
                     exposition → output_verify → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    agent_node,
    exposition_node,
    input_guard_node,
    intent_node,
    output_verify_node,
    route_after_agent,
    route_after_input,
    tool_guard_node,
    tools_node,
)
from agent.state import AgentState

_COMPILED = None


def build_graph():
    """Wire and compile the turn graph.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph ready for ``invoke`` or ``stream``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_guard", tool_guard_node)
    graph.add_node("tools", tools_node)
    graph.add_node("exposition", exposition_node)
    graph.add_node("output_verify", output_verify_node)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "input_guard")
    graph.add_conditional_edges("input_guard", route_after_input, {"agent": "agent", "end": END})
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tool_guard": "tool_guard", "exposition": "exposition"},
    )
    graph.add_edge("tool_guard", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("exposition", "output_verify")
    graph.add_edge("output_verify", END)

    return graph.compile()


def get_graph():
    """Return the process-wide compiled graph, building it on first call.

    Returns
    -------
    CompiledStateGraph
        Singleton compiled graph instance.
    """
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED
