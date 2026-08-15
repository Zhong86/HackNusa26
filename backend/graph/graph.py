"""
Assembles the Sentinel Loop Layer 2 graph.

    START
      |
   score_node
      |
   route_after_score
      |-- confident --> direct_decision_node --> END
      |-- uncertain --> gather_context_node --> reason_node --> auto_decide_node --> END

No human-in-the-loop: this is a detection research system (not deployed in
front of real users), so Layer 2's reasoning always auto-decides — we just
want to observe what the system outputs.

A checkpointer is kept around for future use / trace inspection via
sentinel_graph.get_state(); MemorySaver is fine for local dev, swap for a
persistent checkpointer (e.g. Postgres) before this needs to survive
process restarts. Strict msgpack deserialization is enabled (see
CVE-2026-28277) — required regardless of backing store, since it protects
against a compromised/tampered checkpoint store, not just remote ones.
"""

from __future__ import annotations

import os

# Opt into strict msgpack checkpoint deserialization (see CVE-2026-28277).
# With this set, LangGraph derives the allowlist itself from GraphState's
# schema at compile time (covers EmailPayload, Layer1Score, ContextBundle,
# ReasoningResult, Verdict) instead of silently allowing any type with a
# deprecation warning. Set before importing the checkpoint module so it
# takes effect. Respect an existing value if the process already set one.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    auto_decide_node,
    direct_decision_node,
    gather_context_node,
    reason_node,
    route_after_score,
    score_node,
)
from .state import GraphState


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("score", score_node)
    graph.add_node("direct_decision", direct_decision_node)
    graph.add_node("gather_context", gather_context_node)
    graph.add_node("reason", reason_node)
    graph.add_node("auto_decide", auto_decide_node)

    graph.add_edge(START, "score")

    graph.add_conditional_edges(
        "score",
        route_after_score,
        {
            "confident": "direct_decision",
            "uncertain": "gather_context",
        },
    )
    graph.add_edge("direct_decision", END)

    graph.add_edge("gather_context", "reason")
    graph.add_edge("reason", "auto_decide")
    graph.add_edge("auto_decide", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Singleton compiled graph, imported by the API layer.
sentinel_graph = build_graph()