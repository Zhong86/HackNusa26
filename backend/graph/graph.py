"""
Assembles the Sentinel Loop Layer 2 graph.

    START
      |
   score_node
      |
   route_after_score
      |-- confident --> direct_decision_node --> END
      |-- uncertain --> gather_context_node --> reason_node
                                                    |
                                              route_after_reason
                                                    |-- auto_decide --> auto_decide_node --> END
                                                    |-- needs_human --> human_review_node
                                                                            |
                                                                    finalize_after_human_node --> END

A checkpointer is required for interrupt()/resume to work — MemorySaver
is fine for local dev; swap for a persistent checkpointer (e.g. Postgres)
before this needs to survive process restarts.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    auto_decide_node,
    direct_decision_node,
    finalize_after_human_node,
    gather_context_node,
    human_review_node,
    reason_node,
    route_after_reason,
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
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize_after_human", finalize_after_human_node)

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

    graph.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "auto_decide": "auto_decide",
            "needs_human": "human_review",
        },
    )
    graph.add_edge("auto_decide", END)

    graph.add_edge("human_review", "finalize_after_human")
    graph.add_edge("finalize_after_human", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Singleton compiled graph, imported by the API layer.
sentinel_graph = build_graph()