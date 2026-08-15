"""
Graph state for the Sentinel Loop Layer 2 reasoning agent.

Flow:
    score_email (mocked Layer 1)
        -> uncertain? (conditional edge)
            -> No:  direct_decision -> END
            -> Yes: gather_context -> reason -> auto_decide -> END

No human-in-the-loop: this is a detection research system (not deployed in
front of real users), so Layer 2's reasoning always auto-decides.

Every node returns a partial dict that gets merged into this TypedDict,
which is the standard LangGraph state-update pattern.
"""

from __future__ import annotations

from typing import TypedDict

from schemas import ContextBundle, EmailPayload, Layer1Score, ReasoningResult, Verdict


class GraphState(TypedDict, total=False):
    # --- input ---
    email: EmailPayload

    # --- Layer 1 (mocked score_email) ---
    layer1_score: Layer1Score

    # --- routing ---
    is_uncertain: bool

    # --- Layer 2 context gathering ---
    context: ContextBundle

    # --- Layer 2 reasoning ---
    reasoning: ReasoningResult

    # --- final output ---
    final_verdict: Verdict
    final_justification: str
    trace: list[str]  # ordered log of node names visited, for the dashboard later