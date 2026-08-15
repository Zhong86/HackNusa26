"""
Graph state for the Sentinel Loop Layer 2 reasoning agent.

Flow:
    score_email (mocked Layer 1)
        -> uncertain? (conditional edge)
            -> No:  route_direct   -> END
            -> Yes: gather_context -> reason -> confidence_check
                -> low:  interrupt (human-in-the-loop) -> apply_human_decision -> END
                -> high: decide -> END

Every node returns a partial dict that gets merged into this TypedDict,
which is the standard LangGraph state-update pattern.
"""

from __future__ import annotations

from typing import Optional, TypedDict

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

    # --- human-in-the-loop ---
    needs_human_review: bool
    human_decision: Optional[Verdict]
    human_note: Optional[str]

    # --- final output ---
    final_verdict: Verdict
    final_justification: str
    trace: list[str]  # ordered log of node names visited, for the dashboard later