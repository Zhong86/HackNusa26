"""
Node functions for the Sentinel Loop graph.

Each node takes the current GraphState and returns a partial dict to
merge in — standard LangGraph convention. Keeping nodes as plain
functions (no framework-specific decorators beyond what's needed)
makes them easy to unit test in isolation.
"""

from __future__ import annotations

from langgraph.types import interrupt

from .state import GraphState
from mock_classifier import score_email
from schemas import ContextBundle, ReasoningResult, Verdict
from tools.context_tools import lookup_domain_age, lookup_sender_history, lookup_threat_intel

# Below this, Layer 1 is confident enough to decide alone.
# Between the two thresholds is the "uncertain zone" that escalates to Layer 2.
LOW_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.75

# Below this, Layer 2's own reasoning confidence is too low to auto-decide,
# so it escalates to a human via interrupt().
HUMAN_REVIEW_THRESHOLD = 0.6


def score_node(state: GraphState) -> dict:
    """Layer 1: run the (mocked) classifier."""
    result = score_email(state["email"])
    trace = state.get("trace", [])
    return {
        "layer1_score": result,
        "trace": trace + ["score_node"],
    }


def route_after_score(state: GraphState) -> str:
    """Conditional edge: decide whether Layer 1's score is confident enough."""
    score = state["layer1_score"].score
    if LOW_THRESHOLD <= score <= HIGH_THRESHOLD:
        return "uncertain"
    return "confident"


def direct_decision_node(state: GraphState) -> dict:
    """Layer 1 was confident — decide without invoking Layer 2 at all."""
    score = state["layer1_score"].score
    verdict = Verdict.QUARANTINE if score > HIGH_THRESHOLD else Verdict.ALLOW
    justification = (
        f"Layer 1 classifier confidence ({score:.2f}) outside the uncertain zone "
        f"[{LOW_THRESHOLD}, {HIGH_THRESHOLD}] — decided without escalating to Layer 2."
    )
    trace = state.get("trace", [])
    return {
        "final_verdict": verdict,
        "final_justification": justification,
        "trace": trace + ["direct_decision_node"],
    }


def gather_context_node(state: GraphState) -> dict:
    """Layer 2: pull sender history, domain age, and threat-intel stubs."""
    email = state["email"]
    context = ContextBundle(
        sender_history=lookup_sender_history(email),
        domain_age=lookup_domain_age(email),
        threat_intel=lookup_threat_intel(email),
    )
    trace = state.get("trace", [])
    return {
        "context": context,
        "trace": trace + ["gather_context_node"],
    }


def reason_node(state: GraphState) -> dict:
    """
    Layer 2: reasoning agent. Placeholder rule-based logic for now —
    swap the body for an LLM call (structured output = ReasoningResult)
    once you're ready to wire in real reasoning.
    """
    email = state["email"]
    ctx = state["context"]
    score = state["layer1_score"].score

    evidence: list[str] = []
    risk_points = 0.0

    if ctx.threat_intel.get("domain_flagged"):
        risk_points += 0.4
        evidence.append("Sender domain matched known threat-intel feed")
    if ctx.threat_intel.get("matched_malicious_urls"):
        risk_points += 0.25
        evidence.append("One or more URLs matched known-malicious list")
    if ctx.domain_age.get("newly_registered"):
        risk_points += 0.2
        evidence.append("Sending domain registered under 30 days ago")
    if not ctx.sender_history.get("seen_before"):
        risk_points += 0.1
        evidence.append("No prior history for this sender")
    elif ctx.sender_history.get("prior_flag_count", 0) > 0:
        risk_points += 0.15
        evidence.append(f"Sender has {ctx.sender_history['prior_flag_count']} prior flags")

    # blend with the Layer 1 score that put us here in the first place
    combined = min(1.0, (risk_points * 0.7) + (score * 0.3))

    if combined > 0.6:
        decision = Verdict.QUARANTINE
    elif combined > 0.35:
        decision = Verdict.ESCALATE
    else:
        decision = Verdict.ALLOW

    # crude confidence proxy: how far from the decision boundary we landed
    confidence = round(min(1.0, abs(combined - 0.475) / 0.475 + 0.3), 2)

    result = ReasoningResult(
        decision=decision,
        confidence=confidence,
        justification=(
            f"Layer 1 flagged this as borderline (score={score:.2f}). "
            f"Layer 2 context review found: {'; '.join(evidence) if evidence else 'no additional risk signals'}. "
            f"Combined risk={combined:.2f} -> {decision.value}."
        ),
        evidence_used=evidence,
        mitre_technique_ids=["T1566.001"] if ctx.threat_intel.get("domain_flagged") else ["T1566"],
    )

    trace = state.get("trace", [])
    return {
        "reasoning": result,
        "trace": trace + ["reason_node"],
    }


def route_after_reason(state: GraphState) -> str:
    """Conditional edge: is Layer 2's own confidence high enough to auto-decide?"""
    if state["reasoning"].confidence < HUMAN_REVIEW_THRESHOLD:
        return "needs_human"
    return "auto_decide"


def human_review_node(state: GraphState) -> dict:
    """
    Pause the graph and wait for a human analyst's call.
    interrupt() surfaces the payload to whatever is driving the graph
    (API layer / CLI / future dashboard) and blocks until resumed
    with a Command(resume=...).
    """
    reasoning = state["reasoning"]
    payload = {
        "reason": "low_confidence_reasoning",
        "email_subject": state["email"].subject,
        "sender": state["email"].sender,
        "layer1_score": state["layer1_score"].score,
        "agent_decision": reasoning.decision.value,
        "agent_confidence": reasoning.confidence,
        "agent_justification": reasoning.justification,
        "evidence_used": reasoning.evidence_used,
    }
    human_response = interrupt(payload)

    # human_response is expected to look like: {"decision": "quarantine", "note": "..."}
    trace = state.get("trace", [])
    return {
        "needs_human_review": True,
        "human_decision": Verdict(human_response["decision"]),
        "human_note": human_response.get("note"),
        "trace": trace + ["human_review_node"],
    }


def finalize_after_human_node(state: GraphState) -> dict:
    """Apply the human's decision as the final verdict."""
    trace = state.get("trace", [])
    return {
        "final_verdict": state["human_decision"],
        "final_justification": (
            f"Human analyst override: {state.get('human_note') or 'no note provided'}. "
            f"(Agent had proposed {state['reasoning'].decision.value} "
            f"at confidence {state['reasoning'].confidence:.2f}.)"
        ),
        "trace": trace + ["finalize_after_human_node"],
    }


def auto_decide_node(state: GraphState) -> dict:
    """Layer 2 was confident enough — its decision stands without human review."""
    reasoning = state["reasoning"]
    trace = state.get("trace", [])
    return {
        "needs_human_review": False,
        "final_verdict": reasoning.decision,
        "final_justification": reasoning.justification,
        "trace": trace + ["auto_decide_node"],
    }