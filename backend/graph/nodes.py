"""
Node functions for the Sentinel Loop graph.

Each node takes the current GraphState and returns a partial dict to
merge in — standard LangGraph convention. Keeping nodes as plain
functions (no framework-specific decorators beyond what's needed)
makes them easy to unit test in isolation.

No human-in-the-loop: this is a detection research system, not something
sitting in front of real users' inboxes, so Layer 2's reasoning always
auto-decides. There's nothing to interrupt for — we just want to see what
the system outputs.
"""

from __future__ import annotations

from .state import GraphState
from ml.serving.classifier import score_email
from schemas import ContextBundle, ReasoningResult, Verdict
from tools.context_tools import lookup_domain_age, lookup_sender_history, lookup_threat_intel

# Below this, Layer 1 is confident enough to decide alone.
# Between the two thresholds is the "uncertain zone" that escalates to Layer 2.
LOW_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.75
from logger import get_logger
log = get_logger(__name__)

def score_node(state: GraphState) -> dict:
    """Layer 1: run the (mocked) classifier."""
    result = score_email(state["email"])
    trace = state.get("trace", [])

    log.info("Layer 1 score_node: email=%s score=%.2f", state["email"].sender, result.score)
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


def auto_decide_node(state: GraphState) -> dict:
    """Layer 2's decision is final — no human-in-the-loop, just report the output."""
    reasoning = state["reasoning"]
    trace = state.get("trace", [])
    return {
        "final_verdict": reasoning.decision,
        "final_justification": reasoning.justification,
        "trace": trace + ["auto_decide_node"],
    }