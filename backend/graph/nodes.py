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

from db import record_sender_event
from .state import GraphState
from ml.serving.classifier import score_email
from schemas import ContextBundle, ReasoningResult, Verdict
from tools.context_tools import lookup_sender_history
from llm.client import call_structured, LLMCallError
from pydantic import ValidationError

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
    record_sender_event(state["email"].sender, verdict.value)
    return {
        "final_verdict": verdict,
        "final_justification": justification,
        "trace": trace + ["direct_decision_node"],
    }


def gather_context_node(state: GraphState) -> dict:
    """Layer 2: pull sender history (real, from local DB — offline)."""
    email = state["email"]
    context = ContextBundle(
        sender_history=lookup_sender_history(email),
    )
    trace = state.get("trace", [])
    return {
        "context": context,
        "trace": trace + ["gather_context_node"],
    }


REASON_SYSTEM_PROMPT = (
    "Kamu adalah analis SOC (Security Operations Center) yang menilai apakah "
    "sebuah email adalah phishing, berdasarkan skor Layer 1 dan riwayat sender. "
    "Kamu TIDAK punya data umur domain atau threat-intel eksternal — nilai hanya "
    "dari isi email, skor Layer 1, dan riwayat sender ini. "
    "decision harus salah satu dari: allow, quarantine, escalate. "
    "confidence adalah angka 0.0-1.0. mitre_technique_ids diisi dengan ID teknik MITRE ATT&CK yang relevan."
)

REASON_SCHEMA = {
    "decision": "allow | quarantine | escalate",
    "confidence": "float 0.0-1.0",
    "justification": "string, alasan singkat seperti catatan tiket SOC",
    "evidence_used": ["string"],
    "mitre_technique_ids": ["string"],
}


def reason_node(state: GraphState) -> dict:
    """Layer 2: reasoning agent — memanggil LLM (OpenAI-compatible) untuk keputusan akhir."""
    email = state["email"]
    ctx = state["context"]
    score = state["layer1_score"].score

    user_prompt = (
        f"Layer 1 score (probabilitas phishing): {score:.2f}\n\n"
        f"Email:\n"
        f"- Sender: {email.sender}\n"
        f"- Display name: {email.display_name}\n"
        f"- Subject: {email.subject}\n"
        f"- Body: {email.body}\n"
        f"- URLs: {email.urls}\n\n"
        f"Konteks tambahan:\n"
        f"- Sender history: {ctx.sender_history}\n"
    )

    try:
        raw = call_structured(REASON_SYSTEM_PROMPT, user_prompt, REASON_SCHEMA)
        result = ReasoningResult.model_validate(raw)
    except (LLMCallError, ValidationError) as e:
        log.warning("Layer 2 reason_node: LLM gagal (%s), fallback ke escalate manual", e)
        result = ReasoningResult(
            decision=Verdict.ESCALATE,
            confidence=0.0,
            justification=f"LLM reasoning gagal ({e}); email di-escalate otomatis untuk review manual.",
            evidence_used=["LLM call failed — fallback safety net"],
            mitre_technique_ids=["T1566"],
        )

    log.info("Layer 2 reason_node: decision=%s confidence=%.2f", result.decision, result.confidence)

    trace = state.get("trace", [])
    return {
        "reasoning": result,
        "trace": trace + ["reason_node"],
    }


def auto_decide_node(state: GraphState) -> dict:
    """Layer 2's decision is final — no human-in-the-loop, just report the output."""
    reasoning = state["reasoning"]
    trace = state.get("trace", [])
    record_sender_event(state["email"].sender, reasoning.decision.value)
    return {
        "final_verdict": reasoning.decision,
        "final_justification": reasoning.justification,
        "trace": trace + ["auto_decide_node"],
    }