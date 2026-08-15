"""
API surface for the Sentinel Loop Layer 2 graph.

POST /emails            -> run an email through the graph, returns either
                            a final verdict or a pending human-review interrupt
POST /emails/{id}/resume -> resume a paused (interrupted) run with a human decision
GET  /emails/{id}        -> fetch current state / trace for a given run
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from graph.graph import sentinel_graph
from schemas import EmailPayload, Verdict

router = APIRouter()


class ResumeRequest(BaseModel):
    decision: Verdict
    note: str | None = None


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _serialize_state(thread_id: str) -> dict:
    """Read back the current state for a thread, including interrupt info if paused."""
    snapshot = sentinel_graph.get_state(_thread_config(thread_id))
    state = dict(snapshot.values)

    # pydantic models in state need explicit serialization
    for key in ("layer1_score", "context", "reasoning"):
        if key in state and state[key] is not None:
            state[key] = state[key].model_dump()

    response = {
        "thread_id": thread_id,
        "trace": state.get("trace", []),
        "status": "paused" if snapshot.next else "completed",
        "state": state,
    }

    if snapshot.next:
        # graph is paused at an interrupt — surface the interrupt payload
        interrupts = getattr(snapshot, "interrupts", None) or snapshot.tasks
        pending = []
        for task in snapshot.tasks:
            for i in getattr(task, "interrupts", []):
                pending.append(i.value)
        response["pending_review"] = pending[0] if pending else None

    return response


@router.post("/emails")
def submit_email(email: EmailPayload):
    """Run a new email through the graph. May return a final verdict or pause for human review."""
    thread_id = str(uuid.uuid4())
    sentinel_graph.invoke({"email": email, "trace": []}, config=_thread_config(thread_id))
    return _serialize_state(thread_id)


@router.get("/emails/{thread_id}")
def get_email_run(thread_id: str):
    """Fetch the current state of a run by thread id."""
    snapshot = sentinel_graph.get_state(_thread_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return _serialize_state(thread_id)


@router.post("/emails/{thread_id}/resume")
def resume_email_run(thread_id: str, body: ResumeRequest):
    """Resume a paused run with a human analyst's decision."""
    snapshot = sentinel_graph.get_state(_thread_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="This run is not paused / has no pending review")

    sentinel_graph.invoke(
        Command(resume={"decision": body.decision.value, "note": body.note}),
        config=_thread_config(thread_id),
    )
    return _serialize_state(thread_id)