"""
API surface for the Sentinel Loop Layer 2 graph.

POST /emails         -> run an email through the graph (blocking), returns
                         the final verdict + trace.
POST /emails/stream   -> same input, but streams node-by-node progress over
                         SSE as the graph executes, ending in a 'done' event.
GET  /emails/{id}     -> fetch current state / trace for a given run.

No human-in-the-loop: this is a detection research system (not deployed in
front of real users), so every run goes straight through to a final verdict.
There's nothing to pause for or resume.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph.graph import sentinel_graph
from schemas import EmailPayload

logger = logging.getLogger(__name__)

router = APIRouter()

# Guards against two concurrent runs stomping on the same checkpointed thread.
# A message/run for a thread that's still in flight is rejected, not queued.
_running_threads: set[str] = set()


class EmailRunRequest(BaseModel):
    """Body for POST /emails and /emails/stream: the email to run through the graph."""
    email: EmailPayload


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _serialize_state(thread_id: str) -> dict:
    """Read back the current state for a thread."""
    snapshot = sentinel_graph.get_state(_thread_config(thread_id))
    state = dict(snapshot.values)

    # pydantic models in state need explicit serialization
    for key in ("layer1_score", "context", "reasoning"):
        if key in state and state[key] is not None:
            state[key] = state[key].model_dump()

    return {
        "thread_id": thread_id,
        "trace": state.get("trace", []),
        "status": "completed",
        "state": state,
    }


@router.post("/emails")
def submit_email(body: EmailRunRequest):
    """Run a new email through the graph (blocking) and return the final verdict."""
    thread_id = str(uuid.uuid4())

    _running_threads.add(thread_id)
    try:
        sentinel_graph.invoke({"email": body.email, "trace": []}, config=_thread_config(thread_id))
    finally:
        _running_threads.discard(thread_id)

    return _serialize_state(thread_id)


@router.post("/emails/stream")
async def submit_email_stream(body: EmailRunRequest):
    """Run a new email through the graph, streamed as SSE — one event per node as it completes."""
    thread_id = str(uuid.uuid4())

    if thread_id in _running_threads:
        async def busy():
            yield f"data: {json.dumps({'type': 'error', 'message': 'A run is already in progress for this thread_id'})}\n\n"
        return StreamingResponse(busy(), media_type="text/event-stream")

    graph_input = {"email": body.email, "trace": []}

    async def event_generator():
        _running_threads.add(thread_id)
        try:
            yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"

            # sentinel_graph.stream(...) is sync; run it in a worker thread and
            # drain it via a queue so we don't block the event loop.
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()
            SENTINEL = object()

            def _run_graph():
                try:
                    for update in sentinel_graph.stream(
                        graph_input,
                        config=_thread_config(thread_id),
                        stream_mode="updates",
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, update)
                except Exception as e:  # surface graph-execution errors to the stream
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

            asyncio.get_event_loop().run_in_executor(None, _run_graph)

            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item

                # item looks like {"node_name": {partial_state_dict}}
                for node_name, partial in item.items():
                    safe_partial = {}
                    for key, value in partial.items():
                        if hasattr(value, "model_dump"):
                            safe_partial[key] = value.model_dump()
                        else:
                            safe_partial[key] = value
                    yield f"data: {json.dumps({'type': 'node', 'node': node_name, 'update': safe_partial}, default=str)}\n\n"

            final_state = _serialize_state(thread_id)
            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id, 'state': final_state})}\n\n"

        except Exception:
            logger.exception("Email stream error thread=%s", thread_id)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Terjadi kesalahan saat memproses email ini.'})}\n\n"
        finally:
            _running_threads.discard(thread_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/emails/{thread_id}")
def get_email_run(thread_id: str):
    """Fetch the current state of a run by thread id."""
    snapshot = sentinel_graph.get_state(_thread_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return _serialize_state(thread_id)