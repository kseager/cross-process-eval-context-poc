"""Agent service -- a **separate process** that hosts the agent over HTTP.

This mirrors the ACA topology (Yingying's design): the agent is a **remote
service**, not an in-process object. The runner authors the ``invoke_agent``
span and injects a W3C ``traceparent`` header pointing at it; this service is a
**passive header recipient** -- it rebuilds that remote parent and opens its own
``execute_agent`` span *underneath* ``invoke_agent``, then runs the model.

Crucially, this service does **not** emit an ``invoke_agent`` span -- that span
belongs to the runner. It emits ``execute_agent`` (and the framework's own
``chat`` spans nest under it), exactly like ACA's::

    [invoke_agent]        (runner)
    ↳ [execute_agent]     (this service, via traceparent header)
        ↳ [chat ...]      (agent framework)

The agent business logic itself needs **zero** tracing code; the ``execute_agent``
wrapper here stands in for the remote agent runtime's own instrumentation.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from opentelemetry import trace
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from .agent import build_agent
from opentelemetry import context as otel_context

from .telemetry import setup_observability
from .trace_context import parent_context_from_traceparent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-service")

SERVICE_NAME = "agent-service"

app = FastAPI(title="agent-service")

_tracer: Tracer | None = None
_agent = None


class InvokeRequest(BaseModel):
    """Payload the runner sends to invoke the agent."""

    item_id: str
    query: str


class InvokeResponse(BaseModel):
    """The agent's answer plus the trace it ran under (for correlation)."""

    item_id: str
    response: str
    agent_trace_id: str


@app.on_event("startup")
def _startup() -> None:
    global _tracer, _agent
    load_dotenv()
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
    setup_observability()
    _tracer = trace.get_tracer(SERVICE_NAME)
    _agent = build_agent()
    logger.info("agent-service observability + agent configured")


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(
    req: InvokeRequest,
    traceparent: str | None = Header(default=None),
) -> InvokeResponse:
    """Run the agent so its framework ``invoke_agent`` span nests under the driver.

    The driver passes ``traceparent`` (pointing at its ``invoke_agent`` span) as
    an HTTP header. We attach that rebuilt remote parent as the active context
    and simply run the agent -- Agent Framework's *own* instrumentation emits
    the child ``invoke_agent <agent_name>`` span beneath it. This mirrors ACA
    exactly (``target_util.invoke_agent_with_tracing``): the remote agent is a
    plain framework app and needs **no** span code of its own; nested
    ``invoke_agent`` spans are expected and reconciled downstream by the trace
    consumer's parent-chain dedup.
    """
    assert _tracer is not None and _agent is not None, "service not initialized"

    # Rebuild the driver's invoke_agent span as a remote parent from the header
    # and make it the active context, so the framework's auto agent-span parents
    # to it. No manual span is created here.
    parent_ctx = parent_context_from_traceparent(traceparent) if traceparent else None
    token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
    try:
        result = await _agent.run(req.query)
    finally:
        if token is not None:
            otel_context.detach(token)
    response_text = str(result)

    # The framework's invoke_agent <agent_name> span is created *and finished*
    # inside `_agent.run()`, so its span_id is not observable from here. What we
    # can report -- and all the driver needs for its correlation check -- is the
    # trace it ran under, which is the remote parent's (== the driver's
    # invoke_agent) trace_id.
    parent_span_ctx = trace.get_current_span(parent_ctx).get_span_context()
    agent_trace_id = (
        f"{parent_span_ctx.trace_id:032x}"
        if parent_span_ctx.is_valid
        else "0" * 32
    )
    logger.info(
        "framework invoke_agent span ran under traceparent=%s (trace=%s) for item=%s",
        traceparent,
        agent_trace_id,
        req.item_id,
    )

    return InvokeResponse(
        item_id=req.item_id,
        response=response_text,
        agent_trace_id=agent_trace_id,
    )
