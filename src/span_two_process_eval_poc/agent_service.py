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

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from .agent import build_agent
from .telemetry import setup_observability
from .trace_context import parent_context_from_traceparent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-service")

SERVICE_NAME = "agent-service"
EXECUTE_AGENT_SPAN_NAME = "execute_agent"

app = FastAPI(title="agent-service")

_tracer: Tracer | None = None
_agent = None


class InvokeRequest(BaseModel):
    """Payload the runner sends to invoke the agent."""

    item_id: str
    query: str


class InvokeResponse(BaseModel):
    """The agent's answer plus the execute_agent span ids (for logging)."""

    item_id: str
    response: str
    execute_agent_trace_id: str
    execute_agent_span_id: str


@app.on_event("startup")
def _startup() -> None:
    global _tracer, _agent
    load_dotenv()
    _tracer = setup_observability(SERVICE_NAME)
    _agent = build_agent()
    logger.info("agent-service observability + agent configured")


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(
    req: InvokeRequest,
    traceparent: str | None = Header(default=None),
) -> InvokeResponse:
    """Run the agent under an ``execute_agent`` span parented to invoke_agent.

    The runner passes ``traceparent`` (pointing at its ``invoke_agent`` span) as
    an HTTP header. We rebuild that remote parent and open ``execute_agent``
    beneath it, so this service's spans nest under the runner's invoke_agent --
    matching the ACA hierarchy.
    """
    assert _tracer is not None and _agent is not None, "service not initialized"

    # Rebuild the runner's invoke_agent span as a remote parent from the header.
    parent_ctx = parent_context_from_traceparent(traceparent) if traceparent else None

    with _tracer.start_as_current_span(
        EXECUTE_AGENT_SPAN_NAME, context=parent_ctx
    ) as span:
        result = await _agent.run(req.query)
        response_text = str(result)

        ctx = span.get_span_context()
        logger.info(
            "execute_agent span (trace=%032x span=%016x) parented to "
            "traceparent=%s for item=%s",
            ctx.trace_id,
            ctx.span_id,
            traceparent,
            req.item_id,
        )

        return InvokeResponse(
            item_id=req.item_id,
            response=response_text,
            execute_agent_trace_id=f"{ctx.trace_id:032x}",
            execute_agent_span_id=f"{ctx.span_id:016x}",
        )
