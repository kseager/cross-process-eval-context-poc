"""Agent service: a separate process that hosts the agent over HTTP.

The agent runs as a black box: this service invokes it natively with no incoming
tracing context, so Agent Framework emits the agent's ``invoke_agent`` span. A
``SpanProcessor`` captures that span's ``(trace_id, span_id)`` and
``/invoke-standalone`` returns them, so a separate evaluation process can attach
an ``evaluation_context`` child span to the agent's trace afterward.

Contract with the evaluation driver: every successful ``/invoke-standalone``
response includes ``agent_trace_id`` (32 hex chars) and ``agent_span_id`` (16
hex chars), identifying the span the driver parents ``evaluation_context`` to.
"""

from __future__ import annotations

import contextvars
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from .agent import build_agent

from .telemetry import setup_observability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-service")

_invoke_agent_ctx: contextvars.ContextVar[tuple[int, int] | None] = (
    contextvars.ContextVar("_invoke_agent_ctx", default=None)
)


class _InvokeAgentSpanCapture(SpanProcessor):
    """Records the framework's ``invoke_agent`` span id for the active request.

    Uses ``on_start`` so the id is available while the request coroutine is still
    running. Only the first ``invoke_agent`` span per request is captured.
    """

    def on_start(self, span, parent_context=None) -> None:  # noqa: D401
        if span.name.startswith("invoke_agent") and _invoke_agent_ctx.get() is None:
            ctx = span.get_span_context()
            _invoke_agent_ctx.set((ctx.trace_id, ctx.span_id))

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D401
        return

    def shutdown(self) -> None:  # noqa: D401
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: D401
        return True

SERVICE_NAME = "agent-service"

app = FastAPI(title="agent-service")

_tracer: Tracer | None = None
_agent = None


class InvokeRequest(BaseModel):
    """Payload the caller sends to invoke the agent."""

    item_id: str
    query: str


@app.on_event("startup")
def _startup() -> None:
    global _tracer, _agent
    load_dotenv()
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
    setup_observability()
    _tracer = trace.get_tracer(SERVICE_NAME)
    _agent = build_agent()

    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(_InvokeAgentSpanCapture())
    logger.info("agent-service observability + agent configured")


class StandaloneResponse(BaseModel):
    """Agent answer plus the ids of the span the agent authored.

    ``agent_trace_id`` / ``agent_span_id`` identify the ``invoke_agent`` span the
    driver parents its ``evaluation_context`` span to. Both are lowercase hex.
    """

    item_id: str
    response: str
    agent_trace_id: str
    agent_span_id: str


@app.post("/invoke-standalone", response_model=StandaloneResponse)
async def invoke_standalone(req: InvokeRequest) -> StandaloneResponse:
    """Run the agent as a black box and return the id of the span it authored.

    The agent runs natively, so Agent Framework emits its ``invoke_agent`` span;
    ``_InvokeAgentSpanCapture`` records that span's ``(trace_id, span_id)`` and
    we return them so the evaluation driver can open its ``evaluation_context``
    span as a child of it and stamp ground truth there.
    """
    assert _tracer is not None and _agent is not None, "service not initialized"

    token = _invoke_agent_ctx.set(None)
    try:
        result = await _agent.run(req.query)
        response_text = str(result)
        captured = _invoke_agent_ctx.get()
    finally:
        _invoke_agent_ctx.reset(token)

    if captured is None:
        raise RuntimeError(
            "Did not capture an invoke_agent span for the agent run; cannot "
            "return an attach target."
        )
    trace_id, span_id = captured
    logger.info(
        "standalone invoke_agent span (trace=%032x span=%016x) for item=%s",
        trace_id,
        span_id,
        req.item_id,
    )

    return StandaloneResponse(
        item_id=req.item_id,
        response=response_text,
        agent_trace_id=f"{trace_id:032x}",
        agent_span_id=f"{span_id:016x}",
    )
