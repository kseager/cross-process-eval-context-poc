"""Agent service -- a **separate process** that hosts the agent over HTTP.

The agent runs as a pure **black box**: this service invokes it natively with
**no** incoming tracing context, so Agent Framework's own ``AgentTelemetryLayer``
emits the agent's natural ``invoke_agent <name>`` span. A passive
``SpanProcessor`` captures that span's ``(trace_id, span_id)`` and the
``/invoke-standalone`` endpoint returns them, so the driver can attach an
``evaluation_context`` child span to the agent's real trace **afterward**::

    [invoke_agent <name>]   (agent, authored natively)
    ↳ [chat ...]            (agent framework)
    ↳ [evaluation_context]  (driver, attached cross-process via returned ids)

The agent business logic needs **zero** tracing code, and this service does not
wrap the run in a span of its own -- the agent's real ``invoke_agent`` span *is*
the attach target.
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

# Per-request holder for the id of the framework's own invoke_agent span. The
# standalone flow runs the agent natively, so Agent Framework's
# AgentTelemetryLayer emits an "invoke_agent <name>" span. We capture THAT span
# (the agent's real root) so the driver can attach evaluation_context directly to
# it -- no redundant wrapper span of our own.
_invoke_agent_ctx: contextvars.ContextVar[tuple[int, int] | None] = (
    contextvars.ContextVar("_invoke_agent_ctx", default=None)
)


class _InvokeAgentSpanCapture(SpanProcessor):
    """Records the framework's ``invoke_agent`` span id for the active request.

    Uses ``on_start`` (not ``on_end``) so the id is available while the request
    coroutine is still running, and writes it into the request-scoped
    ``_invoke_agent_ctx`` ContextVar. Only the first invoke_agent span per
    request is captured (the agent's root invocation).
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
    """Payload the runner sends to invoke the agent."""

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

    # Register the capture processor on the SDK tracer provider so we can learn
    # the framework's invoke_agent span id during a standalone request.
    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(_InvokeAgentSpanCapture())
    logger.info("agent-service observability + agent configured")


class StandaloneResponse(BaseModel):
    """Agent answer plus the ids of the span the agent authored on its own.

    The agent runs with **no** incoming ``traceparent``: it authors its own
    ``invoke_agent`` span and reports that span's identity back so the driver can
    attach an ``evaluation_context`` child span to it afterward.
    """

    item_id: str
    response: str
    agent_trace_id: str
    agent_span_id: str


@app.post("/invoke-standalone", response_model=StandaloneResponse)
async def invoke_standalone(req: InvokeRequest) -> StandaloneResponse:
    """Run the agent as a black box and return the id of the span it authored.

    The request path has **zero** tracing coupling -- no ``traceparent`` header,
    no context injection, no ``RawAgent`` bypass. The agent runs exactly as it
    normally would and Agent Framework's own ``AgentTelemetryLayer`` emits its
    natural ``invoke_agent <name>`` span. We capture **that** span's
    ``(trace_id, span_id)`` (via ``_InvokeAgentSpanCapture``) and return them; the
    driver then opens an ``evaluation_context`` span as a *child* of that returned
    span (same trace), stamping ground truth on it.

    We deliberately do **not** wrap the run in an extra span of our own -- the
    agent's real ``invoke_agent`` span *is* the attach target.
    """
    assert _tracer is not None and _agent is not None, "service not initialized"

    # Reset the per-request capture slot, run the agent natively, then read back
    # the framework's invoke_agent span id that the processor recorded.
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
