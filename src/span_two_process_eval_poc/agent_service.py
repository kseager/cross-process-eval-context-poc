"""Agent service -- a **separate process** that hosts the agent over HTTP.

This mirrors the ACA topology (Yingying's design): the agent is a **remote
service**, not an in-process object. The runner authors the
``evaluation_context`` span and injects a W3C ``traceparent`` header pointing at
it; this service is a **passive header recipient** -- it rebuilds that remote
parent and opens its own ``execute_agent`` span *underneath*
``evaluation_context``, then runs the model.

Crucially, this service does **not** author the wrapper span -- that span
belongs to the runner. It emits ``execute_agent`` (and the framework's own
``chat`` spans nest under it), exactly like ACA's::

    [evaluation_context]  (runner)
    ↳ [execute_agent]     (this service, via traceparent header)
        ↳ [chat ...]      (agent framework)

The agent business logic itself needs **zero** tracing code; the ``execute_agent``
wrapper here stands in for the remote agent runtime's own instrumentation.
"""

from __future__ import annotations

import contextvars
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from agent_framework._agents import RawAgent

from .agent import build_agent

from .telemetry import setup_observability
from .trace_context import (
    EXECUTE_AGENT_SPAN_NAME,
    parent_context_from_traceparent,
)

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


class InvokeResponse(BaseModel):
    """The agent's answer plus the execute_agent span ids (for correlation)."""

    item_id: str
    response: str
    execute_agent_trace_id: str
    execute_agent_span_id: str


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


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(
    req: InvokeRequest,
    traceparent: str | None = Header(default=None),
) -> InvokeResponse:
    """Run the agent under an ``execute_agent`` span parented to the driver's
    ``evaluation_context`` span.

    The driver passes ``traceparent`` (pointing at its ``evaluation_context``
    span) as an HTTP header. We rebuild that remote parent and open an
    ``execute_agent`` span beneath it -- matching ACA's hierarchy
    (``evaluation_context -> execute_agent -> chat``), where the remote
    hosted-agent runtime authors ``execute_agent``.

    Because our agent runs *locally* (not via Foundry's hosted runtime), we run
    it through ``RawAgent.run`` to bypass Agent Framework's ``AgentTelemetryLayer``
    -- otherwise the framework would emit its own ``invoke_agent <agent_name>``
    span, which a real hosted agent never produces. The chat client still emits
    its ``chat`` span, so ``chat`` nests directly under our ``execute_agent`` --
    exactly the tree a hosted-agent customer sees. The agent code itself is
    unchanged; only *how the service invokes it* differs.
    """
    assert _tracer is not None and _agent is not None, "service not initialized"

    # Rebuild the driver's evaluation_context span as a remote parent.
    parent_ctx = parent_context_from_traceparent(traceparent) if traceparent else None

    with _tracer.start_as_current_span(
        EXECUTE_AGENT_SPAN_NAME, context=parent_ctx
    ) as span:
        # Bypass AgentTelemetryLayer so no invoke_agent <agent_name> span is
        # emitted; chat nests directly under execute_agent (ACA-faithful).
        result = await RawAgent.run(_agent, req.query)
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
