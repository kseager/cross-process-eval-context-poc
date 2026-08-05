"""Capture the framework's *real* ``invoke_agent`` span.

The whole point of span-exact evaluation is to attach the eval to **the exact
span the agent framework itself created** -- not to a second, runner-fabricated
span that merely happens to be named ``invoke_agent``.

Microsoft Agent Framework, once OpenTelemetry providers are configured, emits a
span named ``invoke_agent`` (per the OTel GenAI semantic conventions, the span
name is ``invoke_agent`` or ``invoke_agent <agent_name>``) around each
``agent.run(...)`` call. This module installs a lightweight
:class:`~opentelemetry.sdk.trace.SpanProcessor` that records the
:class:`~opentelemetry.trace.SpanContext` of that span the moment it starts, so
the runner can build a ``traceparent`` pointing at the framework's authentic
span instead of minting its own.

Usage
-----
Add the processor to the tracer provider (once, after Azure Monitor is
configured), then use :func:`capture_invoke_agent` as a context manager around
``agent.run(...)`` to obtain the captured span context::

    install_invoke_agent_capture()
    with capture_invoke_agent() as captured:
        result = await agent.run(query)
    span_ctx = captured.span_context  # the framework's real invoke_agent span
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import SpanContext

logger = logging.getLogger("span-capture")

# Span-name prefix the framework uses for the agent-invocation span. Per the
# GenAI semantic conventions the span is named "invoke_agent" or
# "invoke_agent <agent_name>", so we match on the prefix.
INVOKE_AGENT_SPAN_PREFIX = "invoke_agent"


@dataclass
class CapturedSpan:
    """Holds the SpanContext of a captured ``invoke_agent`` span."""

    span_context: Optional[SpanContext] = None
    span_name: Optional[str] = None

    @property
    def captured(self) -> bool:
        return self.span_context is not None


# Per-invocation slot. Using a ContextVar keeps concurrent agent runs isolated:
# each ``capture_invoke_agent()`` scope sets its own slot, and the processor --
# which runs on the same execution context that started the span -- writes into
# whichever slot is active for that context.
_active_capture: contextvars.ContextVar[Optional[CapturedSpan]] = (
    contextvars.ContextVar("active_invoke_agent_capture", default=None)
)


class InvokeAgentCaptureProcessor(SpanProcessor):
    """Records the SpanContext of the framework's ``invoke_agent`` span.

    On span start, if the span name matches the ``invoke_agent`` prefix and a
    capture slot is active on the current context, the span's context is stored
    into that slot. This is the framework's authentic span -- not a span this
    POC created -- so evaluation binds to the real agent invocation.
    """

    def on_start(
        self, span: "trace.Span", parent_context: Optional[object] = None
    ) -> None:
        name = getattr(span, "name", "") or ""
        if not name.startswith(INVOKE_AGENT_SPAN_PREFIX):
            return

        slot = _active_capture.get()
        if slot is None:
            # No active capture scope -- nothing to record (e.g. a stray
            # invoke_agent span outside a capture() block).
            return

        if slot.captured:
            # Already captured the outermost invoke_agent for this scope; keep
            # the first (top-level) one and ignore nested duplicates.
            return

        ctx = span.get_span_context()
        slot.span_context = ctx
        slot.span_name = name
        logger.info(
            "captured framework invoke_agent span: name=%s trace_id=%032x "
            "span_id=%016x",
            name,
            ctx.trace_id,
            ctx.span_id,
        )

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D401 - no-op
        return

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


_installed = False


def install_invoke_agent_capture() -> None:
    """Add the capture processor to the active tracer provider (idempotent)."""
    global _installed
    if _installed:
        return

    provider = trace.get_tracer_provider()
    add_processor = getattr(provider, "add_span_processor", None)
    if add_processor is None:
        raise RuntimeError(
            "Active tracer provider does not support add_span_processor; "
            "call setup_observability() before install_invoke_agent_capture()."
        )

    add_processor(InvokeAgentCaptureProcessor())
    _installed = True
    logger.info("InvokeAgentCaptureProcessor installed on tracer provider")


@contextmanager
def capture_invoke_agent() -> Iterator[CapturedSpan]:
    """Scope in which the next ``invoke_agent`` span's context is captured.

    Yields a :class:`CapturedSpan`. After the block runs ``agent.run(...)``,
    ``captured.span_context`` holds the framework's real ``invoke_agent`` span
    context (or ``None`` if the framework emitted no such span -- e.g. its
    instrumentation is disabled).
    """
    slot = CapturedSpan()
    token = _active_capture.set(slot)
    try:
        yield slot
    finally:
        _active_capture.reset(token)
