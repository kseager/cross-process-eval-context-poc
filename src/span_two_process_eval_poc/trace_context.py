"""W3C trace-context helpers for span-exact, cross-process correlation.

Lets one process create a span that is a child of exactly one span created in
another process (e.g. the driver's ``evaluation_context`` span) by carrying that
span's identity as a W3C ``traceparent`` string. Rebuilding a remote parent
context from that string and starting a span inside it sets the new span's
``parentSpanId`` to that exact ``span_id`` (queryable in App Insights via
``operation_ParentId``).
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    TraceFlags,
    format_span_id,
    format_trace_id,
)
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

_TRACEPARENT_HEADER = "traceparent"
_propagator = TraceContextTextMapPropagator()

EVALUATION_CONTEXT_SPAN_NAME = "evaluation_context"

GROUND_TRUTH_ATTRIBUTE = "gen_ai.evaluation.ground_truth"

_TRACEPARENT_VERSION = "00"


def build_traceparent(
    trace_id: int, span_id: int, trace_flags: TraceFlags | int
) -> str:
    """Serialize a span's context into a W3C ``traceparent`` string.

    Args:
        trace_id: The 128-bit trace id (as an int, from ``SpanContext``).
        span_id: The 64-bit span id (as an int, from ``SpanContext``).
        trace_flags: The trace flags (e.g. sampled bit).

    Returns:
        A ``traceparent`` header value:
        ``00-<32 hex trace_id>-<16 hex span_id>-<2 hex flags>``.
    """
    flags = int(trace_flags) & 0xFF
    return (
        f"{_TRACEPARENT_VERSION}-"
        f"{format_trace_id(trace_id)}-"
        f"{format_span_id(span_id)}-"
        f"{flags:02x}"
    )


def parent_context_from_traceparent(traceparent: str) -> Context:
    """Rebuild a *remote* parent :class:`Context` from a ``traceparent``.

    Strict wrapper around ``TraceContextTextMapPropagator.extract``: raises on
    missing/malformed input instead of silently yielding an invalid context that
    would orphan the child as a new root.

    Args:
        traceparent: A W3C ``traceparent`` header value.

    Returns:
        An OTel :class:`Context` suitable as the ``context=`` argument to
        ``tracer.start_as_current_span``.

    Raises:
        ValueError: If *traceparent* is missing or not a well-formed W3C value.
    """
    if not traceparent:
        raise ValueError("Empty traceparent")

    ctx = _propagator.extract(carrier={_TRACEPARENT_HEADER: traceparent})

    span_context = trace.get_current_span(ctx).get_span_context()
    if not span_context.is_valid:
        raise ValueError(f"Malformed traceparent: {traceparent!r}")

    return ctx
