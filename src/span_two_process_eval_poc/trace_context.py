"""W3C trace-context helpers for span-exact, cross-process correlation.

This module is the heart of the POC. It lets one process (the evaluator) create
a span that is a child of *exactly one* span created in another process (the
runner's ``invoke_agent`` span), by carrying that span's identity as a W3C
``traceparent`` string.

Key idea
--------
A ``traceparent`` (``00-<trace_id>-<span_id>-<flags>``) encodes a specific
``(trace_id, span_id)`` pair. If the evaluator rebuilds an OTel *remote* parent
context from that string and starts its span inside it, the new span's
``parentSpanId`` becomes that exact ``span_id`` -- span-exact correlation, not
just trace-level. In App Insights this is queryable via the first-class
``operation_ParentId`` column.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    format_span_id,
    format_trace_id,
)

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


def traceparent_for_span(span: trace.Span) -> str:
    """Build a ``traceparent`` that points at *span* specifically."""
    ctx = span.get_span_context()
    return build_traceparent(ctx.trace_id, ctx.span_id, ctx.trace_flags)


def parent_context_from_traceparent(traceparent: str) -> Context:
    """Rebuild a *remote* parent :class:`Context` from a ``traceparent``.

    The returned context wraps a non-recording span whose ``SpanContext``
    carries the parsed ``(trace_id, span_id)`` and is flagged ``is_remote``.
    Starting a new span with this context as parent makes the new span a child
    of exactly the span the ``traceparent`` came from.

    Args:
        traceparent: A W3C ``traceparent`` header value.

    Returns:
        An OTel :class:`Context` suitable as the ``context=`` argument to
        ``tracer.start_as_current_span``.

    Raises:
        ValueError: If *traceparent* is not a well-formed W3C value.
    """
    parts = traceparent.strip().split("-")
    if len(parts) != 4:
        raise ValueError(f"Malformed traceparent: {traceparent!r}")

    _version, trace_id_hex, span_id_hex, flags_hex = parts
    if len(trace_id_hex) != 32 or len(span_id_hex) != 16:
        raise ValueError(f"Malformed traceparent ids: {traceparent!r}")

    span_context = SpanContext(
        trace_id=int(trace_id_hex, 16),
        span_id=int(span_id_hex, 16),
        is_remote=True,
        trace_flags=TraceFlags(int(flags_hex, 16)),
    )
    return trace.set_span_in_context(NonRecordingSpan(span_context))
