"""W3C trace-context helpers for span-exact, cross-process correlation.

This module is the heart of the POC. It lets one process create a span that is
a child of *exactly one* span created in another process (e.g. the driver's
``evaluation_context`` span), by carrying that span's identity as a W3C
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
    TraceFlags,
    format_span_id,
    format_trace_id,
)
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

_TRACEPARENT_HEADER = "traceparent"
_propagator = TraceContextTextMapPropagator()

# ---------------------------------------------------------------------------
# Shared span/attribute names (single source of truth for both processes).
# ---------------------------------------------------------------------------
# The driver-authored span that carries evaluation metadata (ground truth). In
# the attach-after model the agent authors its own invoke_agent span natively;
# the driver then attaches THIS span as a child of it (same trace, cross-process).
EVALUATION_CONTEXT_SPAN_NAME = "evaluation_context"

# Custom span attribute carrying the ground-truth object (JSON string).
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

    Uses OpenTelemetry's ``TraceContextTextMapPropagator.extract`` -- the
    idiomatic W3C parser (the same mechanism ACA's ``trace_utils`` uses) -- so
    all Trace Context edge cases are handled. Starting a new span with the
    returned context as parent makes the new span a child of exactly the span
    the ``traceparent`` came from.

    Unlike the propagator's default lenient behavior (which yields an *invalid*
    context on a bad header, silently orphaning the child as a new root), this
    wrapper is **strict**: it raises on missing/malformed input so POC mistakes
    surface loudly instead of producing a disconnected trace.

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

    # The propagator never raises; verify it actually parsed a valid parent.
    span_context = trace.get_current_span(ctx).get_span_context()
    if not span_context.is_valid:
        raise ValueError(f"Malformed traceparent: {traceparent!r}")

    return ctx
