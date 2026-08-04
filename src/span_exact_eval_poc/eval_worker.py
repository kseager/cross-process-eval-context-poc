"""Evaluation worker -- a **separate process** from the runner.

Exposes ``POST /evaluate``. The runner calls it once per dataset row, passing:

* ``invoke_agent_traceparent`` -- a W3C ``traceparent`` pointing at *exactly*
  the ``invoke_agent`` span the runner created for this row, and
* the data needed to evaluate (``query``, ``response``, ``ground_truth``).

The worker rebuilds a remote parent context from that ``traceparent`` and starts
a ``gen_ai.evaluation.results`` span **as a child of that specific
invoke_agent span** (span-exact), in the same trace. The evaluation score is set
as span attributes and exported to App Insights.

Because the parent is set from the propagated ids -- not a live in-process span
-- this works fully cross-process, and the agent process is never involved.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from .telemetry import setup_observability
from .trace_context import parent_context_from_traceparent, traceparent_for_span

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-worker")

SERVICE_NAME = "eval-worker"

EVAL_SPAN_NAME = "gen_ai.evaluation.results"

app = FastAPI(title="span-exact-eval-worker")

_tracer: Tracer | None = None


class EvaluateRequest(BaseModel):
    """Payload the runner sends per row."""

    item_id: str
    query: str
    response: str
    ground_truth: str
    # traceparent pointing at the specific invoke_agent span for this row.
    invoke_agent_traceparent: str


class EvaluateResponse(BaseModel):
    """What the worker returns, echoing the correlation ids for verification."""

    item_id: str
    score: float
    evaluator: str
    # The eval span's own ids and the parent it was attached to.
    eval_trace_id: str
    eval_span_id: str
    parent_span_id: str


def _score(response: str, ground_truth: str) -> float:
    """Trivial stand-in scorer (exact-match ratio).

    Replace with a real evaluator (e.g. azure-ai-evaluation) as needed; the
    correlation mechanism is independent of the scoring logic.
    """
    expected = ground_truth.strip().lower()
    actual = response.strip().lower()
    if not expected:
        return 0.0
    return 1.0 if expected in actual else 0.0


@app.on_event("startup")
def _startup() -> None:
    global _tracer
    load_dotenv()
    _tracer = setup_observability(SERVICE_NAME)
    logger.info("eval-worker observability configured")


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Score one row and emit a span-exact ``gen_ai.evaluation.results`` span."""
    assert _tracer is not None, "tracer not initialized"

    # Rebuild the remote parent from the propagated invoke_agent traceparent.
    parent_ctx = parent_context_from_traceparent(req.invoke_agent_traceparent)

    score = _score(req.response, req.ground_truth)

    # Start the eval span AS A CHILD of the specific invoke_agent span.
    with _tracer.start_as_current_span(
        EVAL_SPAN_NAME, context=parent_ctx
    ) as span:
        span.set_attribute("gen_ai.evaluation.item_id", req.item_id)
        span.set_attribute("gen_ai.evaluation.evaluator", "exact_match")
        span.set_attribute("gen_ai.evaluation.score", score)
        span.set_attribute("gen_ai.evaluation.ground_truth", req.ground_truth)
        span.set_attribute("gen_ai.evaluation.response", req.response)

        eval_ctx = span.get_span_context()
        # The parent traceparent's span_id is the invoke_agent span_id we
        # attached to; echo it back for verification/logging.
        parent_span_id = req.invoke_agent_traceparent.split("-")[2]

        logger.info(
            "eval span %s (trace=%s) attached to invoke_agent span %s "
            "(item=%s, score=%s)",
            traceparent_for_span(span).split("-")[2],
            f"{eval_ctx.trace_id:032x}",
            parent_span_id,
            req.item_id,
            score,
        )

        return EvaluateResponse(
            item_id=req.item_id,
            score=score,
            evaluator="exact_match",
            eval_trace_id=f"{eval_ctx.trace_id:032x}",
            eval_span_id=f"{eval_ctx.span_id:016x}",
            parent_span_id=parent_span_id,
        )
