"""Ground-truth attach worker -- a **separate process** from the runner.

Exposes ``POST /evaluate``. The runner calls it once per dataset row, passing:

* ``invoke_agent_traceparent`` -- a W3C ``traceparent`` pointing at *exactly*
  the framework's ``invoke_agent`` span for this row, and
* the ground-truth data (``item_id``, ``query``, ``ground_truth``).

The worker rebuilds a remote parent context from that ``traceparent`` and starts
a ``gen_ai.evaluation.input`` span **as a child of that specific
invoke_agent span** (span-exact), in the same trace. It attaches **only the
ground-truth object** to that child span (as an ``evaluation.ground_truth``
event plus a JSON attribute), mirroring the single-process
``trace-ground-truth-poc``, and exports it to App Insights.

IMPORTANT -- this worker does NOT evaluate. It only attaches ground-truth
*input* to the correct span. Actual evaluation (scoring the agent's responses
against this ground truth) is a **separate post-processing step that runs after
all agent invocations have completed** -- e.g. a batch job that reads these
spans back from App Insights and computes metrics. That step is out of scope for
this POC, which proves only span-exact ground-truth attachment.

Because the parent is set from the propagated ids -- not a live in-process span
-- this works fully cross-process, and the agent process is never involved.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from .telemetry import setup_observability
from .trace_context import parent_context_from_traceparent, traceparent_for_span

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-worker")

SERVICE_NAME = "eval-worker"

EVAL_SPAN_NAME = "gen_ai.evaluation.input"
GROUND_TRUTH_EVENT = "evaluation.ground_truth"

app = FastAPI(title="span-exact-eval-worker")

_tracer: Tracer | None = None


class EvaluateRequest(BaseModel):
    """Payload the runner sends per row (ground-truth attach only)."""

    item_id: str
    query: str
    # Full ground-truth object (dict/str/list) to attach to the span.
    ground_truth: Any
    # traceparent pointing at the specific invoke_agent span for this row.
    invoke_agent_traceparent: str


class EvaluateResponse(BaseModel):
    """What the worker returns, echoing the correlation ids for verification."""

    item_id: str
    # The ground-truth span's own ids and the parent it was attached to.
    ground_truth_trace_id: str
    ground_truth_span_id: str
    parent_span_id: str


@app.on_event("startup")
def _startup() -> None:
    global _tracer
    load_dotenv()
    _tracer = setup_observability(SERVICE_NAME)
    logger.info("eval-worker observability configured")


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Attach ground-truth input to a span-exact child of invoke_agent.

    Creates a ``gen_ai.evaluation.input`` span parented to the exact
    ``invoke_agent`` span (via the propagated traceparent) and attaches the
    ground-truth object to it. The span represents evaluation *input* (ground
    truth) only -- no scoring happens here (see module docstring: evaluation is
    a separate post-processing step run after all agent invocations).
    """
    assert _tracer is not None, "tracer not initialized"

    # Rebuild the remote parent from the propagated invoke_agent traceparent.
    parent_ctx = parent_context_from_traceparent(req.invoke_agent_traceparent)

    # Start the ground-truth span AS A CHILD of the specific invoke_agent span.
    with _tracer.start_as_current_span(
        EVAL_SPAN_NAME, context=parent_ctx
    ) as span:
        # Serialize the ground-truth object once; OTel attribute/event values
        # must be primitives, so structured objects travel as a JSON string.
        ground_truth_json = json.dumps(req.ground_truth, ensure_ascii=False)

        # Attach ONLY the ground truth -- no score, evaluator, or response.
        span.set_attribute("gen_ai.evaluation.item_id", req.item_id)
        span.set_attribute("gen_ai.evaluation.ground_truth", ground_truth_json)

        # Attach the FULL ground-truth object as an event on this span --
        # the same paradigm as trace-ground-truth-poc, but on a span-exact
        # child of the invoke_agent span (in a separate process).
        span.add_event(
            GROUND_TRUTH_EVENT,
            {
                "item_id": req.item_id,
                "query": req.query,
                "ground_truth": ground_truth_json,
            },
        )

        gt_ctx = span.get_span_context()
        # The parent traceparent's span_id is the invoke_agent span_id we
        # attached to; echo it back for verification/logging.
        parent_span_id = req.invoke_agent_traceparent.split("-")[2]

        logger.info(
            "ground-truth span %s (trace=%s) attached to invoke_agent span %s "
            "(item=%s)",
            traceparent_for_span(span).split("-")[2],
            f"{gt_ctx.trace_id:032x}",
            parent_span_id,
            req.item_id,
        )

        return EvaluateResponse(
            item_id=req.item_id,
            ground_truth_trace_id=f"{gt_ctx.trace_id:032x}",
            ground_truth_span_id=f"{gt_ctx.span_id:016x}",
            parent_span_id=parent_span_id,
        )
