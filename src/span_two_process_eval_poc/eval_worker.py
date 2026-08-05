"""Eval-worker -- the **main driver loop** of the POC.

This process owns the execution loop. For every dataset row it:

1. **Authors** an ``invoke_agent`` span and builds a W3C ``traceparent`` that
   points at it.
2. **Invokes the agent** -- a *separate* HTTP service (``AGENT_SERVICE_URL``) --
   passing the ``traceparent`` as a request header so the agent opens its
   ``execute_agent`` span *under* this ``invoke_agent`` span. The agent process
   needs no tracing code; it is a passive header recipient.
3. Sets the **ground-truth object** directly as an **attribute on the
   ``invoke_agent`` span** it created (no separate child span).

Resulting span tree (one trace)::

    invoke_agent                       (this process, span_id=A)
    │  • attribute: gen_ai.evaluation.ground_truth = {...}
    └─ execute_agent                   (agent-service, via traceparent header)
        └─ chat ...                    (agent framework)

The eval-worker only *attaches ground-truth input*. Actual evaluation (scoring
the agent's responses against this ground truth) is a **separate post-processing
step that runs after all agent invocations complete** -- e.g. a batch job that
reads these spans back from App Insights and computes metrics. That step is
intentionally out of scope for this POC.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from opentelemetry import trace
from dotenv import load_dotenv

from .dataset import load_dataset
from .telemetry import setup_observability
from .trace_context import traceparent_for_span

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-worker")

SERVICE_NAME = "eval-worker"

GROUND_TRUTH_ATTRIBUTE = "gen_ai.evaluation.ground_truth"

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "data" / "dataset.jsonl"
)
DEFAULT_AGENT_SERVICE_URL = "http://localhost:8002/invoke"


async def run(dataset_path: Path, agent_service_url: str) -> None:
    """Drive the loop: author invoke_agent, call the agent, stamp ground truth.

    The ground-truth object is set as an attribute directly on the
    ``invoke_agent`` span this process creates -- there is no separate eval span.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
    setup_observability()
    tracer = trace.get_tracer(SERVICE_NAME)

    rows = list(load_dataset(dataset_path))

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in rows:
            print(f"\n[{item.id}]")
            print(f"  query        : {item.query}")

            # Author the invoke_agent span; the agent is a remote service that
            # emits execute_agent (not invoke_agent) beneath it.
            with tracer.start_as_current_span("invoke_agent") as span:
                span_ctx = span.get_span_context()
                invoke_agent_traceparent = traceparent_for_span(span)
                print(
                    f"  invoke_agent : trace_id={span_ctx.trace_id:032x} "
                    f"span_id={span_ctx.span_id:016x}"
                )

                # Invoke the REMOTE agent, injecting the traceparent as a header
                # so its execute_agent span nests under this invoke_agent span.
                agent_resp = await client.post(
                    agent_service_url,
                    json={"item_id": item.id, "query": item.query},
                    headers={"traceparent": invoke_agent_traceparent},
                )
                agent_resp.raise_for_status()
                agent_result = agent_resp.json()
                response_text = agent_result["response"]

                # Attach the ground-truth object DIRECTLY on the invoke_agent
                # span. OTel attribute values must be primitives, so structured
                # objects travel as a JSON string. Ground truth only -- no score.
                ground_truth_json = json.dumps(
                    item.ground_truth, ensure_ascii=False
                )
                span.set_attribute("gen_ai.evaluation.item_id", item.id)
                span.set_attribute(GROUND_TRUTH_ATTRIBUTE, ground_truth_json)

            print(f"  response     : {response_text}")
            print(f"  ground_truth : {item.ground_truth}")
            print(
                f"  execute_agent: span_id={agent_result['execute_agent_span_id']} "
                f"(child of invoke_agent)"
            )

            # Sanity check: the agent's execute_agent span must correlate to
            # this invoke_agent span (same trace, parented to it).
            assert (
                agent_result["execute_agent_trace_id"]
                == f"{span_ctx.trace_id:032x}"
            ), "execute_agent is not in the same trace as invoke_agent"


def cli() -> None:
    """Parse arguments, load config, and run the driver loop."""
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DATASET_PATH", DEFAULT_DATASET)),
        help="Path to the JSONL dataset file.",
    )
    parser.add_argument(
        "--agent-service-url",
        type=str,
        default=os.environ.get("AGENT_SERVICE_URL", DEFAULT_AGENT_SERVICE_URL),
        help="URL of the agent-service /invoke endpoint.",
    )
    args = parser.parse_args()

    asyncio.run(run(args.dataset, args.agent_service_url))
    print(
        "\nDone. invoke_agent (with ground_truth attribute) + execute_agent "
        "exported to App Insights. Evaluation itself is a separate "
        "post-processing step."
    )


if __name__ == "__main__":
    cli()
