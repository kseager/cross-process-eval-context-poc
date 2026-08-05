"""Runner process -- authors ``invoke_agent`` and drives the two other services.

This mirrors the ACA topology (Yingying's design). For every dataset row this
process:

1. **Authors** an ``invoke_agent`` span (the runner owns it). It reads that
   span's id directly -- no framework-span capture is needed, because the agent
   runs in a *separate* service that emits ``execute_agent``, not
   ``invoke_agent``.
2. Calls the **agent-service** (``AGENT_SERVICE_URL``) over HTTP, injecting a
   W3C ``traceparent`` header pointing at the ``invoke_agent`` span. The
   agent-service opens ``execute_agent`` *under* that span (passive header
   recipient) and returns the response.
3. POSTs the **ground-truth object** + the same ``traceparent`` to the separate
   **eval-worker** (``EVAL_WORKER_URL``), which attaches the ground truth to the
   exact ``invoke_agent`` span via a child ``gen_ai.evaluation.input`` span.

Resulting span tree (one trace)::

    invoke_agent                 (runner authors, span_id=A)
    ├─ execute_agent             (agent-service, via traceparent header)
    │   └─ chat ...              (agent framework)
    └─ gen_ai.evaluation.input   (eval-worker, parentSpanId=A)  ← span-exact

The eval-worker only *attaches ground-truth input*. Actual evaluation (scoring
responses against this ground truth) is a **separate post-processing step that
runs after all agent invocations complete** -- intentionally out of scope here.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .dataset import load_dataset
from .telemetry import setup_observability
from .trace_context import traceparent_for_span

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-runner")

SERVICE_NAME = "eval-runner"

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "data" / "dataset.jsonl"
)
DEFAULT_AGENT_SERVICE_URL = "http://localhost:8002/invoke"
DEFAULT_EVAL_WORKER_URL = "http://localhost:8001/evaluate"


async def run(
    dataset_path: Path, agent_service_url: str, eval_worker_url: str
) -> None:
    """Author invoke_agent, call the remote agent, attach ground truth.

    The runner owns the ``invoke_agent`` span and injects its ``traceparent``
    into both the agent-service call (as a header) and the eval-worker call (in
    the body), so both correlate to exactly that span.
    """
    tracer = setup_observability(SERVICE_NAME)

    rows = list(load_dataset(dataset_path))

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in rows:
            print(f"\n[{item.id}]")
            print(f"  query        : {item.query}")

            # The runner AUTHORS invoke_agent. The agent is remote and emits
            # execute_agent (not invoke_agent), so there is no duplicate span
            # and no need to capture a framework span.
            with tracer.start_as_current_span("invoke_agent") as span:
                span_ctx = span.get_span_context()
                # Read the invoke_agent span_id straight off the span we own.
                invoke_agent_traceparent = traceparent_for_span(span)
                print(
                    f"  invoke_agent : trace_id={span_ctx.trace_id:032x} "
                    f"span_id={span_ctx.span_id:016x} (runner-authored)"
                )

                # Call the REMOTE agent-service, injecting the traceparent as a
                # header so its execute_agent span nests under invoke_agent.
                agent_resp = await client.post(
                    agent_service_url,
                    json={"item_id": item.id, "query": item.query},
                    headers={"traceparent": invoke_agent_traceparent},
                )
                agent_resp.raise_for_status()
                agent_result = agent_resp.json()
                response_text = agent_result["response"]

                # Dispatch ground truth to the SEPARATE eval-worker over HTTP.
                # Its sole job is to attach the ground-truth object to the
                # invoke_agent span -- it does NOT evaluate.
                eval_payload = {
                    "item_id": item.id,
                    "query": item.query,
                    "ground_truth": item.ground_truth,
                    "invoke_agent_traceparent": invoke_agent_traceparent,
                }
                eval_resp = await client.post(eval_worker_url, json=eval_payload)
                eval_resp.raise_for_status()
                attach_result = eval_resp.json()

            print(f"  response     : {response_text}")
            print(f"  ground_truth : {item.ground_truth}")
            print(
                f"  execute_agent: span_id={agent_result['execute_agent_span_id']} "
                f"(child of invoke_agent)"
            )
            print(
                f"  gt span      : span_id={attach_result['ground_truth_span_id']} "
                f"parent_span_id={attach_result['parent_span_id']}"
            )

            # Sanity checks: both the agent's execute_agent span and the
            # ground-truth span must correlate to this invoke_agent span.
            assert (
                agent_result["execute_agent_trace_id"]
                == f"{span_ctx.trace_id:032x}"
            ), "execute_agent is not in the same trace as invoke_agent"
            assert (
                attach_result["parent_span_id"]
                == f"{span_ctx.span_id:016x}"
            ), "ground-truth span parent did not match invoke_agent span"
            assert (
                attach_result["ground_truth_trace_id"]
                == f"{span_ctx.trace_id:032x}"
            ), "ground-truth span is not in the same trace as invoke_agent"


def cli() -> None:
    """Parse arguments, load config, and run the POC runner."""
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
    parser.add_argument(
        "--eval-worker-url",
        type=str,
        default=os.environ.get("EVAL_WORKER_URL", DEFAULT_EVAL_WORKER_URL),
        help="URL of the eval-worker /evaluate endpoint.",
    )
    args = parser.parse_args()

    asyncio.run(run(args.dataset, args.agent_service_url, args.eval_worker_url))
    print(
        "\nDone. invoke_agent + execute_agent + gen_ai.evaluation.input "
        "(ground truth) exported to App Insights. Evaluation itself is a "
        "separate post-processing step."
    )


if __name__ == "__main__":
    cli()
