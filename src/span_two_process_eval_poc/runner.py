"""Runner process -- runs the agent and attaches ground truth span-exactly.

For every dataset row this process:

1. Runs the Foundry agent on the query. The **Agent Framework** creates its own
   ``invoke_agent`` span; a span processor captures that real span's context
   (the agent needs no changes).
2. Builds a W3C ``traceparent`` that points at *that specific* framework
   ``invoke_agent`` span and POSTs it -- with the query and ground_truth -- to
   the separate **eval-worker** process (``EVAL_WORKER_URL``).
3. The eval-worker attaches the **ground-truth object** to that exact
   invoke_agent span (span-exact, cross-process) via a child
   ``gen_ai.evaluation.input`` span.

The worker only *attaches ground-truth input*. Actual evaluation (scoring
responses against this ground truth) is a **separate post-processing step that
runs after all agent invocations complete** -- it is intentionally out of scope
here; this POC only proves span-exact ground-truth attachment.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .agent import build_agent
from .dataset import load_dataset
from .span_capture import (
    capture_invoke_agent,
    install_invoke_agent_capture,
)
from .telemetry import setup_observability
from .trace_context import traceparent_from_span_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-runner")

SERVICE_NAME = "eval-runner"

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "data" / "dataset.jsonl"
)
DEFAULT_EVAL_WORKER_URL = "http://localhost:8001/evaluate"


async def run(dataset_path: Path, eval_worker_url: str) -> None:
    """Run the agent over every row and attach ground truth span-exactly.

    For each row this process runs the agent and lets the **framework** create
    its own ``invoke_agent`` span. A span processor captures that real span's
    context, which we serialize into a ``traceparent`` and hand to the separate
    eval-worker so it can attach the ground truth to *that exact* span.
    """
    tracer = setup_observability(SERVICE_NAME)
    # Capture the framework's REAL invoke_agent span (see span_capture.py) --
    # we never fabricate our own invoke_agent span.
    install_invoke_agent_capture()
    agent = build_agent()

    rows = list(load_dataset(dataset_path))

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in rows:
            print(f"\n[{item.id}]")
            print(f"  query        : {item.query}")

            # Run the agent. The Agent Framework creates its own ``invoke_agent``
            # span internally; capture_invoke_agent() records that real span's
            # context. The agent process needs no POC-specific code.
            with capture_invoke_agent() as captured:
                result = await agent.run(item.query)
                response_text = str(result)

            if not captured.captured:
                raise RuntimeError(
                    "Framework did not emit an invoke_agent span; cannot bind "
                    "ground truth span-exactly. Ensure Agent Framework "
                    "telemetry is enabled."
                )

            span_ctx = captured.span_context

            # Build a traceparent pointing at the framework's REAL invoke_agent
            # span, so the worker can attach the ground truth to exactly it.
            invoke_agent_traceparent = traceparent_from_span_context(span_ctx)
            print(
                f"  invoke_agent : trace_id={span_ctx.trace_id:032x} "
                f"span_id={span_ctx.span_id:016x} (framework span)"
            )

            # Dispatch to the SEPARATE worker process over HTTP. The worker's
            # sole job is to attach the ground-truth object to the invoke_agent
            # span -- it does NOT evaluate.
            payload = {
                "item_id": item.id,
                "query": item.query,
                "ground_truth": item.ground_truth,
                "invoke_agent_traceparent": invoke_agent_traceparent,
            }
            resp = await client.post(eval_worker_url, json=payload)
            resp.raise_for_status()
            attach_result = resp.json()

            print(f"  response     : {response_text}")
            print(f"  ground_truth : {item.ground_truth}")
            print(
                f"  gt span      : span_id={attach_result['ground_truth_span_id']} "
                f"parent_span_id={attach_result['parent_span_id']}"
            )
            # Sanity check: the ground-truth span's parent must equal the
            # framework's invoke_agent span (span-exact correlation).
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
        "--eval-worker-url",
        type=str,
        default=os.environ.get("EVAL_WORKER_URL", DEFAULT_EVAL_WORKER_URL),
        help="URL of the eval-worker /evaluate endpoint.",
    )
    args = parser.parse_args()

    asyncio.run(run(args.dataset, args.eval_worker_url))
    print(
        "\nDone. invoke_agent + gen_ai.evaluation.input (ground truth) exported "
        "to App Insights. Evaluation itself is a separate post-processing step."
    )


if __name__ == "__main__":
    cli()
