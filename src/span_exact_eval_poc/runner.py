"""Runner process -- owns the ``invoke_agent`` span and drives evaluation.

For every dataset row this process:

1. Starts an ``invoke_agent`` span (the runner owns it; the agent process needs
   no changes -- it just receives the request).
2. Runs the Foundry agent on the query inside that span.
3. Builds a W3C ``traceparent`` that points at *this specific* ``invoke_agent``
   span and POSTs it -- with the query/response/ground_truth -- to the separate
   **eval-worker** process (``EVAL_WORKER_URL``).
4. The eval-worker creates a ``gen_ai.evaluation.results`` span **as a child of
   that exact invoke_agent span**, span-exact and cross-process.

The evaluation therefore attaches to exactly one span, in another process,
without any agent-side code.
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
from .telemetry import setup_observability
from .trace_context import traceparent_for_span

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-runner")

SERVICE_NAME = "eval-runner"

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "data" / "dataset.jsonl"
)
DEFAULT_EVAL_WORKER_URL = "http://localhost:8001/evaluate"


async def run(dataset_path: Path, eval_worker_url: str) -> None:
    """Run the agent over every row and dispatch span-exact evaluations."""
    tracer = setup_observability(SERVICE_NAME)
    agent = build_agent()

    rows = list(load_dataset(dataset_path))

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in rows:
            print(f"\n[{item.id}]")
            print(f"  query        : {item.query}")

            # The runner OWNS this span; the agent process is not involved in
            # span creation. We name it invoke_agent to mirror the framework's
            # own agent span convention.
            with tracer.start_as_current_span("invoke_agent") as span:
                result = await agent.run(item.query)
                response_text = str(result)

                # Build a traceparent pointing at THIS invoke_agent span, so the
                # evaluator can attach its span as a child of exactly this one.
                invoke_agent_traceparent = traceparent_for_span(span)
                span_ctx = span.get_span_context()
                print(
                    f"  invoke_agent : trace_id={span_ctx.trace_id:032x} "
                    f"span_id={span_ctx.span_id:016x}"
                )

                # Dispatch evaluation to the SEPARATE worker process over HTTP.
                payload = {
                    "item_id": item.id,
                    "query": item.query,
                    "response": response_text,
                    "ground_truth": item.ground_truth,
                    "invoke_agent_traceparent": invoke_agent_traceparent,
                }
                resp = await client.post(eval_worker_url, json=payload)
                resp.raise_for_status()
                eval_result = resp.json()

            print(f"  response     : {response_text}")
            print(f"  ground_truth : {item.ground_truth}")
            print(
                f"  eval span    : span_id={eval_result['eval_span_id']} "
                f"parent_span_id={eval_result['parent_span_id']} "
                f"score={eval_result['score']}"
            )
            # Sanity check: the eval span's parent must equal this
            # invoke_agent span (span-exact correlation).
            assert (
                eval_result["parent_span_id"]
                == f"{span_ctx.span_id:016x}"
            ), "eval span parent did not match invoke_agent span"
            assert (
                eval_result["eval_trace_id"]
                == f"{span_ctx.trace_id:032x}"
            ), "eval span is not in the same trace as invoke_agent"


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
    print("\nDone. invoke_agent + gen_ai.evaluation.results exported to App Insights.")


if __name__ == "__main__":
    cli()
