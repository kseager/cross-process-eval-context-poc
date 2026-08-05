"""Eval-worker -- the **main driver loop** of the POC.

This process owns the execution loop. For every dataset row it:

1. **Authors** an ``invoke_agent`` span and builds a W3C ``traceparent`` that
   points at it.
2. **Invokes the agent** -- a *separate* HTTP service (``AGENT_SERVICE_URL``) --
   passing the ``traceparent`` as a request header so the agent-service opens an
   ``execute_agent`` span *under* this ``invoke_agent`` span (matching ACA's
   ``invoke_agent -> execute_agent -> chat`` hierarchy). The agent *code* needs
   no tracing changes; the service just runs it beneath ``execute_agent``.
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
from opentelemetry.trace import Status, StatusCode
from dotenv import load_dotenv

from .dataset import load_dataset
from .evaluation import check_evaluation_results, evaluate_traces
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


async def run(dataset_path: Path, agent_service_url: str) -> list[dict[str, str]]:
    """Drive the loop: author invoke_agent, call the agent, stamp ground truth.

    The ground-truth object is set as an attribute directly on the
    ``invoke_agent`` span this process creates -- there is no separate eval span.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
    setup_observability()
    tracer = trace.get_tracer(SERVICE_NAME)

    rows = list(load_dataset(dataset_path))

    # Collected for an App Insights lookup summary at the end. operation_Id in
    # App Insights == the OTel trace_id (hex), so this is what you paste into a
    # Transaction search / KQL query to find each run's invoke_agent span.
    results: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for item in rows:
            print(f"\n[{item.id}]")
            print(f"  query        : {item.query}")

            # Author the invoke_agent span; the agent-service opens execute_agent
            # (ACA-style) beneath it via the traceparent header.
            with tracer.start_as_current_span("invoke_agent") as span:
                span_ctx = span.get_span_context()
                operation_id = f"{span_ctx.trace_id:032x}"
                invoke_agent_traceparent = traceparent_for_span(span)
                print(
                    f"  invoke_agent : trace_id={operation_id} "
                    f"span_id={span_ctx.span_id:016x}"
                )
                print(f"  operation_Id : {operation_id}  (App Insights)")

                # Attach the ground-truth object DIRECTLY on the invoke_agent
                # span. OTel attribute values must be primitives, so structured
                # objects travel as a JSON string. Ground truth is driver-owned
                # input, so it is stamped regardless of whether the agent call
                # succeeds. Ground truth only -- no score.
                ground_truth_json = json.dumps(
                    item.ground_truth, ensure_ascii=False
                )
                span.set_attribute("gen_ai.evaluation.item_id", item.id)
                span.set_attribute(GROUND_TRUTH_ATTRIBUTE, ground_truth_json)

                # Invoke the REMOTE agent, injecting the traceparent as a header
                # so its execute_agent span nests under this invoke_agent span.
                try:
                    agent_resp = await client.post(
                        agent_service_url,
                        json={"item_id": item.id, "query": item.query},
                        headers={"traceparent": invoke_agent_traceparent},
                    )
                    agent_resp.raise_for_status()
                    agent_result = agent_resp.json()
                except httpx.HTTPError as exc:
                    # Don't abort the whole run on one transient failure; record
                    # the error on the span and move to the next item.
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    logger.warning("agent call failed for %s: %s", item.id, exc)
                    print(f"  ERROR        : {exc}")
                    results.append(
                        {
                            "item_id": item.id,
                            "operation_id": operation_id,
                            "invoke_agent_span_id": f"{span_ctx.span_id:016x}",
                            "status": "ERROR",
                        }
                    )
                    continue

                response_text = agent_result["response"]
            print(f"  response     : {response_text}")
            print(f"  ground_truth : {item.ground_truth}")
            print(
                f"  execute_agent: span_id={agent_result['execute_agent_span_id']} "
                f"(child of invoke_agent)"
            )

            # Sanity check: the agent's execute_agent span must correlate to
            # this invoke_agent span (same trace).
            assert (
                agent_result["execute_agent_trace_id"]
                == f"{span_ctx.trace_id:032x}"
            ), "execute_agent is not in the same trace as invoke_agent"

            results.append(
                {
                    "item_id": item.id,
                    "operation_id": operation_id,
                    "invoke_agent_span_id": f"{span_ctx.span_id:016x}",
                    "execute_agent_span_id": agent_result["execute_agent_span_id"],
                    "status": "OK",
                }
            )

    return results


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

    results = asyncio.run(run(args.dataset, args.agent_service_url))

    print(
        "\nDone. invoke_agent (with ground_truth attribute) + execute_agent "
        "exported to App Insights. Evaluation itself is a separate "
        "post-processing step."
    )

    # App Insights lookup summary. operation_Id == the OTel trace_id.
    print("\n=== App Insights operation_Id per item ===")
    print(f"{'item_id':<10} {'status':<7} operation_Id")
    op_ids = []
    for r in results:
        print(f"{r['item_id']:<10} {r['status']:<7} {r['operation_id']}")
        op_ids.append(r["operation_id"])

    if op_ids:
        joined = ", ".join(f'"{o}"' for o in op_ids)
        print(
            "\nPaste into App Insights Logs (KQL) to see every span from this run:\n"
            f"union traces, dependencies, requests, exceptions\n"
            f"| where operation_Id in ({joined})\n"
            "| project timestamp, itemType, name, operation_Id, operation_ParentId, "
            'ground_truth = tostring(customDimensions["gen_ai.evaluation.ground_truth"])\n'
            "| order by timestamp asc"
        )

    # -- Evaluation (post-processing) -----------------------------------------
    # After every agent invocation has produced a trace, evaluate those traces
    # BY TRACE ID with Foundry's built-in evaluators. This is the "separate
    # post-processing step" referenced above -- it does NOT run inline per item.
    # Gated behind RUN_EVALUATION (default: on) so the driver can be run purely
    # to emit traces if desired.
    if os.environ.get("RUN_EVALUATION", "true").lower() not in ("1", "true", "yes"):
        print("\nRUN_EVALUATION disabled; skipping trace evaluation.")
        return

    # Only evaluate traces whose agent call actually succeeded; a trace from a
    # failed agent call has no response for the evaluators to score.
    eval_trace_ids = [r["operation_id"] for r in results if r["status"] == "OK"]
    if not eval_trace_ids:
        print("\nNo successful traces to evaluate; skipping evaluation.")
        return

    lookback_hours = int(os.environ.get("EVAL_LOOKBACK_HOURS", "1"))
    print(
        f"\n=== Evaluating {len(eval_trace_ids)} trace(s) by trace_id "
        f"(lookback {lookback_hours}h) ==="
    )
    try:
        summary = evaluate_traces(eval_trace_ids, lookback_hours=lookback_hours)
        check_evaluation_results(summary)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole run
        logger.warning("evaluation step failed: %s", exc)
        print(f"  evaluation step failed: {exc}")


if __name__ == "__main__":
    cli()
