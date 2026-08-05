"""Driver that attaches an evaluation span to an agent's own trace.

The agent runs first, as a complete black box, with **no** incoming
``traceparent``. It authors its own ``invoke_agent`` span and returns that span's
``(trace_id, span_id)``. The driver then opens an ``evaluation_context`` span as
a **child** of the returned span -- same trace, ``operation_ParentId`` == the
agent's span -- and stamps ground truth on it.

Properties:

* **Zero request-path coupling.** The agent needs no ``traceparent`` handling,
  no context propagation, no ``RawAgent`` bypass. Any agent that can report the
  OTel span context it emitted works.
* **No manufactured outer span.** The agent's real ``invoke_agent`` span stays
  the root; no parent is fabricated above it.
* Ground truth is a child *annotation* of the real agent run, in the same trace.

Resulting span tree (one trace)::

    invoke_agent <name>                (agent-service, framework's own span)
    │  └─ chat ...                      (agent framework)
    └─ evaluation_context              (driver, attached via returned ids)
          • attribute: gen_ai.evaluation.ground_truth = {...}

This changes only the *authoring* topology; whether the eval service reads ground
truth off this attached child span is a separate consumption concern.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from opentelemetry import trace

from .dataset import load_dataset
from .telemetry import setup_observability
from .trace_context import (
    EVALUATION_CONTEXT_SPAN_NAME,
    GROUND_TRUTH_ATTRIBUTE,
    build_traceparent,
    parent_context_from_traceparent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attach-after")

SERVICE_NAME = "attach-after-driver"

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "dataset.jsonl"
DEFAULT_AGENT_SERVICE_URL = "http://localhost:8002/invoke-standalone"

# Trace flags to assume for the returned agent span (sampled). The agent-service
# emits sampled spans, so we reconstruct the remote parent as sampled too;
# otherwise the attached child could be dropped by the sampler.
_SAMPLED_FLAGS = 0x01


async def run(dataset_path: Path, agent_service_url: str) -> list[dict[str, str]]:
    """Drive the evaluation loop.

    For each row: invoke the agent as a black box, receive the span ids it
    authored, then attach an ``evaluation_context`` child span carrying ground
    truth to that returned span.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
    setup_observability()
    tracer = trace.get_tracer(SERVICE_NAME)

    rows = list(load_dataset(dataset_path))
    results: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for item in rows:
            print(f"\n[{item.id}]")
            print(f"  query          : {item.query}")

            # 1. Invoke the agent as a BLACK BOX -- no traceparent, no context.
            try:
                resp = await client.post(
                    agent_service_url,
                    json={"item_id": item.id, "query": item.query},
                )
                resp.raise_for_status()
                agent_result = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("agent call failed for %s: %s", item.id, exc)
                print(f"  ERROR          : {exc}")
                results.append(
                    {"item_id": item.id, "operation_id": "", "status": "ERROR"}
                )
                continue

            response_text = agent_result["response"]
            agent_trace_id = agent_result["agent_trace_id"]
            agent_span_id = agent_result["agent_span_id"]
            print(f"  agent_span     : trace_id={agent_trace_id} span_id={agent_span_id}")
            print(f"  response       : {response_text}")
            print(f"  ground_truth   : {item.ground_truth}")

            # 2. Reconstruct the agent's span as a REMOTE PARENT from the ids it
            #    returned, then open evaluation_context as its child, using
            #    nothing but the returned (trace_id, span_id).
            traceparent = build_traceparent(
                int(agent_trace_id, 16), int(agent_span_id, 16), _SAMPLED_FLAGS
            )
            parent_ctx = parent_context_from_traceparent(traceparent)

            with tracer.start_as_current_span(
                EVALUATION_CONTEXT_SPAN_NAME, context=parent_ctx
            ) as span:
                span_ctx = span.get_span_context()
                # operation_Id in App Insights == the trace_id. Because we
                # attached inside the agent's trace, this equals the agent's
                # trace_id -- the GT span lives in the SAME trace as the agent.
                operation_id = f"{span_ctx.trace_id:032x}"

                ground_truth_json = json.dumps(item.ground_truth, ensure_ascii=False)
                span.set_attribute("gen_ai.evaluation.item_id", item.id)
                span.set_attribute(GROUND_TRUTH_ATTRIBUTE, ground_truth_json)

                print(
                    f"  eval_context   : trace_id={operation_id} "
                    f"span_id={span_ctx.span_id:016x} "
                    f"(child of agent span {agent_span_id})"
                )

            # Sanity: the attached span must share the agent's trace.
            assert operation_id == agent_trace_id, (
                "evaluation_context is not in the same trace as the agent span"
            )

            results.append(
                {
                    "item_id": item.id,
                    "operation_id": operation_id,
                    "agent_span_id": agent_span_id,
                    "eval_context_span_id": f"{span_ctx.span_id:016x}",
                    "status": "OK",
                }
            )

    return results


def cli() -> None:
    """Parse arguments, load config, and run the driver."""
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
        default=os.environ.get(
            "AGENT_SERVICE_STANDALONE_URL", DEFAULT_AGENT_SERVICE_URL
        ),
        help="URL of the agent-service /invoke-standalone endpoint.",
    )
    args = parser.parse_args()

    results = asyncio.run(run(args.dataset, args.agent_service_url))

    print(
        "\nDone. Agent authored its own span; evaluation_context (with "
        "ground_truth) was ATTACHED as a child of that span in the same trace."
    )

    print("\n=== App Insights operation_Id per item ===")
    print(f"{'item_id':<10} {'status':<7} operation_Id")
    op_ids = []
    for r in results:
        print(f"{r['item_id']:<10} {r['status']:<7} {r['operation_id']}")
        if r["status"] == "OK":
            op_ids.append(r["operation_id"])

    if op_ids:
        joined = ", ".join(f'"{o}"' for o in op_ids)
        print(
            "\nPaste into App Insights Logs (KQL) to see every span from this run:\n"
            "union traces, dependencies, requests, exceptions\n"
            f"| where operation_Id in ({joined})\n"
            "| project timestamp, itemType, name, operation_Id, operation_ParentId, "
            'ground_truth = tostring(customDimensions["gen_ai.evaluation.ground_truth"])\n'
            "| order by timestamp asc"
        )


if __name__ == "__main__":
    cli()
