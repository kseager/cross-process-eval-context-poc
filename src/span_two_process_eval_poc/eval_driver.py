"""Driver that attaches evaluation ground truth to an agent's own trace.

The agent runs first as a black box. It authors
its own ``invoke_agent`` span and returns that span's ``(trace_id, span_id)``.
The driver then emits a ``gen_ai.evaluation.context`` event stamped with those
ids, correlating ground truth to the agent's span in the same trace
(``operation_ParentId`` == the agent's span) without mutating it. Optionally,
``--evaluate`` runs a trace-id evaluation post-processing step over the run.
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
from opentelemetry._events import Event, get_event_logger

from .dataset import load_dataset
from .telemetry import setup_observability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-driver")

SERVICE_NAME = "eval-driver"

# Cross-process contract: the attribute the eval service lifts ground truth from.
GROUND_TRUTH_ATTRIBUTE = "gen_ai.evaluation.ground_truth"

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "dataset.jsonl"
DEFAULT_AGENT_SERVICE_URL = "http://localhost:8002/invoke-standalone"

_SAMPLED_FLAGS = 0x01


async def run(dataset_path: Path, agent_service_url: str) -> list[dict[str, str]]:
    """Drive the evaluation loop.

    For each row: invoke the agent as a black box, receive the span ids it
    authored, then emit a ``gen_ai.evaluation.context`` OTel event stamped with
    that span's ``(trace_id, span_id)``, carrying ground truth. The event is a
    log record correlated to the agent's ``invoke_agent`` span
    (``operation_ParentId`` == the agent span id) in the same trace -- no child
    span, no span mutation.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
    setup_observability()
    event_logger = get_event_logger(SERVICE_NAME)

    rows = list(load_dataset(dataset_path))
    results: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for item in rows:
            print(f"\n[{item.id}]")
            print(f"  query          : {item.query}")

            # Invoke the agent as a black box.
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

            # Emit a gen_ai.evaluation.context event stamped with the agent's
            # (trace_id, span_id): a log record correlated to the agent's own
            # invoke_agent span (operation_ParentId == agent span id) in the
            # same trace, without mutating the already-ended span.
            operation_id = agent_trace_id

            ground_truth_json = json.dumps(item.ground_truth, ensure_ascii=False)
            evaluation_event = Event(
                name="gen_ai.evaluation.context",
                attributes={
                    "gen_ai.evaluation.item_id": item.id,
                    GROUND_TRUTH_ATTRIBUTE: ground_truth_json,
                },
                trace_id=int(agent_trace_id, 16),
                span_id=int(agent_span_id, 16),
                trace_flags=trace.TraceFlags(_SAMPLED_FLAGS),
            )
            event_logger.emit(evaluation_event)

            print(
                f"  eval_event     : name=gen_ai.evaluation.context "
                f"trace_id={operation_id} "
                f"(parent=agent span {agent_span_id})"
            )

            assert operation_id == agent_trace_id, (
                "evaluation event is not stamped into the agent's trace"
            )

            results.append(
                {
                    "item_id": item.id,
                    "operation_id": operation_id,
                    "agent_span_id": agent_span_id,
                    "eval_event_parent_span_id": agent_span_id,
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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=os.environ.get("RUN_EVALUATION") == "1",
        help=(
            "After attaching ground truth, run the trace-id evaluation "
            "post-processing step (builtin.coherence + builtin.f1_score) over "
            "the run's traces. Off by default; also enabled via RUN_EVALUATION=1."
        ),
    )
    args = parser.parse_args()

    results = asyncio.run(run(args.dataset, args.agent_service_url))

    print(
        "\nDone. Agent authored its own span; a gen_ai.evaluation.context event "
        "(with ground_truth) was EMITTED stamped with that span's ids, in the "
        "same trace (operation_ParentId == the agent span)."
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

    if not args.evaluate:
        print(
            "\nEvaluation step skipped (pass --evaluate or set RUN_EVALUATION=1 "
            "to enable)."
        )
        return

    if not op_ids:
        print("\nNo successful traces to evaluate; skipping evaluation step.")
        return

    from .evaluation import check_evaluation_results, evaluate_traces

    print(f"\n=== Running trace-id evaluation over {len(op_ids)} trace(s) ===")
    summary = evaluate_traces(op_ids)
    check_evaluation_results(summary)


if __name__ == "__main__":
    cli()
