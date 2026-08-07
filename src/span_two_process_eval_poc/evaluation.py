"""Trace-based evaluation -- the **post-processing step** of the POC.

After the driver has run every dataset row (each producing one App Insights
trace whose ``operation_Id`` == the OTel ``trace_id``), this module evaluates
those traces **by trace id** using Azure AI Foundry's built-in evaluators via
the OpenAI-compatible ``evals`` API.

This mirrors the SDK sample ``sample_evaluations_builtin_with_traces.py``:

- ``data_source_config`` = ``{"type": "azure_ai_source", "scenario": "traces"}``
- ``data_source``        = ``{"type": "azure_ai_traces", "trace_ids": [...]}``

The eval service pulls the ``query``/``response`` off each trace's agent spans
and exposes them to evaluators as ``{{sample.query}}`` / ``{{sample.response}}``.

Two evaluators are wired on purpose, to demonstrate the ground-truth gap this
POC exists to surface:

1. **No ground truth** -- ``builtin.coherence``. Scores the response using only
   ``query`` + ``response`` pulled from the trace. This is **expected to pass**.
2. **Requires ground truth** -- ``builtin.f1_score``. Needs a ``ground_truth``
   input. Ground truth is stamped as the custom attribute
   ``gen_ai.evaluation.ground_truth`` on the driver-authored
   ``evaluation_context`` span, but the ``azure_ai_traces`` sample schema only
   exposes ``sample.query`` / ``sample.response`` / ``sample.tool_definitions``
   -- it does **not** surface that custom attribute. So this criterion has no
   ``ground_truth`` to read and is **expected to fail**.

   That failure is the whole point: it proves that today the eval service cannot
   consume ground truth stamped as a trace attribute. Making the GT attribute
   flow into ``sample.ground_truth`` is the next piece of work (the KQL / trace
   mapping change).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

logger = logging.getLogger("evaluation")

# Evaluator that needs NO ground truth: judges the response from query+response.
NON_GT_EVALUATOR_NAME = "coherence"
NON_GT_EVALUATOR_ID = "builtin.coherence"

# Evaluator that REQUIRES ground truth. Mapped to {{item.ground_truth}}, which
# is now surfaced from the trace's gen_ai.evaluation.ground_truth attribute by
# the RAISvc ground-truth KQL lift. We use the model-graded `similarity`
# evaluator (not the lexical `f1_score`): similarity declares `ground_truth` in
# its data_schema (so the mapping is not stripped) AND tolerates the message-list
# shape that trace `response` arrives in -- the lexical evaluators crash on it
# with `'list' object has no attribute 'lower'`.
GT_EVALUATOR_NAME = "similarity"
GT_EVALUATOR_ID = "builtin.similarity"

# Terminal states of an eval run.
_TERMINAL_STATES = {"completed", "failed", "canceled"}


@dataclass
class CriterionOutcome:
    """One evaluator's aggregate outcome across the evaluated traces."""

    name: str
    passed: int
    errored: int
    total: int

    @property
    def all_errored(self) -> bool:
        return self.total > 0 and self.errored == self.total

    @property
    def any_passed(self) -> bool:
        return self.passed > 0


@dataclass
class EvaluationSummary:
    """Result of an evaluation run, keyed by criterion name."""

    eval_id: str
    run_id: str
    status: str
    criteria: dict[str, CriterionOutcome]
    total_items: int = 0
    run_error: str | None = None

    @property
    def infra_failed(self) -> bool:
        """True if the run failed at the service level (no items were scored).

        Distinguishes an infrastructure/service failure (e.g. the Foundry
        workspace identity being unavailable) from a legitimate per-criterion
        outcome. In the infra-failed case no evaluator actually ran, so the
        ground-truth expectation check is not meaningful.
        """
        return self.status != "completed" and self.total_items == 0


def _build_evaluator_config(
    name: str, evaluator_name: str, model_deployment_name: str, *, with_ground_truth: bool, needs_model: bool
) -> dict[str, Any]:
    """Build one ``azure_ai_evaluator`` testing-criterion block.

    ``needs_model`` controls whether a model deployment is attached via
    ``initialization_parameters``. Model-graded evaluators (e.g. ``coherence``)
    require a deployment; lexical evaluators (e.g. ``f1_score``) do not accept a
    model and error if one is supplied, so their criterion omits it.

    For the ground-truth evaluator ``ground_truth`` is mapped to
    ``{{item.ground_truth}}`` -- the field surfaced from the trace's
    ``gen_ai.evaluation.ground_truth`` attribute when present.
    """
    data_mapping: dict[str, str] = {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
    }
    if with_ground_truth:
        data_mapping["ground_truth"] = "{{item.ground_truth}}"

    config: dict[str, Any] = {
        "type": "azure_ai_evaluator",
        "name": name,
        "evaluator_name": evaluator_name,
        "data_mapping": data_mapping,
    }
    if needs_model:
        config["initialization_parameters"] = {"deployment_name": model_deployment_name}
    return config


def evaluate_traces(
    trace_ids: list[str],
    *,
    lookback_hours: int = 1,
    poll_seconds: int = 5,
    timeout_seconds: int = 600,
) -> EvaluationSummary:
    """Evaluate the given App Insights traces by id.

    Runs two built-in evaluators over the traces: one that needs no ground truth
    (``builtin.coherence``) and one that requires it (``builtin.f1_score``). The
    ground-truth evaluator is expected to fail because the trace scenario does
    not expose our custom ``gen_ai.evaluation.ground_truth`` attribute.

    Args:
        trace_ids: App Insights ``operation_Id``s (== OTel ``trace_id``s) to
            evaluate. Typically the successful rows from the driver run.
        lookback_hours: Trace query lookback window passed to the eval service.
        poll_seconds: Delay between run-status polls.
        timeout_seconds: Give up waiting after this many seconds.

    Returns:
        An :class:`EvaluationSummary` with per-criterion pass/error counts.

    Raises:
        RuntimeError: If ``FOUNDRY_PROJECT_ENDPOINT`` or the model deployment
            name is not set, or if no trace ids are provided.
    """
    if not trace_ids:
        raise RuntimeError("No trace ids to evaluate.")

    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    if not endpoint:
        raise RuntimeError("FOUNDRY_PROJECT_ENDPOINT is not set.")

    model_deployment_name = os.environ.get(
        "AZURE_AI_MODEL_DEPLOYMENT_NAME"
    ) or os.environ.get("FOUNDRY_MODEL_NAME")
    if not model_deployment_name:
        raise RuntimeError(
            "Model deployment name is not set "
            "(AZURE_AI_MODEL_DEPLOYMENT_NAME or FOUNDRY_MODEL_NAME)."
        )

    testing_criteria = [
        _build_evaluator_config(
            NON_GT_EVALUATOR_NAME,
            NON_GT_EVALUATOR_ID,
            model_deployment_name,
            with_ground_truth=False,
            needs_model=True,
        ),
        _build_evaluator_config(
            GT_EVALUATOR_NAME,
            GT_EVALUATOR_ID,
            model_deployment_name,
            with_ground_truth=True,
            needs_model=True,
        ),
    ]

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as client,
    ):
        eval_object = client.evals.create(
            name="span_two_process_trace_eval",
            # Custom item_schema that DECLARES ground_truth (plus query/response/
            # context). This is what keeps the ground_truth data_mapping from
            # being stripped: the eval service drops any mapped field the schema
            # does not declare. The auto-derived `azure_ai_source`/`traces`
            # schema omits ground_truth, so mappings to it were silently removed.
            data_source_config={
                "type": "custom",
                "include_sample_schema": False,
                "item_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "response": {"type": "string"},
                        "context": {"type": "string"},
                        "ground_truth": {"type": "string"},
                    },
                    "required": ["query", "response", "ground_truth"],
                },
            },
            testing_criteria=testing_criteria,  # type: ignore[arg-type]
        )
        logger.info("evaluation created (id=%s)", eval_object.id)

        data_source = {
            "type": "azure_ai_traces_preview",
            "trace_ids": list(trace_ids),
            "lookback_hours": lookback_hours,
        }
        run = client.evals.runs.create(
            eval_id=eval_object.id,
            name="span_two_process_trace_run",
            data_source=data_source,  # type: ignore[arg-type]
        )
        logger.info("evaluation run created (id=%s)", run.id)

        deadline = time.monotonic() + timeout_seconds
        while run.status not in _TERMINAL_STATES:
            if time.monotonic() > deadline:
                logger.warning("eval run %s timed out in status %s", run.id, run.status)
                break
            time.sleep(poll_seconds)
            run = client.evals.runs.retrieve(run_id=run.id, eval_id=eval_object.id)
            logger.info("eval run status: %s", run.status)

        criteria, total_items = _summarize_output_items(
            client, eval_object.id, run.id
        )

        run_error = _get(run, "error")

        return EvaluationSummary(
            eval_id=eval_object.id,
            run_id=run.id,
            status=str(run.status),
            criteria=criteria,
            total_items=total_items,
            run_error=str(run_error) if run_error else None,
        )


def _summarize_output_items(
    client: Any, eval_id: str, run_id: str
) -> tuple[dict[str, CriterionOutcome], int]:
    """Aggregate per-criterion pass/error counts from the run's output items.

    Returns a ``(criteria, total_items)`` tuple; ``total_items`` is the number
    of output items the run produced (0 signals nothing was scored).
    """
    counts: dict[str, dict[str, int]] = {}
    try:
        output_items = list(client.evals.runs.output_items.list(run_id=run_id, eval_id=eval_id))
    except Exception as exc:  # pragma: no cover - service/transport variance
        logger.warning("could not list output items: %s", exc)
        return {}, 0

    for item in output_items:
        for res in getattr(item, "results", None) or []:
            # results entries are dict-like across SDK versions.
            name = _get(res, "name") or _get(res, "evaluator_name") or "unknown"
            bucket = counts.setdefault(name, {"passed": 0, "errored": 0, "total": 0})
            bucket["total"] += 1
            if _is_errored(res):
                bucket["errored"] += 1
            elif _passed(res):
                bucket["passed"] += 1

    return (
        {
            name: CriterionOutcome(
                name=name, passed=c["passed"], errored=c["errored"], total=c["total"]
            )
            for name, c in counts.items()
        },
        len(output_items),
    )


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _passed(res: Any) -> bool:
    val = _get(res, "passed")
    return bool(val)


def _is_errored(res: Any) -> bool:
    """A result is errored if it carries an error/sample-error payload."""
    if _get(res, "error"):
        return True
    sample = _get(res, "sample")
    if sample is not None and _get(sample, "error"):
        return True
    return False


def check_evaluation_results(summary: EvaluationSummary) -> bool:
    """Check the eval outcome against the POC's expectations.

    Expectations (post ground-truth-lift):
      * The **non-GT** evaluator (``coherence``) should produce at least one
        passing result -- it works purely from ``query`` + ``response`` on the
        trace.
      * The **GT-requiring** evaluator (``similarity``) should now **score** at
        least one trace, because the RAISvc ground-truth KQL lift surfaces our
        custom ``gen_ai.evaluation.ground_truth`` attribute as
        ``sample.ground_truth`` and the custom item_schema keeps the mapping.

    Returns:
        ``True`` if reality matches those expectations (non-GT passed AND GT
        evaluator scored), else ``False``. Prints a readable report either way.
    """
    non_gt = summary.criteria.get(NON_GT_EVALUATOR_NAME)
    gt = summary.criteria.get(GT_EVALUATOR_NAME)

    print("\n=== Evaluation results ===")
    print(f"eval_id : {summary.eval_id}")
    print(f"run_id  : {summary.run_id}")
    print(f"status  : {summary.status}")
    print(f"items   : {summary.total_items}")
    for name, c in summary.criteria.items():
        print(
            f"  {name:<12} passed={c.passed} errored={c.errored} total={c.total}"
        )

    # Distinguish a service/infrastructure failure (nothing was scored) from a
    # genuine per-criterion outcome.
    if summary.infra_failed:
        print("\n--- Run did NOT produce results (infrastructure failure) ---")
        if summary.run_error:
            print(f"  run error: {summary.run_error[:500]}")
        print(
            "\n[INCONCLUSIVE] The eval service failed before scoring any trace "
            "(0 output items), so the ground-truth flow could not be tested. "
            "This is a backend/resource problem, not a POC bug -- re-run "
            "against a healthy Foundry project."
        )
        return False

    non_gt_ok = non_gt is not None and non_gt.any_passed
    # GT evaluator is now expected to SCORE: it must have run without erroring
    # on every trace and produced at least one passing result.
    gt_scored = (
        gt is not None and gt.total > 0 and gt.any_passed and not gt.all_errored
    )

    print("\n--- Expectation check ---")
    print(
        f"  non-GT '{NON_GT_EVALUATOR_NAME}' produced a passing score : "
        f"{'YES (expected)' if non_gt_ok else 'NO (UNEXPECTED)'}"
    )
    print(
        f"  GT '{GT_EVALUATOR_NAME}' scored (ground_truth surfaced) : "
        f"{'YES (expected)' if gt_scored else 'NO (UNEXPECTED)'}"
    )

    meets_expectations = non_gt_ok and gt_scored
    if meets_expectations:
        print(
            "\n[OK] E2E matches expectations: coherence scored the trace, and the "
            "ground-truth evaluator scored because the RAISvc lift surfaced "
            "gen_ai.evaluation.ground_truth as sample.ground_truth."
        )
    else:
        print(
            "\n[FAIL] E2E did NOT match expectations -- inspect the eval run output "
            "items above. (Either coherence did not score, or the GT evaluator "
            "did not score -- check that ground_truth reached the evaluator.)"
        )
    return meets_expectations
