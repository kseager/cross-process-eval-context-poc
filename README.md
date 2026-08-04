# span-exact-eval-poc

A proof-of-concept that attaches an **evaluation result** to **exactly one**
framework-generated **`invoke_agent` span** — **across two separate processes** —
**without any code changes in the agent process**.

It is the span-exact, cross-process evolution of
[`trace-ground-truth-poc`](https://github.com/singankit/trace-ground-truth-poc):
where that POC stamped ground truth as an **event on its own in-process
`invoke_agent` span**, this POC creates the evaluation as a **child span of a
specific `invoke_agent` span living in another process**, correlated by the W3C
`traceparent` (`trace_id` + `span_id`).

## Why this design

| Concern | `trace-ground-truth-poc` | this POC |
|---|---|---|
| Processes | Single | **Two** (runner + eval-worker) |
| Attaches to | An **event** on the agent span | A **child span** of the agent span |
| Correlation grain | Same live span (in-process) | **Span-exact** across processes (`operation_ParentId`) |
| Agent code changes | n/a (single process) | **None** — runner owns `invoke_agent`, injects context |

Trace-level correlation (shared `trace_id`) only says "same run"; if a trace has
more than one agent invocation it can't say *which* one an eval belongs to. This
POC binds the eval to the **specific `invoke_agent` `span_id`**, so correlation
is unambiguous and queryable via the first-class `operation_ParentId` column.

## How it works

```
┌──────────────── runner process (owns invoke_agent) ────────────────┐
│  with tracer.start_as_current_span("invoke_agent") as span:        │
│      result = await agent.run(query)        # agent = no changes    │
│      tp = traceparent_for_span(span)        # 00-<trace>-<span>-01   │
│      POST /evaluate { query, response, ground_truth, tp } ─────────┐│
└────────────────────────────────────────────────────────────────────┘│
                                                                       ▼
┌──────────────── eval-worker process (separate) ─────────────────────┐
│  parent_ctx = parent_context_from_traceparent(tp)  # remote parent   │
│  with tracer.start_as_current_span(                                  │
│          "gen_ai.evaluation.results", context=parent_ctx) as s:      │
│      s.set_attribute("gen_ai.evaluation.score", score)              │
│      # s.parentSpanId == the invoke_agent span_id  → span-exact      │
└──────────────────────────────────────────────────────────────────────┘
```

Resulting span tree (one `trace_id`):

```
invoke_agent                         (runner, span_id=A)
├─ chat / tool spans                 (agent framework, if instrumented)
└─ gen_ai.evaluation.results         (eval-worker, parentSpanId=A)  ← span-exact
```

The **agent process is never involved in correlation** — the runner creates the
`invoke_agent` span, captures its `span_id`, and hands that exact id to the
eval-worker over HTTP.

## Layout

```
data/dataset.jsonl                         # query + ground_truth rows
src/span_exact_eval_poc/
  dataset.py                               # JSONL loader
  telemetry.py                             # Azure Monitor setup (both processes)
  trace_context.py                         # build/parse traceparent (the core)
  agent.py                                 # Foundry agent (no POC-specific code)
  eval_worker.py                           # FastAPI /evaluate — creates child span
  runner.py                                # owns invoke_agent, dispatches eval
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A Microsoft Foundry project with a model deployment
- An Application Insights resource (connection string)
- `az login`

## Setup

```bash
cp .env.example .env   # fill in the values
uv sync
```

## Run (two processes)

Terminal 1 — the eval-worker:

```bash
uv run uvicorn span_exact_eval_poc.eval_worker:app --port 8001
```

Terminal 2 — the runner:

```bash
uv run run-poc
# or: uv run run-poc --dataset path/to/other.jsonl --eval-worker-url http://localhost:8001/evaluate
```

The runner prints, per row, the `invoke_agent` `trace_id`/`span_id` and the eval
span's `parent_span_id`, and **asserts** they match (span-exact, same trace).

## Verify in App Insights (KQL)

Because the eval span's parent is a first-class column, the join is trivial:

```kql
dependencies
| where name == "gen_ai.evaluation.results"
| join kind=inner (
    dependencies | where name == "invoke_agent"
) on $left.operation_ParentId == $right.id     // eval.parent == invoke_agent.spanId
| project timestamp, operation_Id,
          invoke_agent_span = id1,
          eval_parent_span = operation_ParentId,
          score = todouble(customDimensions["gen_ai.evaluation.score"])
```

`operation_ParentId` (parentSpanId) equals the exact `invoke_agent` `id`
(spanId) — span-exact correlation across the two processes.

## Notes / trade-offs

- The eval span is a **child of `invoke_agent`**, so its timestamps fall *after*
  the parent closed and it counts toward that span's subtree. This is the
  deliberate trade for the cheapest, most precise KQL (first-class parent id).
  A peer **span link** or an explicit `evaluated_span_id` attribute are
  alternatives if you want peer semantics instead of nesting.
- The scorer in `eval_worker._score` is a trivial exact-match stand-in; swap in
  a real evaluator without changing the correlation mechanism.
- If you also want the agent's *internal* spans in the same trace, enable OTel
  auto-instrumentation on the agent via **config** (still no source changes).
