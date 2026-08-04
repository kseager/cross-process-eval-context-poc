# span-exact-eval-poc

A proof-of-concept that attaches a **ground-truth object** to a span that is a
**span-exact child** of a framework-generated **`invoke_agent` span** — **across
two separate processes** — **without any code changes in the agent process**.

It combines three ideas:

- **[`trace-ground-truth-poc`](https://github.com/singankit/trace-ground-truth-poc)**
  (Ankit): attach the ground-truth object as an `evaluation.ground_truth`
  **event**. That POC does it on its *own in-process* `invoke_agent` span.
- **Two-process reality**: a separate evaluator process **cannot** mutate the
  runner's already-created `invoke_agent` span (spans are immutable once ended,
  and you can't hold another process's live span). So instead the evaluator
  creates a **child span** of that exact `invoke_agent` span and puts the
  ground-truth event on the *child*.
- **Trace-context propagation** (Yingying's eval-results-traces design): the
  runner carries the specific `invoke_agent` span's identity to the evaluator as
  a W3C `traceparent`, so the child-attach is seamless and **span-exact**.

Net result: the ground-truth object lands on a span whose `parentSpanId` is
**exactly** the `invoke_agent` span — queryable in App Insights via the
first-class `operation_ParentId` column.

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

## Requirements

The design must satisfy all of the following. Each is a hard requirement, not a
nice-to-have.

1. **Span-level (not trace-level) correlation.** The ground-truth data must
   attach to **exactly one** `invoke_agent` span, identified by its `span_id` —
   *not* merely to the trace it belongs to.
2. **Two independent processes.** The evaluator runs in a **separate process**
   from the runner that owns the `invoke_agent` span (both alive concurrently).
3. **No agent-process code changes.** The agent/target process must require
   **zero** code changes. Only the runner and evaluator are ours to modify.
4. **Attach a ground-truth object.** Carry a structured ground-truth object
   (not a bare string), mirroring `trace-ground-truth-poc`'s
   `evaluation.ground_truth` event.
5. **Cheap, unambiguous backend query.** Correlation must be resolvable in App
   Insights with a simple, first-class join (no fragile string/JSON parsing).

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
│          "gen_ai.evaluation.input", context=parent_ctx) as s:      │
│      s.add_event("evaluation.ground_truth", {ground_truth: {...}})   │
│      # s.parentSpanId == the invoke_agent span_id  → span-exact      │
└──────────────────────────────────────────────────────────────────────┘
```

Resulting span tree (one `trace_id`):

```
invoke_agent                         (runner, span_id=A)
├─ chat / tool spans                 (agent framework, if instrumented)
└─ gen_ai.evaluation.input           (eval-worker, parentSpanId=A)  ← span-exact
      • event: evaluation.ground_truth { item_id, query, ground_truth }
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

Because the eval span's parent is a first-class column, the join is trivial. The
ground-truth object is on the eval span's `evaluation.ground_truth` event and in
its `gen_ai.evaluation.ground_truth` attribute (JSON):

```kql
dependencies
| where name == "gen_ai.evaluation.input"
| join kind=inner (
    dependencies | where name == "invoke_agent"
) on $left.operation_ParentId == $right.id     // eval.parent == invoke_agent.spanId
| project timestamp, operation_Id,
          invoke_agent_span = id1,
          eval_parent_span = operation_ParentId,
          ground_truth = tostring(customDimensions["gen_ai.evaluation.ground_truth"])
```

`operation_ParentId` (parentSpanId) equals the exact `invoke_agent` `id`
(spanId) — span-exact correlation across the two processes.

## Notes / trade-offs

- The eval span is a **child of `invoke_agent`**, so its timestamps fall *after*
  the parent closed and it counts toward that span's subtree. This is the
  deliberate trade for the cheapest, most precise KQL (first-class parent id).
  A peer **span link** or an explicit `evaluated_span_id` attribute are
  alternatives if you want peer semantics instead of nesting.
- The ground-truth object is attached both as an **event**
  (`evaluation.ground_truth`, mirroring `trace-ground-truth-poc`) and as a JSON
  **attribute** (`gen_ai.evaluation.ground_truth`) on the eval span. OTel
  attribute/event values must be primitives, so structured objects travel as a
  JSON string.
- The scorer in `eval_worker._score` is a trivial exact-match stand-in; swap in
  a real evaluator without changing the correlation mechanism.
- If you also want the agent's *internal* spans in the same trace, enable OTel
  auto-instrumentation on the agent via **config** (still no source changes).
