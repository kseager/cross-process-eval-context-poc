# span-two-process-eval-poc

A proof-of-concept that attaches a **ground-truth object** to a span that is a
**span-exact child** of a framework-generated **`invoke_agent` span** — **across
two separate processes** — **without any code changes in the agent process**.

It combines three ideas:

- **[`trace-ground-truth-poc`](https://github.com/singankit/trace-ground-truth-poc)**
  (Ankit): attach the ground-truth object as an `evaluation.ground_truth`
  **event**. That POC does it on its *own in-process* `invoke_agent` span.
- **Two-process reality**: a separate evaluator process **cannot** mutate the
  framework's already-created `invoke_agent` span (spans are immutable once
  ended, and you can't hold another process's live span). So instead the
  evaluator creates a **child span** of that exact `invoke_agent` span and puts
  the ground-truth event on the *child*.
- **Trace-context propagation** (Yingying's eval-results-traces design): the
  runner captures the specific framework `invoke_agent` span's identity and
  carries it to the evaluator as a W3C `traceparent`, so the child-attach is
  seamless and **span-exact**.

Net result: the ground-truth object lands on a span whose `parentSpanId` is
**exactly** the `invoke_agent` span — queryable in App Insights via the
first-class `operation_ParentId` column.

## Why this design

| Concern | `trace-ground-truth-poc` | this POC |
|---|---|---|
| Processes | Single | **Two** (runner + eval-worker) |
| Attaches to | An **event** on the agent span | A **child span** of the agent span |
| Correlation grain | Same live span (in-process) | **Span-exact** across processes (`operation_ParentId`) |
| Agent code changes | n/a (single process) | **None** — the *framework* creates `invoke_agent`; the runner only captures its id |

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
   from the runner that drives the `invoke_agent` span (both alive concurrently).
3. **No agent-process code changes.** The agent/target process must require
   **zero** code changes — the framework emits `invoke_agent` on its own; the
   runner only *captures* that span's id. Only the runner and evaluator are ours
   to modify.
4. **Attach a ground-truth object only.** Carry a structured ground-truth object
   (not a bare string), mirroring `trace-ground-truth-poc`'s
   `evaluation.ground_truth` event. **No scores or results** are attached here —
   only evaluation *input*.
5. **Cheap, unambiguous backend query.** Correlation must be resolvable in App
   Insights with a simple, first-class join (no fragile string/JSON parsing).

## How it works

```
┌──────────────── runner process ───────────────────────────────────┐
│  install_invoke_agent_capture()   # span processor grabs real span  │
│  with capture_invoke_agent() as cap:                                │
│      result = await agent.run(query)  # framework emits invoke_agent │
│  # cap.span_context == the framework's REAL invoke_agent span        │
│  tp = traceparent_from_span_context(cap.span_context) # 00-<t>-<s>-01│
│  POST /evaluate { item_id, query, ground_truth, tp } ─────────────┐ │
└────────────────────────────────────────────────────────────────────┘│
                                                                       ▼
┌──────────────── eval-worker process (separate) ─────────────────────┐
│  parent_ctx = parent_context_from_traceparent(tp)  # remote parent   │
│  with tracer.start_as_current_span(                                  │
│          "gen_ai.evaluation.input", context=parent_ctx) as s:      │
│      s.add_event("evaluation.ground_truth", {ground_truth: {...}})   │
│      # s.parentSpanId == the invoke_agent span_id  → span-exact      │
│      # ground truth ONLY — no scoring here (see post-processing note) │
└──────────────────────────────────────────────────────────────────────┘
```

Resulting span tree (one `trace_id`):

```
invoke_agent                         (framework, span_id=A)
├─ chat / tool spans                 (agent framework, if instrumented)
└─ gen_ai.evaluation.input           (eval-worker, parentSpanId=A)  ← span-exact
      • event: evaluation.ground_truth { item_id, query, ground_truth }
```

The **agent process is never involved in correlation** — the *framework* creates
the `invoke_agent` span, a span processor in the runner captures its `span_id`,
and the runner hands that exact id to the eval-worker over HTTP.

> **Evaluation is a separate post-processing step.** The eval-worker only
> *attaches ground-truth input* to the correct span. Scoring the agent's
> responses against this ground truth happens **later, after all agent
> invocations complete** — e.g. a batch job that reads these spans back from App
> Insights and computes metrics. That step is intentionally out of scope here.

## Layout

```
data/dataset.jsonl                         # query + ground_truth rows
src/span_two_process_eval_poc/
  dataset.py                               # JSONL loader
  telemetry.py                             # Azure Monitor setup (both processes)
  trace_context.py                         # build/parse traceparent (the core)
  span_capture.py                          # captures the framework's real invoke_agent span
  agent.py                                 # Foundry agent (no POC-specific code)
  eval_worker.py                           # FastAPI /evaluate — attaches ground truth to child span
  runner.py                                # runs agent, captures span, dispatches ground truth
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
uv run uvicorn span_two_process_eval_poc.eval_worker:app --port 8001
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
- The runner captures the framework's **real** `invoke_agent` span (via
  `span_capture.InvokeAgentCaptureProcessor`) rather than fabricating its own.
  This guarantees the ground truth attaches to the *authentic* agent-invocation
  span, not a wrapper that merely shares the name.
- Evaluation/scoring is **not** performed here — only ground-truth *input* is
  attached. Scoring is a separate **post-processing** step run after all agent
  invocations complete (read the spans back from App Insights and compute
  metrics). This keeps the `gen_ai.evaluation.input` span faithful to its name.
- If you also want the agent's *internal* spans in the same trace, enable OTel
  auto-instrumentation on the agent via **config** (still no source changes).
