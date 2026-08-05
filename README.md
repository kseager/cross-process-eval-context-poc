# span-two-process-eval-poc

A proof-of-concept that attaches a **ground-truth object** to a span that is a
**span-exact child** of a runner-authored **`invoke_agent` span** — **across
three separate processes** — **without any tracing code in the agent process**.
It mirrors the **ACA topology** (Yingying's design): a *remote* agent service,
a runner that authors `invoke_agent`, and a separate ground-truth attach worker.

It combines three ideas:

- **[`trace-ground-truth-poc`](https://github.com/singankit/trace-ground-truth-poc)**
  (Ankit): attach the ground-truth object as an `evaluation.ground_truth`
  **event**. That POC does it on its *own in-process* `invoke_agent` span.
- **Remote-agent reality (ACA-faithful)**: the runner **authors** the
  `invoke_agent` span and calls the agent as a **separate HTTP service**,
  injecting a W3C `traceparent` header. The agent is a *passive header
  recipient* — it opens `execute_agent` under `invoke_agent` and needs **no**
  tracing code. It never emits `invoke_agent`, so there is no duplicate span.
- **Trace-context propagation** (Yingying's eval-results-traces design): a
  separate eval-worker **cannot** mutate the `invoke_agent` span (spans are
  immutable and cross-process), so it creates a **child span** of that exact
  span — using the same `traceparent` — and attaches the ground truth there.

Net result: the ground-truth object lands on a span whose `parentSpanId` is
**exactly** the `invoke_agent` span — queryable in App Insights via the
first-class `operation_ParentId` column.

## Why this design

| Concern | `trace-ground-truth-poc` | this POC |
|---|---|---|
| Processes | Single | **Three** (agent-service + runner + eval-worker) |
| Attaches to | An **event** on the agent span | A **child span** of the agent span |
| Correlation grain | Same live span (in-process) | **Span-exact** across processes (`operation_ParentId`) |
| Agent code changes | n/a (single process) | **None** — runner authors `invoke_agent` and injects `traceparent`; the remote agent just receives the header |

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
2. **Independent processes.** The agent runs as a **separate service**, and the
   eval-worker runs in **another separate process** from the runner that authors
   the `invoke_agent` span (all alive concurrently).
3. **No agent-process code changes.** The agent/target process must require
   **zero** tracing code — the runner authors `invoke_agent` and injects a
   `traceparent` header; the remote agent merely opens `execute_agent` under it.
   Only the runner and eval-worker are ours to instrument.
4. **Attach a ground-truth object only.** Carry a structured ground-truth object
   (not a bare string), mirroring `trace-ground-truth-poc`'s
   `evaluation.ground_truth` event. **No scores or results** are attached here —
   only evaluation *input*.
5. **Cheap, unambiguous backend query.** Correlation must be resolvable in App
   Insights with a simple, first-class join (no fragile string/JSON parsing).

## How it works

```
┌──────────────── runner process ───────────────────────────────────┐
│  with tracer.start_as_current_span("invoke_agent") as span:        │
│      tp = traceparent_for_span(span)          # 00-<trace>-<span>-01 │
│      # 1) call the REMOTE agent, injecting tp as a header           │
│      POST {AGENT}/invoke  headers={traceparent: tp}  { query } ───┐ │
│      # 2) attach ground truth to the SAME invoke_agent span         ││
│      POST {EVAL}/evaluate { item_id, query, ground_truth, tp } ──┐ ││
└──────────────────────────────────────────────────────────────────┼─┼┘
                          ▼ (1)                              ▼ (2)  │ │
┌──────── agent-service (separate) ───────┐ ┌──── eval-worker (separate) ──────┐
│ parent = ctx_from(traceparent header)   │ │ parent = ctx_from(tp in body)     │
│ with start_span("execute_agent",        │ │ with start_span(                  │
│        context=parent):                 │ │     "gen_ai.evaluation.input",    │
│     result = await agent.run(query)     │ │      context=parent) as s:        │
│ # execute_agent nests UNDER invoke_agent│ │   s.add_event(                    │
│ # agent has NO tracing code             │ │     "evaluation.ground_truth",…)  │
└─────────────────────────────────────────┘ │   # parentSpanId == invoke_agent  │
                                             │   # ground truth ONLY, no scoring │
                                             └───────────────────────────────────┘
```

Resulting span tree (one `trace_id`):

```
invoke_agent                         (runner authors, span_id=A)
├─ execute_agent                     (agent-service, via traceparent header)
│   └─ chat / tool spans             (agent framework, if instrumented)
└─ gen_ai.evaluation.input           (eval-worker, parentSpanId=A)  ← span-exact
      • event: evaluation.ground_truth { item_id, query, ground_truth }
```

The **agent process is never involved in correlation** — the *runner* authors
the `invoke_agent` span, injects its `traceparent` into the remote agent call
(so `execute_agent` nests under it) and into the eval-worker call (so the
ground-truth span attaches to the exact same `span_id`). This matches ACA, where
the agent is remote and emits `execute_agent`, not `invoke_agent`.

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
  telemetry.py                             # Azure Monitor setup (all three processes)
  trace_context.py                         # build/parse traceparent (the core)
  agent.py                                 # Foundry agent builder (no POC-specific code)
  agent_service.py                         # FastAPI /invoke — remote agent, opens execute_agent
  eval_worker.py                           # FastAPI /evaluate — attaches ground truth to child span
  runner.py                                # authors invoke_agent, calls agent + eval-worker
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

## Run (three processes)

Terminal 1 — the agent-service (the remote agent):

```bash
uv run uvicorn span_two_process_eval_poc.agent_service:app --port 8002
```

Terminal 2 — the eval-worker:

```bash
uv run uvicorn span_two_process_eval_poc.eval_worker:app --port 8001
```

Terminal 3 — the runner:

```bash
uv run run-poc
# or: uv run run-poc --dataset path/to/other.jsonl \
#       --agent-service-url http://localhost:8002/invoke \
#       --eval-worker-url http://localhost:8001/evaluate
```

The runner prints, per row, the `invoke_agent` `trace_id`/`span_id`, the
`execute_agent` span id (child, from the agent-service), and the ground-truth
span's `parent_span_id`, and **asserts** they all correlate (span-exact, same
trace).

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
- The runner **authors** the `invoke_agent` span; the agent is a **remote
  service** that emits `execute_agent` (not `invoke_agent`) beneath it via the
  propagated `traceparent` header. Because there is exactly one `invoke_agent`
  (the runner's) and the agent never emits its own, binding the ground truth to
  it is authoritative — not a fabricated/duplicate span. This matches ACA.
- Evaluation/scoring is **not** performed here — only ground-truth *input* is
  attached. Scoring is a separate **post-processing** step run after all agent
  invocations complete (read the spans back from App Insights and compute
  metrics). This keeps the `gen_ai.evaluation.input` span faithful to its name.
- If you also want the agent's *internal* spans in the same trace, enable OTel
  auto-instrumentation on the agent via **config** (still no source changes).
