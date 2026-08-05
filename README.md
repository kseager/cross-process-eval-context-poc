# span-two-process-eval-poc

A proof-of-concept that attaches a **ground-truth object** directly to an
**`invoke_agent` span** — with the agent running as a **separate process** —
**without any tracing code in the agent process**. It mirrors the **ACA
topology** (Yingying's design): the driver authors `invoke_agent`, calls a
*remote* agent service, and stamps the ground truth onto that span.

It combines two ideas:

- **[`trace-ground-truth-poc`](https://github.com/singankit/trace-ground-truth-poc)**
  (Ankit): attach the ground-truth object to the `invoke_agent` span. This POC
  attaches it as a **JSON attribute** (`gen_ai.evaluation.ground_truth`) on that
  span.
- **Remote-agent reality (ACA-faithful)**: the **eval-worker** (the driver)
  **authors** the `invoke_agent` span and calls the agent as a **separate HTTP
  service**, injecting a W3C `traceparent` header. The agent is a *passive
  header recipient* — it opens `execute_agent` under `invoke_agent` and needs
  **no** tracing code. It never emits `invoke_agent`, so there is no duplicate
  span.

Net result: the ground-truth object lands **directly on the `invoke_agent`
span** (the one and only such span, authored by the driver), and the remote
agent's `execute_agent` spans nest underneath it in the same trace.

## Why this design

| Concern | `trace-ground-truth-poc` | this POC |
|---|---|---|
| Processes | Single | **Two** (agent-service + eval-worker/driver) |
| Agent location | In-process | **Remote HTTP service** |
| Ground truth attached to | The `invoke_agent` span | The `invoke_agent` span (JSON **attribute**) |
| Agent code changes | n/a (single process) | **None** — the driver authors `invoke_agent` and injects `traceparent`; the remote agent just receives the header |

The `invoke_agent` span is authored by the driver and is the single,
authoritative agent-invocation span. The ground truth is set directly on it, and
the remote agent's `execute_agent` span nests under it (via the propagated
`traceparent`), so the whole invocation — request, agent work, and ground truth
— lives in one coherent, unambiguous span subtree.

## Requirements

The design must satisfy all of the following. Each is a hard requirement, not a
nice-to-have.

1. **Ground truth on the exact `invoke_agent` span.** The ground-truth object
   must be attached to **the** `invoke_agent` span for the row, identified by its
   `span_id`.
2. **Separate agent process.** The agent runs as a **separate HTTP service**
   from the driver that authors the `invoke_agent` span (both alive
   concurrently).
3. **No agent-process code changes.** The agent/target process must require
   **zero** tracing code — the driver authors `invoke_agent` and injects a
   `traceparent` header; the remote agent merely opens `execute_agent` under it.
   Only the driver is ours to instrument.
4. **Attach a ground-truth object only.** Carry a structured ground-truth object
   (not a bare string). **No scores or results** are attached here — only
   evaluation *input*.
5. **Cheap, unambiguous backend query.** Correlation must be resolvable in App
   Insights with a simple, first-class join (no fragile string/JSON parsing).

## How it works

```
┌──────────────── eval-worker process (the driver) ──────────────────┐
│  for each row:                                                       │
│    with tracer.start_as_current_span("invoke_agent") as span:       │
│        tp = traceparent_for_span(span)        # 00-<trace>-<span>-01 │
│        # call the REMOTE agent, injecting tp as a header            │
│        POST {AGENT}/invoke headers={traceparent: tp} { query } ───┐ │
│        # stamp ground truth DIRECTLY on this invoke_agent span      ││
│        span.set_attribute(                                          ││
│            "gen_ai.evaluation.ground_truth", json({...}))           ││
└───────────────────────────────────────────────────────────────────┼─┘
                                                              ▼        │
┌──────── agent-service (separate process) ────────────────────────┐  │
│ parent = ctx_from(traceparent header)                             │  │
│ with start_span("execute_agent", context=parent):                 │  │
│     result = await agent.run(query)                               │  │
│ # execute_agent nests UNDER invoke_agent; agent has NO tracing    │  │
└───────────────────────────────────────────────────────────────────┘ │
```

Resulting span tree (one `trace_id`):

```
invoke_agent                         (eval-worker/driver authors, span_id=A)
│  • attribute: gen_ai.evaluation.ground_truth = {...}   ← ground truth here
└─ execute_agent                     (agent-service, via traceparent header)
    └─ chat / tool spans             (agent framework, if instrumented)
```

The **agent process is never involved in correlation** — the *driver* authors
the `invoke_agent` span, injects its `traceparent` into the remote agent call
(so `execute_agent` nests under it), and sets the ground truth as an attribute
on that same span. This matches ACA, where the agent is remote and emits
`execute_agent`, not `invoke_agent`.

> **Evaluation is a separate post-processing step.** The driver only *attaches
> ground-truth input* to the `invoke_agent` span. Scoring the agent's responses
> against this ground truth happens **later, after all agent invocations
> complete** — e.g. a batch job that reads these spans back from App Insights and
> computes metrics. That step is intentionally out of scope here.

## Layout

```
data/dataset.jsonl                         # query + ground_truth rows
src/span_two_process_eval_poc/
  dataset.py                               # JSONL loader
  telemetry.py                             # Azure Monitor setup (both processes)
  trace_context.py                         # build/parse traceparent (the core)
  agent.py                                 # Foundry agent builder (no POC-specific code)
  agent_service.py                         # FastAPI /invoke — remote agent, opens execute_agent
  eval_worker.py                           # DRIVER: authors invoke_agent, calls agent, stamps ground truth
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

Terminal 1 — the agent-service (the remote agent):

```bash
uv run uvicorn span_two_process_eval_poc.agent_service:app --port 8002
```

Terminal 2 — the eval-worker (the driver loop):

```bash
uv run run-poc
# or: uv run run-poc --dataset path/to/other.jsonl \
#       --agent-service-url http://localhost:8002/invoke
```

The eval-worker prints, per row, the `invoke_agent` `trace_id`/`span_id` and the
`execute_agent` span id (child, from the agent-service), and **asserts** the
`execute_agent` span is in the same trace as `invoke_agent`.

## Verify in App Insights (KQL)

The ground truth is a JSON **attribute on the `invoke_agent` span itself**, so no
join is needed — query the span directly. The remote agent's `execute_agent`
span nests under it via `operation_ParentId`:

```kql
dependencies
| where name == "invoke_agent"
| project timestamp, operation_Id,
          invoke_agent_span = id,
          ground_truth = tostring(customDimensions["gen_ai.evaluation.ground_truth"])
// The remote agent's execute_agent span nests under invoke_agent:
//   dependencies | where name == "execute_agent"
//   | where operation_ParentId == <invoke_agent id>
```

## Notes / trade-offs

- The ground-truth object is set **directly as a JSON attribute**
  (`gen_ai.evaluation.ground_truth`) on the `invoke_agent` span the driver
  authors — there is no separate eval/child span. OTel attribute values must be
  primitives, so the structured object travels as a JSON string.
- The driver **authors** the `invoke_agent` span; the agent is a **remote
  service** that emits `execute_agent` (not `invoke_agent`) beneath it via the
  propagated `traceparent` header. Because there is exactly one `invoke_agent`
  (the driver's) and the agent never emits its own, stamping the ground truth on
  it is authoritative — not a fabricated/duplicate span. This matches ACA.
- Evaluation/scoring is **not** performed here — only ground-truth *input* is
  attached. Scoring is a separate **post-processing** step run after all agent
  invocations complete (read the spans back from App Insights and compute
  metrics).
- If you also want the agent's *internal* spans in the same trace, enable OTel
  auto-instrumentation on the agent via **config** (still no source changes).
