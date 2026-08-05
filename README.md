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
  service**, injecting a W3C `traceparent` header. The agent-service opens an
  `execute_agent` span beneath it (matching ACA's
  `invoke_agent → execute_agent → chat`), and the agent itself needs **no**
  tracing code of its own.

Net result: the ground-truth object lands **directly on the `invoke_agent`
span** authored by the driver, and the remote agent's framework spans nest
underneath it in the same trace.

## Adopt this with your own agent (BYO agent)

You do **not** need to understand or touch any of the tracing internals
(`invoke_agent`, `execute_agent`, `traceparent`, span processors, telemetry
setup). Treat the agent as a **black box**. The harness wraps it, drives it, and
produces the trace for you.

**The only file you edit is [`src/span_two_process_eval_poc/agent.py`](src/span_two_process_eval_poc/agent.py).**
Replace the body of `build_agent()` so it returns *your* agent. Everything else
stays exactly as shipped.

### Step 1 — plug in your agent

`build_agent()` must return an object with an **async `run(query: str)` method**
that returns the agent's answer (any object; it is stringified). That is the
only contract.

```python
# src/span_two_process_eval_poc/agent.py
def build_agent():
    # Build and return YOUR agent however you normally do.
    # No tracing, no spans, no OpenTelemetry — just your agent.
    return MyAgent(...)          # must expose:  async def run(self, query: str)
```

If your agent already uses the Microsoft Agent Framework, you can keep the
shipped implementation and only change the model/instructions. If it is a
LangChain / custom / HTTP agent, wrap it in a tiny class:

```python
class MyAgentAdapter:
    def __init__(self, my_agent):
        self._agent = my_agent
    async def run(self, query: str) -> str:
        return await self._agent.ainvoke(query)   # adapt to your API
```

### Step 2 — configure `.env`

```
cp .env.example .env
```

Fill in your model endpoint / deployment and your **Application Insights
connection string** (where traces are sent). Nothing here is about tracing
mechanics — just credentials and the destination.

### Step 3 — run it (two processes)

```bash
uv sync

# 1. start the agent host (wraps your build_agent())
uv run uvicorn span_two_process_eval_poc.agent_service:app --port 8002

# 2. in another terminal, run the driver over your dataset
uv run run-poc --dataset data/dataset.jsonl
```

Put your `query` / `ground_truth` pairs in `data/dataset.jsonl` (one JSON object
per line; `ground_truth` can be a string or a structured object).

### That's it

The driver prints an `operation_Id` per row and a ready-to-paste KQL query. Open
App Insights → Logs, paste it, and you'll see each trace with your ground truth
attached to the `invoke_agent` span. **You never wrote a line of telemetry
code.**

> **What you do NOT touch:** `eval_worker.py`, `agent_service.py`,
> `trace_context.py`, `telemetry.py`, and the `execute_agent` / `traceparent`
> wiring are the harness. They are shipped as-is and require no changes. The
> agent is a black box behind `build_agent()`.

## Why this design

| Concern | `trace-ground-truth-poc` | this POC |
|---|---|---|
| Processes | Single | **Two** (agent-service + eval-worker/driver) |
| Agent location | In-process | **Remote HTTP service** |
| Ground truth attached to | The `invoke_agent` span | The `invoke_agent` span (JSON **attribute**) |
| Agent code changes | n/a (single process) | **None** — the driver authors `invoke_agent` and injects `traceparent`; the remote agent just receives the header |

The `invoke_agent` span is authored by the driver and carries the ground truth.
The remote agent-service's `execute_agent` span nests under it (via the
propagated `traceparent`), so the whole invocation — request, agent work, and
ground truth — lives in one coherent span subtree.

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
   `traceparent` header; the agent-service opens `execute_agent` beneath it.
   The agent itself is a black box behind `build_agent()`.
4. **Attach a ground-truth object only.** Carry a structured ground-truth object
   (not a bare string). **No scores or results** are attached here — only
   evaluation *input*.
5. **Cheap, unambiguous backend query.** Correlation must be resolvable in App
   Insights with a simple, first-class join (no fragile string/JSON parsing).

## How it works

```mermaid
sequenceDiagram
    participant EW as eval_worker.py<br/>(driver, per item)
    participant TP as W3C traceparent<br/>(HTTP header)
    participant AS as agent_service.py<br/>(remote process)
    participant AF as Agent Framework<br/>(chat instrumentation)
    participant AI as App Insights

    Note over EW: rows = [id -> DatasetItem]<br/>(ground-truth registry)

    rect rgb(255, 249, 219)
    Note over EW: with start_as_current_span("invoke_agent") as span
    EW->>EW: span.set_attribute("gen_ai.evaluation.ground_truth", json(gt))
    EW->>EW: span.set_attribute("gen_ai.evaluation.item_id", id)
    EW->>TP: traceparent_for_span(span) = 00-<trace>-<span>-01
    EW->>AS: POST /invoke { item_id, query }<br/>header: traceparent
    end

    AS->>AS: parent_ctx = parent_context_from_traceparent(header)
    Note over AS: with start_as_current_span("execute_agent", context=parent_ctx)
    AS->>AF: RawAgent.run(agent, query)<br/>(bypasses AgentTelemetryLayer)
    AF-->>AF: emits "chat" / tool spans<br/>(nested UNDER execute_agent)
    AS-->>EW: { response, execute_agent_trace_id, execute_agent_span_id }
    EW->>EW: assert execute_agent_trace_id == invoke_agent trace_id

    par export (both processes -> same trace_id)
        EW->>AI: invoke_agent span + ground_truth attribute
    and
        AS->>AI: execute_agent + chat spans
    end
    Note over AI: post-processing step (later):<br/>read spans back, score vs ground truth
```

Resulting span tree (one `trace_id`):

```
invoke_agent                         (eval-worker/driver authors, span_id=A)
│  • attribute: gen_ai.evaluation.ground_truth = {...}   ← ground truth here
│  • attribute: gen_ai.evaluation.item_id = "<id>"
└─ execute_agent                     (agent-service authors, via traceparent header)
    └─ chat / tool spans             (agent framework)
```

The **agent process is never involved in correlation** — the *driver* authors
the `invoke_agent` span, injects its `traceparent` into the remote agent call,
and sets the ground truth as an attribute on that same span. The agent-service
opens an `execute_agent` span (parented to the driver's `invoke_agent` via the
header) and runs the agent beneath it, so `chat`/tool spans nest under
`execute_agent`. This matches ACA's hierarchy
(`invoke_agent → execute_agent → chat`), where the remote hosted-agent runtime
authors `execute_agent`.

> **Local-agent note.** In ACA, `execute_agent` is emitted by Foundry's *hosted*
> agent runtime. Because this POC runs the agent **locally**, the agent-service
> runs it via `RawAgent.run` to bypass Agent Framework's `AgentTelemetryLayer` —
> otherwise the framework would add its own `invoke_agent <agent_name>` span,
> which a hosted agent never produces. The agent *code* is unchanged; only *how
> the service invokes it* differs, yielding the exact tree a hosted-agent
> customer sees.

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
  agent_service.py                         # FastAPI /invoke — remote agent; framework emits invoke_agent <name>
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

The eval-worker prints, per row, the `invoke_agent` `trace_id`/`span_id`
(= `operation_Id`), confirms the remote agent ran under the **same trace**
(`agent_trace_id`), and at the end prints an `operation_Id` summary plus a
ready-to-paste App Insights KQL query.

## Verify in App Insights (KQL)

The ground truth is a JSON **attribute on the `invoke_agent` span itself**, so no
join is needed — query the span directly. The remote agent-service's
`execute_agent` span nests under it via `operation_ParentId`:

```kql
dependencies
| where name == "invoke_agent"
| project timestamp, operation_Id,
          invoke_agent_span = id,
          ground_truth = tostring(customDimensions["gen_ai.evaluation.ground_truth"])
// The remote agent-service's execute_agent span nests under invoke_agent:
//   dependencies | where name == "execute_agent"
//   | where operation_ParentId == <invoke_agent id>
```

To see every span from a specific run, filter by the `operation_Id`s the driver
prints (== the OTel `trace_id`s):

```kql
union traces, dependencies, requests, exceptions
| where operation_Id in ("<op_id_1>", "<op_id_2>", ...)
| project timestamp, itemType, name, operation_Id, operation_ParentId,
          ground_truth = tostring(customDimensions["gen_ai.evaluation.ground_truth"])
| order by timestamp asc
```

## Notes / trade-offs

- The ground-truth object is set **directly as a JSON attribute**
  (`gen_ai.evaluation.ground_truth`) on the `invoke_agent` span the driver
  authors — there is no separate eval/child span. OTel attribute values must be
  primitives, so the structured object travels as a JSON string.
- The driver **authors** the `invoke_agent` span; the agent is a **remote
  service** that opens an `execute_agent` span beneath it via the propagated
  `traceparent` header. This matches ACA's hierarchy
  (`invoke_agent → execute_agent → chat`): the driver span carries the ground
  truth, the `execute_agent` span carries the agent payload. Stamping ground
  truth on the driver's span is authoritative — the agent needs no tracing code.
- Evaluation/scoring is **not** performed here — only ground-truth *input* is
  attached. Scoring is a separate **post-processing** step run after all agent
  invocations complete (read the spans back from App Insights and compute
  metrics).
- If you also want the agent's *internal* spans in the same trace, enable OTel
  auto-instrumentation on the agent via **config** (still no source changes).
