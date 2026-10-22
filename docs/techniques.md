# Technique map

Every technique I use to build AI systems and my own tools, as of October 2026, and the example
that shows it. Each example comes from one of three settings: plant operations, industrial planning
and software building (see [the README](../README.md#the-examples)).

**Status.** *Built* means it runs today, with tests. *Planned* means the example is designed and
the technique is listed here so the map is complete; it is built next.

## Model access and the agent loop

| Technique | Shown in | Status |
|---|---|---|
| One interface for OpenAI, Anthropic and local models, no agent framework | [`providers/`](../src/aisyslab/providers/) | built |
| Function calling with typed, validated arguments (pydantic, cleaned JSON schema) | [`core/tools.py`](../src/aisyslab/core/tools.py), every lab | built |
| Structured JSON output checked by a schema grader | ticket triage, [`evals/graders.py`](../src/aisyslab/evals/graders.py) | built |
| Bounded loop: step, tool-call and input-token budgets enforced in code | [`core/agent.py`](../src/aisyslab/core/agent.py) | built |
| Reasoning effort setting; reasoning blocks replayed verbatim, history append-only | `--effort`, [Anthropic adapter](../src/aisyslab/providers/anthropic_messages.py) | built |
| Prompt caching through a stable prefix | Anthropic adapter (`cache_control`), static system prompts | built |
| Token counting and cost per completed task | [`core/tokens.py`](../src/aisyslab/core/tokens.py), `bench --price-in/--price-out` | built |
| Retries with backoff and `Retry-After` for 429/503, in code | [`core/resilience.py`](../src/aisyslab/core/resilience.py) | built |
| Arithmetic in a calculator tool, never in the model's head | [`core/calculator.py`](../src/aisyslab/core/calculator.py) | built |
| Model calls made inside tools (vision) are traced and costed | `RunContext.side_call` in [`core/lab.py`](../src/aisyslab/core/lab.py) | built |
| Mixed models: the models under test, a stronger optimiser, a separately chosen judge | ticket triage (`aisyslab eval`) | built |
| Local models through OpenAI-compatible servers (LM Studio, Ollama) | `--provider local` | built |
| Many items classified in one request under a provider-enforced JSON schema, compared with one request per item | ticket triage | planned |
| Responses cached for repeated identical inputs (duplicate tickets, repeated notes), never across tenants | [maintenance request service](runtime.md#one-gateway-for-every-model-call) | built |
| One gateway for every model call: routes, fallback across providers, circuit breaker, call log | [maintenance request service](runtime.md#one-gateway-for-every-model-call), [`runtime/gateway.py`](../src/aisyslab/runtime/gateway.py) | built |

## Tool design

| Technique | Shown in | Status |
|---|---|---|
| Narrow tools with one responsibility instead of a mirror of the API | every lab | built |
| Defaults set in code; errors returned as data with a hint for the next call | [`core/tools.py`](../src/aisyslab/core/tools.py) | built |
| Bounded output that tells the model how to narrow the query | `Toolbox(max_output_chars=...)`, [`service-repair`](labs.md#service-repair) | built |
| The contract discovered first: `help`, the database schema, the documents | [`service-repair`](labs.md#service-repair), [`plant-database`](labs.md#plant-database), [`spare-part-run`](labs.md#spare-part-run) | built |
| Safeguards inside the tool, not in the prompt: read-only engine, approval token | [`plant-database`](labs.md#plant-database), [`maintenance-portal`](labs.md#maintenance-portal) | built |
| Tools shaped by the questions they must answer; their names, descriptions and schemas measured without the system prompt | every lab, `--promptless` ([evals](evals.md#tool-descriptions-as-a-prompt)) | built |
| Two tools instead of a long list: `search_tools` and `call_tool` | [`mcp/discovery.py`](../src/aisyslab/mcp/discovery.py) | built |
| Exact formats (records, reports) rendered by code from the fields the model fills | quality claim assistant | planned |
| Tools as a library: the agent writes code that calls them, instead of one call per step | production schedule | planned |

## MCP

| Technique | Shown in | Status |
|---|---|---|
| Every lab as an MCP server: tools, JSON schemas, the task as a resource, over stdio or streamable HTTP | [`mcp/server.py`](../src/aisyslab/mcp/server.py), `aisyslab mcp` | built |
| A host that connects several servers, audits them and merges their tools under prefixes | [`mcp/host.py`](../src/aisyslab/mcp/host.py) | built |
| Any lab agent routed through the protocol, same verifier and budgets | `--via-mcp`, [`mcp/route.py`](../src/aisyslab/mcp/route.py) | built |

## Context and memory

| Technique | Shown in | Status |
|---|---|---|
| Two-step reading: metadata first, full content on demand | [`inbox`](labs.md#inbox) | built |
| Deterministic reduction before any model call: unique events, grouped repeats | [`line-incident`](labs.md#line-incident) | built |
| Code chooses the context, the model answers on it | [`line-incident`](labs.md#line-incident), [`inbox`](labs.md#inbox) | built |
| A context budget with token counting before submitting | [`line-incident`](labs.md#line-incident) | built |
| Compaction: the history is replaced by observations, never edited | `_compact` in [`core/agent.py`](../src/aisyslab/core/agent.py) | built |
| Separate contexts per agent; documents passed by reference, not pasted | [`shift-briefing`](labs.md#shift-briefing) | built |
| Long-term memory across sessions: an observer records, a reflector condenses | shift handover memory | planned |
| Memory scopes (line, asset, person, plant) and facts that supersede older ones | shift handover memory | planned |
| A working area for new observations; only confirmed facts enter the canonical record | shift handover memory | planned |
| A knowledge base compressed to fit a token limit, kept readable by people | shift handover memory | planned |

## Retrieval

| Technique | Shown in | Status |
|---|---|---|
| The simplest search that works first: query operators and keywords on data with a stable shape | [`inbox`](labs.md#inbox) | built |
| Embeddings and vector ranking (offline trigram, or a provider's embedding model) | [`core/retrieval.py`](../src/aisyslab/core/retrieval.py), `--embedder` | built |
| Hybrid search: BM25 and vectors fused with Reciprocal Rank Fusion | [`core/retrieval.py`](../src/aisyslab/core/retrieval.py) | built |
| Agentic search: the agent writes the queries, across languages | [`relocation-rag`](labs.md#relocation-rag) | built |
| Citations checked by the verifier; "not covered" instead of a guess | [`relocation-rag`](labs.md#relocation-rag) | built |
| Chunks cut along the document's structure (sections), with metadata on every chunk | [`relocation-rag`](labs.md#relocation-rag) | built |
| Source metadata (edition, status) to avoid superseded documents | [`relocation-rag`](labs.md#relocation-rag) | built |
| Text-to-SQL after schema discovery; the computation done by the database | [`plant-database`](labs.md#plant-database) | built |
| Graph RAG: when the relations are the knowledge, traversal tools instead of similarity | asset and parts graph | planned |
| Extraction from documents into a structured store, validated by a parser | asset and parts graph | planned |
| A web search API and a crawler with rate limits and retries | replacement part research | planned |
| Deep research loop: plan, search, read, name the gaps, search again | replacement part research | planned |
| A long report written from an outline, section by section, every claim cited | replacement part research | planned |

## Images and voice

| Technique | Shown in | Status |
|---|---|---|
| Route files by content (magic bytes), not by name; decode structured formats in code | [`supply-intake`](labs.md#supply-intake) | built |
| Vision on a photo, cross-checked by a deterministic validator | [`supply-intake`](labs.md#supply-intake) | built |
| Speech-to-text grounded in the plant's vocabulary (asset names, part numbers) | voice on the shop floor | planned |
| Text-to-speech cached by text, voice and model; an utterance that may have been delivered is never replayed | voice on the shop floor | planned |
| A voice agent as a cascade (speech-to-text, model, text-to-speech) with a latency budget per stage | voice on the shop floor | planned |
| Image generation, a template rendered in code, and a vision check before the result is used | visual work instructions | planned |

## Several agents

| Technique | Shown in | Status |
|---|---|---|
| An orchestrator delegating to specialists with their own context, tools, budgets and verifiers | [`shift-briefing`](labs.md#shift-briefing), [orchestration](orchestration.md) | built |
| A parallel dependency graph when the plan is known | [`orchestration/dag.py`](../src/aisyslab/orchestration/dag.py) | built |
| A shared workspace (blackboard) with versioned, conflict-checked writes | [`orchestration/workspace.py`](../src/aisyslab/orchestration/workspace.py) | built |
| Sub-agent cost and trace events attributed to the parent run | `Tracer.child`, [`orchestration/delegate.py`](../src/aisyslab/orchestration/delegate.py) | built |
| Delegation as a waiting state of a durable job, nested like a call stack | [maintenance request service](runtime.md#waiting-is-a-state) | built |
| Background agents grouped in time instead of linked by dependencies | [maintenance request service](runtime.md#grouped-in-time-not-linked-by-dependencies) (morning briefing) | built |
| Triggers other than a chat: API calls, cron, an external bootstrap call, a recurring check (heartbeat) | [maintenance request service](runtime.md#grouped-in-time-not-linked-by-dependencies), [`runtime/cron.py`](../src/aisyslab/runtime/cron.py) | built |

## Verification and evaluation

| Technique | Shown in | Status |
|---|---|---|
| A deterministic verifier whose feedback guides the next attempt without leaking the answer | every lab | built |
| The model proposes, code judges: simulator, validator, solver | [`spare-part-run`](labs.md#spare-part-run), [`supply-intake`](labs.md#supply-intake) | built |
| Several diverse candidates instead of one guess | [`spare-part-run`](labs.md#spare-part-run) | built |
| A code baseline next to the agent, under the same verifier | `aisyslab bench`, every lab | built |
| Fault injection: APIs that fail at a chosen rate | `--flaky` | built |
| A trace of every model call, tool call, retry and verdict, in Langfuse's run/step/generation shape | [`core/trace.py`](../src/aisyslab/core/trace.py) | built |
| Eval set with train, dev and holdout splits and a content fingerprint | [evals](evals.md) | built |
| Graders: code where code can decide, a model judge only for the rest | [evals](evals.md) | built |
| The judge calibrated against human labels: agreement, Cohen's kappa, disagreements | [evals](evals.md) | built |
| Models compared side by side before a migration; regressions listed item by item | [evals](evals.md) | built |
| Prompt optimisation checked on a holdout; prompts versioned by hash | [evals](evals.md) | built |
| An agent's instructions drafted by a meta-prompt from an interview with the process owner | ticket triage | planned |
| Classic ML instead of a model where the answer is in labelled numbers | [`quality-gate`](labs.md#quality-gate) | built |
| Fine-tuning compared with prompting on the same holdout, with cost per thousand items | fine-tuned triage | planned |

## Safety

| Technique | Shown in | Status |
|---|---|---|
| Policy enforced by the environment, not by the prompt | [`service-repair`](labs.md#service-repair) | built |
| Untrusted text treated as data: prompt injection handled by design | [`maintenance-portal`](labs.md#maintenance-portal) | built |
| Human approval bound to one irreversible action | [`maintenance-portal`](labs.md#maintenance-portal), `--operator ask` | built |
| Read-only by default, enforced by the database engine | [`plant-database`](labs.md#plant-database) | built |
| A sender trust rule against look-alike domains | [`inbox`](labs.md#inbox) | built |
| Data classified before anything is sent to a model; the most sensitive goes only to a local model | quality claim assistant | planned |
| A quarantined extraction step: a model without tools turns untrusted text into validated fields | quality claim assistant | planned |
| Anonymisation in layers, with reversible placeholders | quality claim assistant | planned |
| A moderation hook: input checked before it reaches any provider | [maintenance request service](runtime.md#one-gateway-for-every-model-call) | built |
| Moderation by a moderation model, with categories and thresholds | quality claim assistant | planned |
| A topic guard: a second model checks that the answer stays on topic | quality claim assistant | planned |
| A tamper-evident audit log of model calls and decisions | quality claim assistant | planned |
| Code written by the model runs in a sandbox with time, memory and network limits | production schedule | planned |

## Running it as a service

| Technique | Shown in | Status |
|---|---|---|
| Endpoints with a fixed shape instead of an open chat endpoint | [maintenance request service](runtime.md#endpoints-with-a-fixed-shape), [`runtime/api.py`](../src/aisyslab/runtime/api.py) | built |
| A message is a job: claimed with a heartbeat, recovered when a worker dies, fenced against double work | [maintenance request service](runtime.md#a-message-is-a-job), [`runtime/engine.py`](../src/aisyslab/runtime/engine.py) | built |
| A tool-call journal with idempotency keys, so a resumed job repeats nothing it finished | [maintenance request service](runtime.md#a-message-is-a-job), [`runtime/store.py`](../src/aisyslab/runtime/store.py) | built |
| The simulated world rebuilt from the journal; a replay that differs fails the run | [maintenance request service](runtime.md#a-message-is-a-job), [`runtime/worlds.py`](../src/aisyslab/runtime/worlds.py) | built |
| Items (turns, tool calls, results) instead of messages; branch instead of delete | [maintenance request service](runtime.md#items-not-messages-branches-not-deletes) | built |
| Events and the model's text streamed over SSE, resumable after a dropped connection | [maintenance request service](runtime.md#endpoints-with-a-fixed-shape), provider adapters (`complete_stream`) | built |
| Tenants with their own tokens and token budgets; a snapshot of the agent's settings with every run | [maintenance request service](runtime.md#agent-settings-are-recorded-with-every-run) | built |
| Graceful degradation: model output validated, a rule-based fallback flagged as degraded | [maintenance request service](runtime.md#triage-that-degrades-visibly) | built |
| A web UI with streaming, a queue for messages sent while the agent works, and structured cards | [maintenance request service](runtime.md#the-web-ui), [`static/index.html`](../src/aisyslab/runtime/static/index.html) | built |
| A process described in files that the agents read when they need them | [maintenance request service](runtime.md#process-files-read-on-demand), [`processes/`](../src/aisyslab/runtime/processes/) | built |
| Offline test suite, type checking and linting | [`tests/`](../tests/), `mypy`, `ruff` | built |

## Known, and deliberately not used here

- **Mesh and swarm topologies.** No problem here needs agents that negotiate without a coordinator;
  an orchestrator and a dependency graph are easier to verify.
- **External vector and graph databases.** In-process indexes and graphs keep every example
  runnable without services; the retrieval interface stays the same.
- **An observability server.** Traces are JSONL in the same hierarchy and can be exported.
- **No-code automation platforms.** The maintenance request service has its own API.
- **A fallback to code inside the lab agents.** Switched off on purpose: the benchmark would
  measure the fallback, not the agent. The service, where availability matters more, has one.
