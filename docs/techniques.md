# Technique map

Every technique I use to build AI systems and my own tools, as of October 2026, and the example
that shows it. Each example comes from one of three settings: plant operations, industrial planning
and software building.

Every technique below runs today and has tests. Where a technique was also run against a live
model, the linked page says what happened, including what failed.

## The examples

**Plant operations**

| Example | The question | What it shows |
|---|---|---|
| [`inbox`](labs.md#inbox) | Three facts from the plant mailbox, with corrections and look-alike senders | two-step retrieval, query operators, a sender trust rule |
| [`line-incident`](labs.md#line-incident) | Why did line 3 stop? | a context budget, reduction in code before any model call |
| [`maintenance-portal`](labs.md#maintenance-portal) | Re-code one work order and close another | prompt injection handled by design, human approval |
| [`quality-gate`](labs.md#quality-gate) | Which of tonight's batches pass the gate? | classic ML in code, target leakage |
| [`shift-briefing`](labs.md#shift-briefing) | One report from three investigations | an orchestrator and specialists, a shared workspace |
| [`quality-claims`](labs.md#quality-claims) | A customer complaint about a batch: decide, record, reply | data classification, anonymisation, moderation, quarantined extraction, a topic guard, an audit log |
| [`shift-handover`](labs.md#shift-handover) | What does the next shift need to know? | observational memory, scopes, facts that supersede older ones |
| [`procedures-rag`](labs.md#procedures-rag) | What do the procedures say? Answer, cite, or say it is not covered | hybrid retrieval, search across languages, checked citations |
| [Voice on the shop floor](voice.md) | Voice notes at the machine filed as work requests; a spoken answer for the shift lead | speech-to-text grounded in the plant's vocabulary, speech cached and said at most once, a voice cascade on a latency budget |
| [Visual work instructions](visual.md) | An instruction card for a maintenance step, released only after a check | a card drawn in code, a generated illustration with no text in it, two vision checks decided in code |

**Industrial planning**

| Example | The question | What it shows |
|---|---|---|
| [`supply-intake`](labs.md#supply-intake) | Which supplier delay stops a line tonight? | routing by content, vision cross-checked in code |
| [`spare-part-run`](labs.md#spare-part-run) | Get a spare pump to the stopped line before the deadline | the model proposes, a simulator judges |
| [`plant-database`](labs.md#plant-database) | Which work orders are blocked by a missing part? | schema discovery, read-only SQL |
| [Ticket triage](evals.md) | Maintenance tickets to failure codes and priorities | eval sets, a calibrated judge, model comparison, prompt optimisation |
| [`production-schedule`](labs.md#production-schedule) | Tomorrow's orders on three presses, with the planner's notes | code mode: the model writes a program that plans, run in a sandbox |
| [`part-research`](labs.md#part-research) | A component is discontinued: what replaces it, from whom, how fast? | web search, a polite crawler, a deep research loop, a cited report |
| [`parts-graph`](labs.md#parts-graph) | A supplier delays a part: which lines and orders does it hold up, and which can an alternate save? | graph RAG, extraction validated by a parser, facts with dates |
| [Triage, batched, interviewed, trained](evals.md#many-tickets-in-one-request-under-a-schema) | The triage prompt run in batches, drafted from an interview with the planner, and set against a model trained on its examples | batches under a provider-enforced schema, a meta-prompt from an interview, training compared with prompting |

**Software building**

| Example | The question | What it shows |
|---|---|---|
| [`service-repair`](labs.md#service-repair) | Get the line's edge gateway running again | a restricted shell, policy enforced by the environment |
| [MCP servers and host](mcp.md) | Use the labs' tools from any MCP client, and MCP servers from the agent | servers over stdio and HTTP, a host, tool discovery |
| [Agent toolkit](architecture.md) | The loop under every example | typed tools, budgets, retries, providers, traces |
| [Maintenance request service](runtime.md) | The team's requests as durable jobs, with a web UI | a model gateway, a durable runtime, streaming, tenants, cron |

## Model access and the agent loop

| Technique | Shown in |
|---|---|
| One interface for OpenAI, Anthropic and local models, no agent framework | [`providers/`](../src/aisyslab/providers/) |
| Function calling with typed, validated arguments (pydantic, cleaned JSON schema) | [`core/tools.py`](../src/aisyslab/core/tools.py), every lab |
| Structured JSON output checked by a schema grader | ticket triage, [`evals/graders.py`](../src/aisyslab/evals/graders.py) |
| Bounded loop: step, tool-call and input-token budgets enforced in code | [`core/agent.py`](../src/aisyslab/core/agent.py) |
| Reasoning effort setting; reasoning blocks replayed verbatim, history append-only | `--effort`, [Anthropic adapter](../src/aisyslab/providers/anthropic_messages.py) |
| Prompt caching through a stable prefix | Anthropic adapter (`cache_control`), static system prompts |
| Token counting and cost per completed task | [`core/tokens.py`](../src/aisyslab/core/tokens.py), `bench --price-in/--price-out` |
| Retries with backoff and `Retry-After` for 429/503, in code | [`core/resilience.py`](../src/aisyslab/core/resilience.py) |
| Arithmetic in a calculator tool, never in the model's head | [`core/calculator.py`](../src/aisyslab/core/calculator.py) |
| Model calls made inside tools (vision) are traced and costed | `RunContext.side_call` in [`core/lab.py`](../src/aisyslab/core/lab.py) |
| Mixed models: the models under test, a stronger optimiser, a separately chosen judge | ticket triage (`aisyslab eval`) |
| Local models through OpenAI-compatible servers (LM Studio, Ollama) | `--provider local` |
| Many items classified in one request under a provider-enforced JSON schema, compared with one request per item | [evals](evals.md#many-tickets-in-one-request-under-a-schema), [`evals/batch.py`](../src/aisyslab/evals/batch.py) |
| Responses cached for repeated identical inputs (duplicate tickets, repeated notes), never across tenants | [maintenance request service](runtime.md#one-gateway-for-every-model-call) |
| One gateway for every model call: routes, fallback across providers, circuit breaker, call log | [maintenance request service](runtime.md#one-gateway-for-every-model-call), [`runtime/gateway.py`](../src/aisyslab/runtime/gateway.py) |

## Tool design

| Technique | Shown in |
|---|---|
| Narrow tools with one responsibility instead of a mirror of the API | every lab |
| Defaults set in code; errors returned as data with a hint for the next call | [`core/tools.py`](../src/aisyslab/core/tools.py) |
| Bounded output that tells the model how to narrow the query | `Toolbox(max_output_chars=...)`, [`service-repair`](labs.md#service-repair) |
| The contract discovered first: `help`, the database schema, the documents | [`service-repair`](labs.md#service-repair), [`plant-database`](labs.md#plant-database), [`spare-part-run`](labs.md#spare-part-run) |
| Safeguards inside the tool, not in the prompt: read-only engine, approval token | [`plant-database`](labs.md#plant-database), [`maintenance-portal`](labs.md#maintenance-portal) |
| Tools shaped by the questions they must answer; their names, descriptions and schemas measured without the system prompt | every lab, `--promptless` ([evals](evals.md#tool-descriptions-as-a-prompt)) |
| Two tools instead of a long list: `search_tools` and `call_tool`, with the answer tools pinned | [`mcp/discovery.py`](../src/aisyslab/mcp/discovery.py) |
| Exact formats (records, reports) rendered by code from the fields the model fills | [`quality-claims`](labs.md#quality-claims) (8D record) |
| Tools as a library: the agent writes code that calls them, instead of one call per step | [`production-schedule`](labs.md#production-schedule), [sandbox](sandbox.md) |

## MCP

| Technique | Shown in |
|---|---|
| Every lab as an MCP server: tools, JSON schemas, the task as a resource, over stdio or streamable HTTP | [`mcp/server.py`](../src/aisyslab/mcp/server.py), `aisyslab mcp` |
| A host that connects several servers, audits them and merges their tools under prefixes | [`mcp/host.py`](../src/aisyslab/mcp/host.py) |
| Any lab agent routed through the protocol, same verifier and budgets | `--via-mcp`, [`mcp/route.py`](../src/aisyslab/mcp/route.py) |

## Context and memory

| Technique | Shown in |
|---|---|
| Two-step reading: metadata first, full content on demand | [`inbox`](labs.md#inbox) |
| Deterministic reduction before any model call: unique events, grouped repeats | [`line-incident`](labs.md#line-incident) |
| Code chooses the context, the model answers on it | [`line-incident`](labs.md#line-incident), [`inbox`](labs.md#inbox) |
| A context budget with token counting before submitting | [`line-incident`](labs.md#line-incident) |
| Compaction: the history is replaced by observations, never edited | `_compact` in [`core/agent.py`](../src/aisyslab/core/agent.py) |
| Separate contexts per agent; documents passed by reference, not pasted | [`shift-briefing`](labs.md#shift-briefing) |
| Long-term memory across sessions: an observer records, a reflector condenses | [`shift-handover`](labs.md#shift-handover), [`memory/observational.py`](../src/aisyslab/memory/observational.py) |
| Memory scopes (line, asset, person, plant) and facts that supersede older ones | [`shift-handover`](labs.md#shift-handover) |
| A working area for new observations; only the reflector writes the record that goes into prompts | [`shift-handover`](labs.md#shift-handover) |
| A record compressed to a token budget the environment enforces, kept readable by people | [`shift-handover`](labs.md#shift-handover) |

## Retrieval

| Technique | Shown in |
|---|---|
| The simplest search that works first: query operators and keywords on data with a stable shape | [`inbox`](labs.md#inbox) |
| Embeddings and vector ranking (offline trigram, or a provider's embedding model) | [`core/retrieval.py`](../src/aisyslab/core/retrieval.py), `--embedder` |
| Hybrid search: BM25 and vectors fused with Reciprocal Rank Fusion | [`core/retrieval.py`](../src/aisyslab/core/retrieval.py) |
| Agentic search: the agent writes the queries, across languages | [`procedures-rag`](labs.md#procedures-rag) |
| Citations checked by the verifier; "not covered" instead of a guess | [`procedures-rag`](labs.md#procedures-rag) |
| Chunks cut along the document's structure (sections), with metadata on every chunk | [`procedures-rag`](labs.md#procedures-rag) |
| Source metadata (edition, status) to avoid superseded documents | [`procedures-rag`](labs.md#procedures-rag) |
| Text-to-SQL after schema discovery; the computation done by the database | [`plant-database`](labs.md#plant-database) |
| Graph RAG: when the relations are the knowledge, traversal tools instead of similarity | [`parts-graph`](labs.md#parts-graph), [graph](graph.md#traversal-tools) |
| Extraction from documents into a structured store, validated by a parser; rejected lines sent back with a reason | [`graph/extract.py`](../src/aisyslab/graph/extract.py), [graph](graph.md#tables-by-code-notes-by-a-model-both-checked) |
| Facts with dates: since and until, functional relations, planned changes that do not count yet | [`graph/store.py`](../src/aisyslab/graph/store.py), [graph](graph.md#time-on-the-edges) |
| A web search API behind one interface (offline index, or a search model on the real web) and a polite crawler | [research](research.md#one-search-interface-two-backends), `aisyslab search` |
| Deep research loop: search, read, note with sources, name the gaps, search again | [`part-research`](labs.md#part-research) |
| A long report written from an outline, section by section, every citation checked in code | [research](research.md#a-long-report-section-by-section) |

## Images and voice

| Technique | Shown in |
|---|---|
| Route files by content (magic bytes), not by name; decode structured formats in code | [`supply-intake`](labs.md#supply-intake) |
| Vision on a photo, cross-checked by a deterministic validator | [`supply-intake`](labs.md#supply-intake) |
| Speech-to-text grounded in the plant's vocabulary: terms up front, then IDs checked against the registry, teen/ty corrected, the rest asked | [voice](voice.md#speech-to-text-grounded-in-the-plants-vocabulary), [`voice/vocabulary.py`](../src/aisyslab/voice/vocabulary.py) |
| A speech pipeline evaluated step by step on recorded audio, each step's gain measured | [voice](voice.md#the-evaluation-ten-voice-notes) |
| Text-to-speech cached by text, voice and model; an utterance that may have been delivered is never replayed | [voice](voice.md#speech-that-is-paid-for-once-and-said-once), [`voice/delivery.py`](../src/aisyslab/voice/delivery.py) |
| A voice agent as a cascade (speech-to-text, model, text-to-speech) with a latency budget per stage, streamed by sentence, a filler when the model is slow | [voice](voice.md#a-voice-agent-as-a-cascade-with-a-latency-budget-per-stage), [`voice/cascade.py`](../src/aisyslab/voice/cascade.py) |
| Image generation, a template rendered in code, and a vision check before the result is used | [visual](visual.md), [`visual/check.py`](../src/aisyslab/visual/check.py) |
| A vision check that lists what it sees, with the decision in code; a read-back of the rendered result against its source | [visual](visual.md#two-checks-before-the-card-is-used) |

## Several agents

| Technique | Shown in |
|---|---|
| An orchestrator delegating to specialists with their own context, tools, budgets and verifiers | [`shift-briefing`](labs.md#shift-briefing), [orchestration](orchestration.md) |
| A parallel dependency graph when the plan is known | [`orchestration/dag.py`](../src/aisyslab/orchestration/dag.py) |
| A shared workspace (blackboard) with versioned, conflict-checked writes | [`orchestration/workspace.py`](../src/aisyslab/orchestration/workspace.py) |
| Sub-agent cost and trace events attributed to the parent run | `Tracer.child`, [`orchestration/delegate.py`](../src/aisyslab/orchestration/delegate.py) |
| Delegation as a waiting state of a durable job, nested like a call stack | [maintenance request service](runtime.md#waiting-is-a-state) |
| Background agents grouped in time instead of linked by dependencies | [maintenance request service](runtime.md#grouped-in-time-not-linked-by-dependencies) (morning briefing) |
| Triggers other than a chat: API calls, cron, an external bootstrap call, a recurring check (heartbeat); a due schedule fires once across schedulers, and one that fails records its error without stopping the rest | [maintenance request service](runtime.md#grouped-in-time-not-linked-by-dependencies), [`runtime/cron.py`](../src/aisyslab/runtime/cron.py) |

## Verification and evaluation

| Technique | Shown in |
|---|---|
| A deterministic verifier whose feedback guides the next attempt without leaking the answer | every lab |
| The model proposes, code judges: simulator, validator, solver | [`spare-part-run`](labs.md#spare-part-run), [`supply-intake`](labs.md#supply-intake) |
| Several diverse candidates instead of one guess | [`spare-part-run`](labs.md#spare-part-run) |
| A code baseline next to the agent, under the same verifier | `aisyslab bench`, every lab |
| Fault injection: APIs that fail at a chosen rate | `--flaky` |
| A trace of every model call, tool call, retry and verdict, shaped to map onto Langfuse traces and observations | [`core/trace.py`](../src/aisyslab/core/trace.py) |
| Eval set with train, dev and holdout splits and a content fingerprint | [evals](evals.md) |
| Graders: code where code can decide, a model judge only for the rest | [evals](evals.md) |
| The judge calibrated against human labels: agreement, Cohen's kappa, disagreements | [evals](evals.md) |
| Models compared side by side before a migration; regressions listed item by item | [evals](evals.md) |
| Error analysis: every item's output saved and read before a score is trusted | [evals](evals.md#a-live-run-what-the-numbers-hid) |
| Prompt optimisation checked on a holdout; prompts versioned by hash | [evals](evals.md) |
| An agent's instructions drafted by a meta-prompt from an interview with the process owner | [evals](evals.md#a-prompt-drafted-from-an-interview-with-the-process-owner), [`evals/interview.py`](../src/aisyslab/evals/interview.py) |
| Classic ML instead of a model where the answer is in labelled numbers | [`quality-gate`](labs.md#quality-gate) |
| Fine-tuning compared with prompting on the same holdout, with cost per thousand items; training data validated before paying, the job journaled | [evals](evals.md#fine-tuning-compared-with-prompting), [`evals/finetune.py`](../src/aisyslab/evals/finetune.py) |
| Dataset hygiene: every item unique, nothing from the holdout in any training file | [evals](evals.md#the-dataset) |

## Safety

| Technique | Shown in |
|---|---|
| Policy enforced by the environment, not by the prompt | [`service-repair`](labs.md#service-repair) |
| Untrusted text treated as data: prompt injection handled by design | [`maintenance-portal`](labs.md#maintenance-portal) |
| Human approval bound to one irreversible action | [`maintenance-portal`](labs.md#maintenance-portal), `--operator ask` |
| Read-only by default, enforced by the database engine | [`plant-database`](labs.md#plant-database) |
| A sender trust rule against look-alike domains | [`inbox`](labs.md#inbox) |
| Data classified before anything is sent to a model; the most sensitive goes only to a local model | [guardrails](guardrails.md#classify-then-anonymise-in-layers), `--local-model` |
| A quarantined extraction step: a model without tools turns untrusted text into validated fields | [guardrails](guardrails.md#quarantined-extraction) |
| Anonymisation in layers, with reversible placeholders | [guardrails](guardrails.md#classify-then-anonymise-in-layers) |
| A moderation hook: input checked before it reaches any provider | [maintenance request service](runtime.md#one-gateway-for-every-model-call) |
| Moderation by a moderation model, with categories and thresholds | [guardrails](guardrails.md#moderation-with-thresholds-per-category), `--moderation openai` |
| A topic guard: a second model checks that the answer stays on topic | [guardrails](guardrails.md#a-topic-guard-on-replies) |
| A tamper-evident audit log of model calls and decisions | [guardrails](guardrails.md#a-signed-chained-audit-log) |
| A leak check at the boundary: a prompt holding personal data is blocked before it leaves | [guardrails](guardrails.md#the-leak-check) |
| Code written by the model runs in a sandbox with time, memory, output and network limits, and without keys in its environment | [`sandbox/runner.py`](../src/aisyslab/sandbox/runner.py), [`production-schedule`](labs.md#production-schedule) |

## Running it as a service

| Technique | Shown in |
|---|---|
| Endpoints with a fixed shape instead of an open chat endpoint | [maintenance request service](runtime.md#endpoints-with-a-fixed-shape), [`runtime/api.py`](../src/aisyslab/runtime/api.py) |
| A message is a job: claimed as a lease with a heartbeat, recovered when a worker dies, fenced against double work, also across processes | [maintenance request service](runtime.md#a-message-is-a-job), [`runtime/engine.py`](../src/aisyslab/runtime/engine.py) |
| A tool-call journal with idempotency keys, so a resumed job repeats nothing it finished | [maintenance request service](runtime.md#a-message-is-a-job), [`runtime/store.py`](../src/aisyslab/runtime/store.py) |
| The simulated world rebuilt from the journal, and dropped after a step that ended abnormally; a replay that differs fails the run | [maintenance request service](runtime.md#a-message-is-a-job), [`runtime/worlds.py`](../src/aisyslab/runtime/worlds.py) |
| Items (turns, tool calls, results) instead of messages; branch instead of delete | [maintenance request service](runtime.md#items-not-messages-branches-not-deletes) |
| Events and the model's text streamed over SSE, resumable after a dropped connection | [maintenance request service](runtime.md#endpoints-with-a-fixed-shape), provider adapters (`complete_stream`) |
| Tenants with their own tokens and token budgets; a snapshot of the agent's settings with every run | [maintenance request service](runtime.md#agent-settings-are-recorded-with-every-run) |
| Graceful degradation: model output validated, a rule-based fallback flagged as degraded | [maintenance request service](runtime.md#triage-that-degrades-visibly) |
| A web UI with streaming, a queue for messages sent while the agent works, and structured cards | [maintenance request service](runtime.md#the-web-ui), [`static/index.html`](../src/aisyslab/runtime/static/index.html) |
| A process described in files that the agents read when they need them | [maintenance request service](runtime.md#process-files-read-on-demand), [`processes/`](../src/aisyslab/runtime/processes/) |
| Offline test suite, type checking and linting | [`tests/`](../tests/), `mypy`, `ruff` |

## Known, and deliberately not used here

- **Mesh and swarm topologies.** No problem here needs agents that negotiate without a coordinator;
  an orchestrator and a dependency graph are easier to verify.
- **External vector and graph databases.** In-process indexes and graphs keep every example
  runnable without services; the retrieval interface stays the same.
- **An observability server.** Traces are JSONL in the same hierarchy and can be exported.
- **No-code automation platforms.** The maintenance request service has its own API.
- **A fallback to code inside the lab agents.** Switched off on purpose: the benchmark would
  measure the fallback, not the agent. The service, where availability matters more, has one.
