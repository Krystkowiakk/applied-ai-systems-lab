# Technique index

Where each technique lives in this repository.

## Tools and the agent loop

| Technique | Where |
|---|---|
| Typed tools: pydantic arguments, generated and cleaned JSON schema | [`core/tools.py`](../src/aisyslab/core/tools.py) |
| Errors returned as data, with a hint for the next call | [`core/tools.py`](../src/aisyslab/core/tools.py), search hints in [`labs/inbox/env.py`](../src/aisyslab/labs/inbox/env.py) |
| Bounded tool output that says how to narrow the query | `Toolbox(max_output_chars=...)`, shell output cap in [`labs/service_repair/env.py`](../src/aisyslab/labs/service_repair/env.py) |
| Budgets enforced in code (steps, tool calls, input tokens) | [`core/agent.py`](../src/aisyslab/core/agent.py) |
| Compaction into observations instead of editing history | `_compact` in [`core/agent.py`](../src/aisyslab/core/agent.py) |
| Arithmetic in a tool, not in the model's head | [`core/calculator.py`](../src/aisyslab/core/calculator.py) |
| Tool calls made inside tools are traced and costed | `RunContext.side_call` in [`core/lab.py`](../src/aisyslab/core/lab.py) |

## Context and retrieval

| Technique | Where |
|---|---|
| Two-step retrieval: metadata first, full content on demand | [`labs/inbox`](../src/aisyslab/labs/inbox/agent.py) |
| Context budget with explicit token counting before submitting | [`labs/line_incident`](../src/aisyslab/labs/line_incident/agent.py) |
| Grouped repeats and deterministic noise filtering before any model call | `search(collapse_repeats=True)` and the baseline in [`labs/line_incident`](../src/aisyslab/labs/line_incident/) |
| Hybrid retrieval: BM25 + vector ranking fused with Reciprocal Rank Fusion | [`core/retrieval.py`](../src/aisyslab/core/retrieval.py) |
| Agentic search: the agent writes keyword queries, in the corpus language | prompt in [`labs/relocation_rag/agent.py`](../src/aisyslab/labs/relocation_rag/agent.py) |
| Source metadata (edition, status) to avoid superseded documents | [`labs/relocation_rag/corpus.py`](../src/aisyslab/labs/relocation_rag/corpus.py) |
| Citations checked by the verifier; "not covered" instead of guessing | [`labs/relocation_rag/env.py`](../src/aisyslab/labs/relocation_rag/env.py) |
| Schema discovery before the query: tables, definitions, sample rows | `list_tables`, `describe_table` in [`labs/plant_database/agent.py`](../src/aisyslab/labs/plant_database/agent.py) |
| Computation pushed into SQL instead of the model | [`labs/plant_database/baseline.py`](../src/aisyslab/labs/plant_database/baseline.py) and the agent prompt |

## Multimodality

| Technique | Where |
|---|---|
| Route by content (magic bytes), not by file name | `sniff` in [`labs/supply_intake/env.py`](../src/aisyslab/labs/supply_intake/env.py) |
| Decode structured formats in code; model only for what needs a reader | `decode` in [`labs/supply_intake/agent.py`](../src/aisyslab/labs/supply_intake/agent.py) |
| Vision answer cross-checked by a deterministic validator, without leaking it | `inspect_image` + `count_bins` |

## Verification and planning

| Technique | Where |
|---|---|
| Verifier feedback as the contract for the next attempt | every `env.submit` returning a `Verdict` |
| Grounding check: every claimed event must exist in the source | timeline verifier in [`labs/line_incident/env.py`](../src/aisyslab/labs/line_incident/env.py) |
| The model proposes, a simulator judges | `simulate_plan` in [`labs/spare_part_run`](../src/aisyslab/labs/spare_part_run/agent.py) |
| Several diverse candidates instead of one guess | spare-part-run agent prompt |
| Discover documents and the API contract before acting | `search_docs`, shell `help` |
| Exact search as the baseline for fully specified planning | `solve_grid` in [`labs/spare_part_run/env.py`](../src/aisyslab/labs/spare_part_run/env.py) |
| A set answer graded with categorised feedback (kinds of mistake, no IDs) and an F1 score | `submit` in [`labs/plant_database/env.py`](../src/aisyslab/labs/plant_database/env.py) |

## Safety and control

| Technique | Where |
|---|---|
| Policy enforced by the environment (forbidden paths, bans, normalised paths) | [`labs/service_repair/env.py`](../src/aisyslab/labs/service_repair/env.py) |
| Read plane separated from write plane | [`labs/maintenance_portal`](../src/aisyslab/labs/maintenance_portal/) |
| HTML parsed in code; free text labelled as untrusted; no bulk tools | [`labs/maintenance_portal/pages.py`](../src/aisyslab/labs/maintenance_portal/pages.py) |
| Human approval token for irreversible actions (`--operator ask`) | `request_approval` / `close` in the portal env, `--operator` in [`cli.py`](../src/aisyslab/cli.py) |
| Verifier fails attempts that were blocked, not only bad final states | `finish` in the portal env |
| Sender trust rule against look-alike domains | inbox prompt and verifier |
| Read-only enforced by the database engine: authorizer, `query_only`, one statement per call | `_authorize` in [`labs/plant_database/env.py`](../src/aisyslab/labs/plant_database/env.py) |
| Query cost limit (progress handler) and capped result sets | `_tick` and `query` in [`labs/plant_database/env.py`](../src/aisyslab/labs/plant_database/env.py) |

## Production concerns

| Technique | Where |
|---|---|
| Retries with backoff for 503/429 in code; exhausted retries visible to the model | [`core/resilience.py`](../src/aisyslab/core/resilience.py), `RunContext.tool_call` |
| Chaos testing: flaky APIs from a separate random stream | `LabEnv._api` in [`core/lab.py`](../src/aisyslab/core/lab.py), `--flaky` |
| Provider translation layer instead of a framework | [`providers/`](../src/aisyslab/providers/) |
| Append-only history; assistant turns replayed verbatim | Anthropic adapter |
| Prompt caching through a stable prefix | Anthropic adapter (`cache_control`); static system prompts |
| Local models through OpenAI-compatible servers | `--provider local` |
| JSONL tracing, cost per completed task | [`core/trace.py`](../src/aisyslab/core/trace.py), [`core/runner.py`](../src/aisyslab/core/runner.py) |
| No silent fallback inside an agent | `run_agent` and every `agent.py` |
| Baseline against agent, same verifier | the whole repository |
