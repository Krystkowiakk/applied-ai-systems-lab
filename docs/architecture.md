# Architecture

```
src/aisyslab/
  core/        harness: messages, model contract, tools, agent loop, tracing, retries, lab contract, runner, bench
  providers/   OpenAI Chat Completions (and any OpenAI-compatible server), Anthropic Messages
  labs/        nine labs: env.py, baseline.py, agent.py, __init__.py each
  cli.py       aisyslab list | show | run | bench | trace
```

No agent framework. The harness is small enough to read in one sitting, and owning it keeps the
things that matter in production under control: what goes into the context, how tool errors are
shown to the model, how budgets are enforced, and what each run costs.

## Messages and providers

`core/messages.py` defines one neutral conversation format: user messages (optionally with images),
assistant turns (text plus tool calls) and tool results (with an error flag). Providers translate it:

- **`providers/openai_chat.py`** speaks Chat Completions with function calling. The same adapter
  serves LM Studio, Ollama and vLLM through `base_url`, which is how the labs run on a local model
  without a key. Temperature is left unset by default because reasoning models reject it.
- **`providers/anthropic_messages.py`** speaks the Messages API through the official SDK. Assistant
  turns are replayed exactly as the API returned them, so reasoning blocks survive the tool loop;
  all tool results of a turn go back in one user message with `is_error` on failures; automatic
  prompt caching is on, and server-side refusal fallback is enabled by default. The default model is
  the current Opus at medium effort; `--model` and `--effort` override both.

Every provider returns a `ModelTurn` with normalised stop reasons (`end_turn`, `tool_use`,
`max_tokens`, `refusal`) and usage that includes cached input tokens.

The conversation is **append-only**. Nothing edits an earlier turn: providers that sign reasoning
blocks reject edited history, and an unchanged prefix is what prompt caching pays for.

## The agent loop

`core/agent.py`, about 200 lines. Each step: call the model, record usage, run every requested tool,
append the results, then check, in code:

| Check | Status when it trips |
|---|---|
| the environment says the goal is reached (e.g. the verifier accepted) | `goal_reached` |
| the model stopped calling tools | `finished` (or `truncated` at the output limit) |
| the model refused | `refused` |
| model calls, tool calls or cumulative input tokens exceed the budget | `step_budget`, `tool_budget`, `token_budget` |
| the provider failed after its own retries | `model_error` |

Budgets come from each lab's card and can be changed per run. None of them is a sentence in a prompt.

**Compaction.** With `compact_above_tokens` set, a step whose prompt grew past the threshold triggers
an observer call: the model writes observations from the transcript (facts with exact values,
actions and results, verifier feedback, what is missing), and the history is replaced by the task
plus those observations. Replacing the whole history, rather than trimming its middle, keeps the
conversation valid for providers that bind reasoning to the prefix.

## Tools

`core/tools.py`. A tool is a name, a description, a pydantic model for its arguments and a handler.

- **Schemas** are generated from the pydantic model, with cosmetic titles removed, objects closed to
  extra keys, and `$ref` pointers inlined, because not every OpenAI-compatible server resolves them.
- **Errors are data.** Invalid arguments, unknown tools and expected failures (`ToolError`, with an
  optional hint) come back to the model as JSON it can act on. A bug inside a handler is caught,
  flagged as an internal error in the trace, and also returned as data: one broken tool never ends
  the run.
- **Output is bounded.** Results longer than the toolbox limit are cut with a hint to narrow the
  query instead of retrying it.

Shared tools: `core/calculator.py` (arithmetic only, parsed with `ast`; the model writes the
expression, code computes it) and the retrieval index in `core/retrieval.py`.

## Failures that are not the model's

`core/resilience.py`. Environments raise `TransientError` for 503 and 429 (with `retry_after`).
`RunContext.call` retries them with exponential backoff and jitter, and traces every retry. Inside an
agent, `RunContext.tool_call` turns exhausted retries into a `ToolError`, so the model learns that a
service is down instead of receiving a stack trace or a false fact. Chaos is injected per environment
from a random stream separate from the data, so a flaky run sees the same world as a clean one.

## Labs

`core/lab.py`. An environment subclasses `LabEnv`: it is built from a seed, routes every API method
through `_api` (counting calls and injecting chaos), and records each verdict. A `Lab` card ties it
to a baseline (`solve(env, ctx)`) and an agent (`solve(env, model, ctx)`) and carries the summary,
techniques, origin and default budgets.

Verifiers return a `Verdict`: accepted or not, a machine-readable code, feedback written for the
solver, and details (per-field status, missing systems, a score for partial grading). The same
verifier judges both solvers.

**Side calls.** When a tool needs its own model call, such as vision in `supply-intake`,
`RunContext.side_call` makes it, traces it and adds its tokens to the run, so cost per completed
task includes it.

## Runs, traces, benchmarks

`core/runner.py` builds the environment, runs one solver and returns a `RunReport`: accepted, score,
status, submissions, API calls, model calls, tool calls, tokens (input, cached, output), retries,
seconds and the trace path. Crashes are reported, not raised, so a benchmark finishes.

`core/trace.py` writes one JSON line per event (`run_start`, `agent_start`, `generation`, `tool`,
`side_generation`, `retry`, `verdict`, `compaction`, `model_error`, `note`, `agent_end`, `run_end`). The
shape follows the session, trace, generation and tool hierarchy that observability tools such as
Langfuse use, so exporting later means writing a sink, not changing call sites.
`aisyslab trace <file>` prints a run as a timeline.

`core/bench.py` aggregates reports per lab, solver and model into a Markdown table, with a cost
column when prices are given (`--price-in`, `--price-out`, USD per million tokens). Prices are not
hard-coded because they change faster than code.

## Choices and their trade-offs

- **Own provider layer instead of an SDK abstraction.** Advanced features (reasoning replay, caching,
  refusal fallbacks, tool-result grouping) differ per provider and arrive faster than abstraction
  libraries follow them. The price is maintaining two adapters.
- **Chat Completions for OpenAI.** It is the dialect every OpenAI-compatible local server speaks.
  The price is not using features specific to newer OpenAI endpoints.
- **Deterministic verifiers.** They make results comparable across solvers and models. The price is
  that the labs are narrower than real work, where a model judge is sometimes the only option.
- **Scripted models in tests.** Tests stay offline, fast and free, and prove the wiring. They prove
  nothing about model quality; that is what the benchmark is for.

## Adding a lab

1. Create `src/aisyslab/labs/<name>/` with `env.py` (subclass `LabEnv`, call `self._api(...)` in
   every API method, return `self._record(Verdict(...))` from the verifier).
2. Write `baseline.py` with `solve(env, ctx)` using `ctx.call` and `ctx.submit`.
3. Write `agent.py` with tools built on `ctx.tool_call` and `ctx.submit(..., as_tool=True)`, ending in
   `run_agent(..., goal_reached=lambda: env.accepted)`.
4. Add the `Lab` card in `__init__.py`, register it in `labs/__init__.py`, and let
   `tests/test_labs.py` pick it up.
