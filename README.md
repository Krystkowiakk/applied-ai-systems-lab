# Applied AI Systems Lab

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**The techniques I use to build AI systems and tools, as of October 2026.** The field changes every
week, so this is a dated snapshot. Each technique runs on a small example from plant operations,
industrial planning or software building.

Every example has a verifier written in code, and most are solved twice: by plain code and by an
LLM agent. Side by side they show where a model earns its place and where code does the job better.

## What's inside

| Area | Techniques | Shown on |
|---|---|---|
| Agents and tools | a tool-use loop with budgets in code, typed tools, retries, traces; OpenAI, Anthropic and local models | every lab |
| Context | a token budget, logs reduced in code before a model reads them, compaction into observations | [`line-incident`](docs/labs.md#line-incident) |
| Retrieval | hybrid search, questions in another language, citations checked in code | [`procedures-rag`](docs/labs.md#procedures-rag), [`inbox`](docs/labs.md#inbox) |
| Research | web search, a polite crawler, a deep research loop, a cited report | [`part-research`](docs/labs.md#part-research) |
| Graph RAG | a graph of dated facts, extraction validated by a parser | [`parts-graph`](docs/labs.md#parts-graph) |
| Memory | observational memory, newer facts replacing older ones | [`shift-handover`](docs/labs.md#shift-handover) |
| Several agents | an orchestrator with specialists; a parallel graph when the plan is known | [`shift-briefing`](docs/labs.md#shift-briefing) |
| Planning | the model proposes, a simulator judges; the model writes a planning program, a sandbox runs it | [`spare-part-run`](docs/labs.md#spare-part-run), [`production-schedule`](docs/labs.md#production-schedule) |
| Data and ML | schema discovery and read-only SQL; a decision tree fitted in code | [`plant-database`](docs/labs.md#plant-database), [`quality-gate`](docs/labs.md#quality-gate) |
| Safety | prompt injection handled by design, human approval, a restricted shell, anonymisation, moderation, an audit log | [`maintenance-portal`](docs/labs.md#maintenance-portal), [`service-repair`](docs/labs.md#service-repair), [`quality-claims`](docs/labs.md#quality-claims) |
| Voice and images | speech-to-text grounded in the plant's vocabulary, a voice cascade, vision checked in code, generated illustrations | [voice](docs/voice.md), [visual](docs/visual.md), [`supply-intake`](docs/labs.md#supply-intake) |
| MCP | the labs as MCP servers, an MCP host, tool discovery | [MCP](docs/mcp.md) |
| Evaluation | eval sets, a calibrated judge, prompt optimisation, batches under a schema, fine-tuning against prompting | [ticket triage](docs/evals.md) |
| Production | a durable job runtime, a model gateway with fallback, streaming, tenants, cron, a web UI | [maintenance request service](docs/runtime.md) |

Every technique, with links into the code: [docs/techniques.md](docs/techniques.md).

## What it shows

- Plain code solves 9 of the 15 labs on every seed, with no model, in milliseconds.
- In the other six, code gets part of the way: keyword rules miss what has to be read, a fixed rule
  ignores the planner's notes, and the first search hit is the wrong answer. That is where the
  model earns its place.
- In a dated live run, `gpt-5-mini` solved 9 of 10 labs, mostly for one or two cents a run.

Numbers, live runs and their limits: [docs/results.md](docs/results.md).

## How I build

- The model proposes, code decides: plans go through simulators, values through validators,
  actions through narrow APIs.
- Policy lives in code, not in the prompt.
- Text from outside is data, never instructions.
- Every run leaves a trace with its tokens and cost.
- A prompt is measured on an eval set before it is trusted.

More in [docs/architecture.md](docs/architecture.md).

## Try it

```bash
git clone https://github.com/Krystkowiakk/applied-ai-systems-lab
cd applied-ai-systems-lab
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

aisyslab list                  # the fifteen labs
aisyslab run supply-intake     # the plain-code solver, no API key needed
aisyslab bench --seeds 0-4     # every plain-code solver on five seeds
pytest                         # 375 offline tests
```

With a model: install `.[all]`, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (or use a local model)
and run `aisyslab run procedures-rag --solver agent --provider openai`. All commands:
[docs/usage.md](docs/usage.md).

## About

Krystian Krystkowiak. Eighteen years in operations, today business operations in manufacturing.

Written in Python. MIT licensed. All names in the examples are fictional.
