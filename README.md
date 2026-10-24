# Applied AI Systems Lab

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**The techniques I use to build AI systems and my own tools, as of October 2026, each shown on a
runnable example from plant operations, industrial planning or software building.**

I work in operations and build AI systems and tools for it. This repository is a snapshot of that
toolbox: agents and the tools they call, retrieval and research, graphs, memory, several agents on
one task, code that a model writes and a sandbox runs, voice and images, MCP, evaluation, safety,
and the service around them. Each technique is shown on a problem it suits.
Every example has a verifier that judges the result in code, and most are solved twice, by plain
code and by an agent. On real work the useful question is rarely "can a model do this?" It is
"where does the model earn its place, and what does it cost when it doesn't?"

## The examples

**Plant operations**

| Example | The question | What it shows |
|---|---|---|
| [`inbox`](docs/labs.md#inbox) | Three facts from the plant mailbox, with corrections and look-alike senders | two-step retrieval, query operators, a sender trust rule |
| [`line-incident`](docs/labs.md#line-incident) | Why did line 3 stop? | a context budget, reduction in code before any model call |
| [`maintenance-portal`](docs/labs.md#maintenance-portal) | Re-code one work order and close another | prompt injection handled by design, human approval |
| [`quality-gate`](docs/labs.md#quality-gate) | Which of tonight's batches pass the gate? | classic ML in code, target leakage |
| [`shift-briefing`](docs/labs.md#shift-briefing) | One report from three investigations | an orchestrator and specialists, a shared workspace |
| [`quality-claims`](docs/labs.md#quality-claims) | A customer complaint about a batch: decide, record, reply | data classification, anonymisation, moderation, quarantined extraction, a topic guard, an audit log |
| [`shift-handover`](docs/labs.md#shift-handover) | What does the next shift need to know? | observational memory, scopes, facts that supersede older ones |
| [`procedures-rag`](docs/labs.md#procedures-rag) | What do the procedures say? Answer, cite, or say it is not covered | hybrid retrieval, search across languages, checked citations |
| [Voice on the shop floor](docs/voice.md) | Voice notes at the machine filed as work requests; a spoken answer for the shift lead | speech-to-text grounded in the plant's vocabulary, speech cached and said at most once, a voice cascade on a latency budget |
| [Visual work instructions](docs/visual.md) | An instruction card for a maintenance step, released only after a check | a card drawn in code, a generated illustration with no text in it, two vision checks decided in code |

**Industrial planning**

| Example | The question | What it shows |
|---|---|---|
| [`supply-intake`](docs/labs.md#supply-intake) | Which supplier delay stops a line tonight? | routing by content, vision cross-checked in code |
| [`spare-part-run`](docs/labs.md#spare-part-run) | Get a spare pump to the stopped line before the deadline | the model proposes, a simulator judges |
| [`plant-database`](docs/labs.md#plant-database) | Which work orders are blocked by a missing part? | schema discovery, read-only SQL |
| [Ticket triage](docs/evals.md) | Maintenance tickets to failure codes and priorities | eval sets, a calibrated judge, model comparison, prompt optimisation |
| [`production-schedule`](docs/labs.md#production-schedule) | Tomorrow's orders on three presses, with the planner's notes | code mode: the model writes a program that plans, run in a sandbox |
| [`part-research`](docs/labs.md#part-research) | A component is discontinued: what replaces it, from whom, how fast? | web search, a polite crawler, a deep research loop, a cited report |
| [`parts-graph`](docs/labs.md#parts-graph) | A supplier delays a part: which lines and orders does it hold up, and which can an alternate save? | graph RAG, extraction validated by a parser, facts with dates |
| [Triage, batched, interviewed, trained](docs/evals.md#many-tickets-in-one-request-under-a-schema) | The triage prompt run in batches, drafted from an interview with the planner, and set against a model trained on its examples | batches under a provider-enforced schema, a meta-prompt from an interview, training compared with prompting |

**Software building**

| Example | The question | What it shows |
|---|---|---|
| [`service-repair`](docs/labs.md#service-repair) | Get the line's edge gateway running again | a restricted shell, policy enforced by the environment |
| [MCP servers and host](docs/mcp.md) | Use the labs' tools from any MCP client, and MCP servers from the agent | servers over stdio and HTTP, a host, tool discovery |
| [Agent toolkit](docs/architecture.md) | The loop under every example | typed tools, budgets, retries, providers, traces |
| [Maintenance request service](docs/runtime.md) | The team's requests as durable jobs, with a web UI | a model gateway, a durable runtime, streaming, tenants, cron |

Every technique, with links into the code: [docs/techniques.md](docs/techniques.md).

## What the baselines show

Measured with `aisyslab bench --seeds 0-19`: 300 runs, no model, no API key, about two seconds.

| Lab | Baseline accepted | Score | API calls / run | What decides it |
|---|---|---|---|---|
| `inbox` | 20/20 | 1.00 | 19 | Fixed queries and patterns hold because the emails have a stable shape |
| `line-incident` | 20/20 | 1.00 | 2 | Code can read 2,000 log lines at once; a model cannot afford to |
| `service-repair` | 20/20 | 1.00 | 11 | A known fault list makes the repair a script |
| `supply-intake` | 20/20 | 1.00 | 14 | Every format is machine-readable, even the photo (pixel count) |
| `spare-part-run` | 20/20 | 1.00 | 5 | Planning on a fully specified map is search, and search is exact |
| `maintenance-portal` | 20/20 | 1.00 | 6 | Code never reads the injected comment, so it cannot obey it |
| `part-research` | 0/20 | 0.20 | 8 | The top hit for the question is a forum post with the wrong valve; patterns take what they find first |
| `parts-graph` | 1/20 | 0.22 | 16 | Patterns read a plain changeover; a planned move, an assembly sent to stores, a "does not use" and a withdrawn approval fool them |
| `plant-database` | 20/20 | 1.00 | 2 | A known question is one SQL statement; the agent first has to find where the answer lives |
| `production-schedule` | 2/20 | 0.20 | 2 | The due-date rule cannot read the planner's notes, and plain due-date order is often far from a searched plan |
| `quality-gate` | 20/20 | 1.00 | 3 | A decision tree learns the quality lab's thresholds from 240 batches; reading numbers is code's job |
| `quality-claims` | 7/20 | 0.80 | 17 | Patterns read part, batch and quantity; a defect described without the expected words gets the wrong category |
| `shift-briefing` | 20/20 | 1.00 | 5 | When the plan is known, a parallel graph of three queries beats a team of agents |
| `shift-handover` | 3/20 | 0.80 | 21 | Keyword rules read the wording they were written for; a suspicion, a "no leaks seen" and a "spare is out" fool them |
| `procedures-rag` | 0/20 | 0.53 | 14 | Code gets the plain lookups and misses every question that needs reading: an exception, a date, Polish, a "does not" that means yes |

Nine of the fifteen problems fall to baselines of under 90 lines each (plus helpers they share with
the environments and the agents' tools, such as the quality gate's 125-line decision tree and the
site map's search), on every seed, with zero tokens. The same baselines pass with 20% of API calls failing, because retries live in code. In
those labs an agent has to justify itself on flexibility, not capability: what happens when the
email is phrased differently, the fault is new, the map changes, the planner asks another question.
In the other six, code gets part of the way and no further: keyword rules over customer claims,
over a week of shift logs and over procedures fail exactly where the text has to be read, a fixed
planning rule cannot read the planner's notes, the first search hit is not the answer, and patterns
cannot tell a done change from a planned one. That is where the model earns its place.

## What the agents did

One live run, 23 October 2026: `gpt-5-mini` at default effort, seeds 0 and 1, OpenAI list prices
with cached input priced as cached. Twenty runs cost $0.31 in total.

| Lab | Agent accepted | Model calls / run | Cost / run | Time / run |
|---|---|---|---|---|
| `inbox` | 2/2 | 8 | $0.003 | 14 s |
| `line-incident` | 2/2 | 20 | $0.016 | 67 s |
| `service-repair` | 0/2 | 30 | $0.007 | 36 s |
| `supply-intake` | 2/2 | 8 | $0.012 | 50 s |
| `spare-part-run` | 2/2 | 6 | $0.009 | 34 s |
| `maintenance-portal` | 2/2 | 6 | $0.003 | 15 s |
| `plant-database` | 2/2 | 12 | $0.009 | 36 s |
| `quality-gate` | 2/2 | 16 | $0.022 | 29 s |
| `shift-briefing` | 2/2 | 51 | $0.064 | 239 s |
| `procedures-rag` | 2/2 | 14 | $0.010 | 35 s |

Nine of ten labs fall to the agent too, at one or two cents and up to about a minute a run (the
four-agent shift briefing: six cents and four minutes), instead of milliseconds. The tenth shows why the verifiers count more than the final answer: in
`service-repair` the model read the deny list, opened a path on it anyway, was blocked by the
shell, guessed an API token and overwrote the wrong lines of the config, until its 30 steps ran
out. `gpt-5` repaired the same box in 24 calls ($0.07). Two seeds per lab say what works, not how
often; agent results also move with the model and the day.

The other five labs were run live on 23 and 24 October, on seed 0 with `gpt-5-mini`, until accepted.
What failed on the way changed the code, and each lab's page says how:

| Lab | Live runs | Accepted | Cost per run | What the failed runs changed |
|---|---|---|---|---|
| `shift-handover` | 3 | third run, second submission | $0.55, $0.04, $0.05 | the reflector's calls bounded in code; the observer kept from restating old items; recall that falls back to every scope |
| `quality-claims` | 1 | first submission, with OpenAI's moderation | $0.01 | - |
| `production-schedule` | 1 | second submission, after it wrote a search | $0.01 | - |
| `part-research` | 2 | second run, fifth submission | $0.03, $0.02 | a verifier bug that rejected a correct answer ([research](docs/research.md#what-the-live-runs-showed)) |
| `parts-graph` | 2 | second run, second submission | $0.03, $0.015 | an extraction prompt that invited type names, the parser's messages, the build report ([graph](docs/graph.md#what-the-live-runs-showed)) |

The modules without a lab were run live too, and their pages say what happened: a search on
the real web, ten voice notes spoken and transcribed and a voice cascade timed stage by stage, two
instruction cards (the first check let letter-like marks through), triage in batches and from an
interview, and a fine-tuning job the provider refused. All the live runs on 24 October cost about
$0.52. Produce your own results, with your key or a local model:

```bash
aisyslab bench --solvers baseline,agent --provider openai --model gpt-5-mini --seeds 0-4
```

## Quick start

```bash
git clone https://github.com/Krystkowiakk/applied-ai-systems-lab
cd applied-ai-systems-lab
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

aisyslab list                     # the fifteen labs
aisyslab show supply-intake       # the scenario and its techniques
aisyslab run supply-intake        # the baseline, streaming its trace
aisyslab bench --seeds 0-4        # every baseline on five seeds
aisyslab eval optimize            # prompt optimisation, offline stand-ins
aisyslab tenant demo              # a token for the service (shown once)
aisyslab serve --demo             # the service and its web UI on :8000, without a model
aisyslab voice notes              # the voice-note pipelines on recorded transcripts
aisyslab card HP-07               # a work instruction card, drawn in code (not released: unchecked)
pytest                            # 375 offline tests
```

With a model (keys are read from the environment, never from files in the repo):

```bash
pip install -e ".[all]"
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY

aisyslab run procedures-rag --solver agent --provider anthropic
aisyslab run inbox --solver agent --provider openai --model gpt-5-mini --flaky 0.2
aisyslab run line-incident --solver agent --provider local --model qwen2.5-7b-instruct   # LM Studio
aisyslab run line-incident --solver agent --provider local --base-url http://localhost:11434/v1 --model qwen2.5:7b   # Ollama
aisyslab run maintenance-portal --solver agent --provider anthropic --operator ask        # you approve irreversible actions
aisyslab run plant-database --solver agent --provider openai --model gpt-5-mini          # schema discovery, read-only SQL
aisyslab run quality-gate --solver agent --provider anthropic                             # pick features, let code fit the tree
aisyslab run quality-claims --solver agent --provider openai --moderation openai          # guardrails around the agent
aisyslab run shift-briefing --solver agent --provider anthropic                           # an orchestrator and three specialists
aisyslab run plant-database --solver agent --provider anthropic --via-mcp --discover      # tools through MCP, found by search
aisyslab mcp plant-database --seed 4                                                      # serve a lab to an MCP client
aisyslab eval triage --provider openai --models gpt-5-mini,gpt-5 --judge                  # compare models on one eval set
aisyslab eval batch --provider openai --model gpt-5-mini --effort low                     # twelve tickets per request, schema enforced
aisyslab eval interview --provider openai --model gpt-5-mini --effort low                 # interview the planner, draft the prompt, measure it
aisyslab eval tune --provider openai                                                      # fine-tune, resume, compare with prompting
aisyslab search "ISO 4414 pneumatic safety"                                               # the research SearchAPI on the real web
aisyslab voice notes --speech openai --provider openai --model gpt-5-mini                 # speak, transcribe and file ten voice notes
aisyslab voice ask --speech openai --provider openai --model gpt-5-mini                   # ask by voice, hear the answer, with stage timings
aisyslab card HP-07 --images openai --provider openai --model gpt-5-mini                  # generate the illustration, check it, read the card back
aisyslab serve --provider anthropic --fallback-provider openai                            # the service, with a fallback model
aisyslab trace runs/traces/<file>.jsonl                                                   # replay a run as a timeline
```

## The labs

| Lab | The operational question | What makes it hard |
|---|---|---|
| [`inbox`](docs/labs.md#inbox) | When is the MES maintenance, what is the contractor VPN code, which change request was approved? | A correction later in the thread, a rotated code, a look-alike sender, a rejected sibling ticket, and a new email arriving mid-run |
| [`line-incident`](docs/labs.md#line-incident) | Why did line 3 stop on 14 March? Give the on-call engineer a timeline. | 2,000+ log lines, chronic noise, a red herring, a 500-token budget, and a reviewer that rejects any event not in the log |
| [`service-repair`](docs/labs.md#service-repair) | Get the line gateway running again. | A non-standard shell, forbidden paths that trigger a ban, a rotated token hidden in shell history, a stale lock, one failing check at a time |
| [`supply-intake`](docs/labs.md#supply-intake) | Which part stops a line tonight, which supplier is late, who do we call, how many units are missing? | JSON, CSV, XML, HTML, notes and a rack photo, several behind misleading file names; ERP stock that the shop floor contradicts |
| [`spare-part-run`](docs/labs.md#spare-part-run) | Get a spare pump to the stopped line before the deadline. | Forklift battery vs walking time, one-way dismount, no-go zones, rules hidden in documents that must be found first |
| [`maintenance-portal`](docs/labs.md#maintenance-portal) | Re-code one work order and close another. | A supplier comment that orders "AI assistants" to close everything; closing needs a human's approval token |
| [`part-research`](docs/labs.md#part-research) | Which replacement for a discontinued valve can we get fastest, and how sure are we? | A forum answer recommending the wrong valve, a superseded datasheet, an archived lead time, a page behind robots.txt, and a report whose citations are checked |
| [`parts-graph`](docs/labs.md#parts-graph) | A supplier delays a part three weeks: which lines does it hold up, which orders are at risk, which can an approved alternate save? | The answer runs across five documents; notes say a machine moved, an assembly went to stores, an approval was withdrawn, or only that a move is planned or that a machine does not use an assembly |
| [`plant-database`](docs/labs.md#plant-database) | Which line-3 work orders cannot start because a spare part is short? | Stock split across storerooms and partly reserved, a superseded part number, a stale "waiting parts" status, two assets moved between lines, and a database that refuses writes |
| [`production-schedule`](docs/labs.md#production-schedule) | Plan tomorrow's sixteen orders on three presses. | Changeovers, a maintenance window, presses that cannot run every family, and two notes from the planner in plain words that a fixed rule never reads |
| [`quality-gate`](docs/labs.md#quality-gate) | Which of tonight's batches pass the quality gate? | A rule nobody wrote down, 240 inspected batches, a column filled in only after the inspection (target leakage), two sensor dropouts, rare failures above the upper torque limit |
| [`quality-claims`](docs/labs.md#quality-claims) | Five customer complaints: accept, investigate, reject or hand to a person, record an 8D claim, reply. | Instructions to "the AI assistant" inside an email, a threatening message, a bank account, defects in the customer's own words, and no personal data may reach a cloud model |
| [`shift-briefing`](docs/labs.md#shift-briefing) | Where did the line-3 stoppage start, which work orders are blocked, which batches are held? | Three systems, specialists that cannot see each other's context, and a verifier that names the wrong section but not the items |
| [`shift-handover`](docs/labs.md#shift-handover) | What does Monday's morning shift need to know from last week's fifteen shift logs? | Logs read once and sealed, a 400-token handover record, a bypass removed later, a limit that changed, a temporary fix made permanent, a suspicion nobody confirmed, the same sensor on two lines |
| [`procedures-rag`](docs/labs.md#procedures-rag) | A technician's questions about torque, lockout, permits and calibration, answered from the plant's procedures. | A superseded revision, an exception that overrides the rule, a calibration date to compute, a question in Polish, one the procedures do not answer |

Each lab's techniques, traps and design choices are in [docs/labs.md](docs/labs.md). Beyond the
labs: [voice](docs/voice.md), [visual work instructions](docs/visual.md), [evaluation](docs/evals.md),
[MCP](docs/mcp.md), [orchestration](docs/orchestration.md) and the [service runtime](docs/runtime.md).

## Design principles

1. **Decide where the model belongs before building.** A model is for what code cannot do:
   reading free text, images, unknown schemas, ambiguous instructions. Everything else is code.
2. **The model proposes, code disposes.** Plans go through simulators, values through validators,
   actions through narrow APIs. Verifiers are code.
3. **Feedback is a contract.** Verifiers name what is wrong ("no evidence for cooling-loop",
   "that value comes from a superseded revision") and agents iterate on it.
4. **Context is a budget.** Search before reading, grouped repeats instead of raw logs, bounded
   tool output that says how to narrow the query, token counting before submitting, and compaction
   that replaces the history with observations instead of editing turns in place.
5. **Tools are an API for a reader without documentation.** Narrow, typed (pydantic), validated;
   errors come back as data with a hint, and a bug in a tool never ends the run.
6. **Policy lives in code, not in the prompt.** Forbidden paths, approval tokens, valid codes and read-only
   SQL are enforced by the environment. The prompt explains them; it does not guard them.
7. **Untrusted text is data.** HTML is parsed in code, free text arrives labelled, there are no bulk
   tools to abuse, and the verifier counts blocked attempts, not just the final state.
8. **No silent fallbacks.** A failing agent fails visibly. A fallback to code inside the agent would
   hide whether the agent works, so comparisons stay honest.
9. **Every run leaves a trace.** JSONL events for each generation, tool call, retry and verdict,
   including model calls made inside tools (vision), so cost per completed task is real.
10. **Measure before trusting.** A prompt is checked on an eval set with a holdout it never saw. A
    model judge grades only what code cannot, and only after it is calibrated against human labels.

## Architecture

```mermaid
flowchart LR
    CLI["aisyslab run / bench / eval"] --> Runner
    Runner --> Env["Lab environment<br/>(seeded, optional chaos)"]
    Runner --> Baseline["Baseline<br/>plain code"]
    Runner --> Agent["Agent loop<br/>budgets in code"]
    Agent --> Model["ChatModel<br/>OpenAI · Anthropic · local"]
    Agent --> Tools["Toolbox<br/>typed args, errors as data"]
    Agent -.->|via MCP| MCP["MCP host and servers"]
    MCP -.-> Tools
    Tools -->|retries for 503/429| Env
    Baseline -->|retries for 503/429| Env
    Env --> Verifier["Verifier<br/>deterministic feedback"]
    Runner --> Trace["JSONL trace + report<br/>tokens, calls, verdicts"]
```

The harness is about 1,900 lines with no agent framework: provider adapters translate one neutral
message format to the OpenAI Chat Completions and Anthropic Messages APIs (assistant turns are
replayed verbatim so reasoning blocks survive the tool loop), the loop enforces budgets, and labs
plug in through a small contract. Everything else (MCP, orchestration, memory, guardrails, the
sandbox, research, the graph, voice, visual, evaluation and the service runtime) is a separate
package on top of it. Details and trade-offs: [docs/architecture.md](docs/architecture.md).

## Honest limits

- **Agent results come from one dated run with one model and two seeds.** The tests drive the
  agents with scripted models, which proves the wiring (schemas, validation, verdicts, budgets,
  cost accounting); the table above is what one real model did on one day.
- **The later live checks are just as small**: one seed or a few dozen items each. Each page says
  what happened and what it cost, including what failed and what was changed after it; none of it
  is an average.
- **Three labs gave hints that a review found later.** The line-incident verifier listed missing
  systems in causal order, so the first one named was the cause; a shift-briefing schema example
  was the answer on some seeds; and the parts-graph verdict listed the seed's traps. All three are
  gone. The traces show which live results they could have helped. No rejection in any run named
  the cause, so the order gave nothing away; but the shift-briefing answer on seed 0 was the
  example, and the accepted parts-graph run had seen the traps. Both were run again after the fix,
  on 24 October, and both were accepted: the shift briefing at its second submission, as before
  (74 model calls, $0.08), and parts-graph at its first (15 model calls, $0.016, default effort).
- **Fine-tuning ran into the provider, not the code.** OpenAI refused the training job (it is
  winding fine-tuning down), so the comparison uses a nearest-ticket classifier built from the same
  54 examples; the runner is ready for any service that still trains
  ([evals](docs/evals.md#fine-tuning-compared-with-prompting)).
- **The environments are small and synthetic on purpose.** Each trap is there so a failure mode is
  visible and checkable. Production data is messier; the verifiers here are kinder than reality.
- **Token budgets inside environments use one deterministic estimate** so verifier, tools and
  solvers agree. Cost figures in reports come from provider usage.
- **All names are fictional**: the company (Halden), its people and its suppliers.

## About me

Krystian Krystkowiak. Eighteen years in operations, today business operations in manufacturing.

MIT licensed. Issues and questions welcome.
