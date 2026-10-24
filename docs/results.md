# Results

What plain code and the agents did on the fifteen labs, and how far those numbers go.

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
site map's search), on every seed, with zero tokens. The same baselines pass with 20% of API calls
failing, because retries live in code. In those labs an agent has to justify itself on
flexibility, not capability: what happens when the
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
four-agent shift briefing: six cents and four minutes), instead of milliseconds. The tenth shows
why the verifiers count more than the final answer: in `service-repair` the model read the deny list, opened a path on it anyway, was blocked by the
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
| `part-research` | 2 | second run, fifth submission | $0.03, $0.02 | a verifier bug that rejected a correct answer ([research](research.md#what-the-live-runs-showed)) |
| `parts-graph` | 2 | second run, second submission | $0.03, $0.015 | an extraction prompt that invited type names, the parser's messages, the build report ([graph](graph.md#what-the-live-runs-showed)) |

The modules without a lab were run live too, and their pages say what happened: a search on
the real web, ten voice notes spoken and transcribed and a voice cascade timed stage by stage, two
instruction cards (the first check let letter-like marks through), triage in batches and from an
interview, and a fine-tuning job the provider refused. All the live runs on 24 October cost about
$0.52. Produce your own results, with your key or a local model:

```bash
aisyslab bench --solvers baseline,agent --provider openai --model gpt-5-mini --seeds 0-4
```

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
  ([evals](evals.md#fine-tuning-compared-with-prompting)).
- **The environments are small and synthetic on purpose.** Each trap is there so a failure mode is
  visible and checkable. Production data is messier; the verifiers here are kinder than reality.
- **Token budgets inside environments use one deterministic estimate** so verifier, tools and
  solvers agree. Cost figures in reports come from provider usage.
- **All names are fictional**: the company (Halden), its people and its suppliers.
