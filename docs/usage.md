# Running it

Python 3.11 or newer. Everything on this page except the model commands runs offline,
with no API key.

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
aisyslab eval users               # synthetic users try the procedures assistant, offline stand-ins
aisyslab tenant demo              # a token for the service (shown once)
aisyslab serve --demo             # the service and its web UI on :8000, without a model
aisyslab voice notes              # the voice-note pipelines on recorded transcripts
aisyslab card HP-07               # a work instruction card, drawn in code (not released: unchecked)
pytest                            # 390 offline tests
```

## With a model

Keys are read from the environment, never from files in the repo.

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
aisyslab eval users --provider openai --model gpt-5-mini --user-model gpt-5-nano --effort low --judge --max-cost 0.06   # personas in a chat, a release gate
aisyslab search "ISO 4414 pneumatic safety"                                               # the research SearchAPI on the real web
aisyslab voice notes --speech openai --provider openai --model gpt-5-mini                 # speak, transcribe and file ten voice notes
aisyslab voice ask --speech openai --provider openai --model gpt-5-mini                   # ask by voice, hear the answer, with stage timings
aisyslab card HP-07 --images openai --provider openai --model gpt-5-mini                  # generate the illustration, check it, read the card back
aisyslab serve --provider anthropic --fallback-provider openai                            # the service, with a fallback model
aisyslab trace runs/traces/<file>.jsonl                                                   # replay a run as a timeline
```
