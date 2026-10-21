# Evaluation: measuring a prompt before trusting it

Code: [`src/aisyslab/evals/`](../src/aisyslab/evals/). CLI: `aisyslab eval triage|optimize|judge`.

The labs measure agents on whole tasks. This module measures the smaller component that every
system is full of: one model call with a prompt. The example is a classifier that turns a
maintenance ticket written by an operator ("pump 2 grinding at the drive end") into a failure
code and a priority. Editing the prompt or swapping the model changes its behaviour silently; an
evaluation set is what makes the change visible.

## The dataset

`generate_tickets(seed)` builds 90 tickets from a seed: six failure codes, varied wording,
distractors that name another component ("bearing was replaced last month" on a seal leak) and
urgent cases. Every fifth ticket goes to `dev`, every fifth to `holdout`, the rest to `train`.
The holdout also gets symptom wordings that appear nowhere else, so a prompt that memorised the
training phrases is caught.

A score is tied to its dataset by a content hash (`fingerprint`), to the prompt by a short hash,
and to the model by name. `compare` refuses to compare runs on different datasets.

## Graders: code where code can decide

| Grader | Checks | How |
|---|---|---|
| `schema` | valid JSON, known code, priority 1-3, a reason | code |
| `code` | the expected failure code | code |
| `priority` | the expected priority | code |
| `judged_reason` | the reason names the reported symptom and invents nothing | model judge |

The judge grades only the free-text reason, the one thing code cannot check. The expensive,
noisy grader is kept to the narrowest question.

## How far to trust the judge

A judge is one more model with an error rate. `calibrate` runs it over answers a human already
graded and reports agreement, Cohen's kappa (agreement beyond chance) and every disagreement.
The offline keyword judge agrees on 58% of the human labels with a kappa of 0.17: it passes
"Hydraulic issue reported by the night shift supervisor Anna" because the word *hydraulic*
matches, although the name is invented. Its 1.00 on the reason grader in the triage run is
therefore not evidence of anything, which is the point of calibrating before use.

## Comparing runs and models

`run_eval` returns per-item results, rates per grader, a confusion table and token usage.
`compare(before, after)` lists the items a change fixed and the items it broke: an average that
went up can still hide a regression. `to_markdown` puts several models side by side; a grader a
run did not use shows as `-`, not as a zero.

```bash
aisyslab eval triage --provider openai --models gpt-5-mini,gpt-5 --judge
```

## Prompt optimisation with a holdout

`optimize_prompt` shows a stronger optimiser model the current prompt and the training cases it
failed, and asks for a better prompt. A candidate is kept only if it scores higher on `train`.
At the end the starting and the best prompt are measured on `dev` and `holdout`.

The offline run (seed 0) shows why the holdout exists:

| Split | Starting prompt | Best prompt |
|---|---|---|
| train | 0.17 | 1.00 |
| dev | 0.17 | 1.00 |
| holdout | 0.17 | 0.78 |

The training score went up by construction. Only the holdout, with wordings the loop never saw,
says how much of the gain is real.

```bash
aisyslab eval optimize --provider anthropic --model claude-haiku-5-5 --optimizer-model claude-opus-5-5
```

## Tool descriptions as a prompt

`--promptless` runs any lab agent without its system prompt, with the same tools, environment
and verifier. The difference between the two runs measures how much of the work the tool names,
descriptions and schemas do on their own.

```bash
aisyslab bench plant-database --solver agent --provider openai --promptless
```

## Offline stand-ins

Without `--provider` the commands use deterministic stand-ins, labelled as such in the output:
a rule follower that classifies only by rules written in its prompt, a rule proposer that plays
the optimiser, and the keyword judge. They make the pipeline runnable and testable; they are not
language models, and their numbers say nothing about any model.
