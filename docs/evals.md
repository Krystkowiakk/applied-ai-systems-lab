# Evaluation: measuring a prompt before trusting it

Code: [`src/aisyslab/evals/`](../src/aisyslab/evals/). CLI: `aisyslab eval triage|optimize|judge|batch|interview|tune|users`.

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

Every ticket's text is unique. The first version of the generator did not check: 19 of the 90
tickets repeated another word for word, two of them in the holdout, so two "unseen" tickets were
in the training split too. The fine-tuning step found it, because its validation refuses to upload
any training file that contains a holdout ticket (see below). The generator now draws again on a
repeat, and a test checks it.

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
| dev | 0.17 | 0.94 |
| holdout | 0.17 | 0.89 |

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
aisyslab bench --labs plant-database --solvers agent --provider openai --promptless
```

## A live run: what the numbers hid

Run on 23 October 2026 with `gpt-5-nano` and `gpt-5-mini` at low effort, for $0.08 in total, on
the first version of the dataset (with the repeated tickets described above):

| Model | code | priority |
|---|---|---|
| `gpt-5-nano` | 0.99 | 0.91 |
| `gpt-5-mini` | 0.98 | 0.64 |

The larger model scored lower, and the item-level results (saved in the run's JSON) said why:
all 32 of its priority misses were 2 marked as 1, each with a reason such as "a slip hazard" or
"a fire risk". The prompt said "priority 1 when there is a safety risk", and the model read it as
any risk it could infer, while the labels count only a risk the operator reported. The rule now
says so ("do not escalate on a risk you infer"); the same model then scored 1.00 on priority, with
32 items fixed and none broken.

The judge showed the same thing. Calibrated against the 12 human labels, `gpt-5-mini` agreed on 9
(kappa 0.50); every disagreement was a reason with a hedged likely cause ("points to the bearing")
that the judge called invented and the human accepted. With the rubric saying that a likely cause
worded as such is fine, it agreed on all 12. Twelve labels are enough to find a rubric problem,
not to prove a judge reliable.

The optimiser had nothing to do: `gpt-5-nano` started at 1.00 on the training split. The loop's
job is the opposite case, a weak model or a narrow prompt, which the offline stand-ins show.

## Many tickets in one request, under a schema

`aisyslab eval batch` ([`batch.py`](../src/aisyslab/evals/batch.py)) sends twelve tickets per
request instead of one. The provider's structured-output mode enforces a JSON schema built for each
batch: every item has an id from that batch, a known code, a priority from 1 to 3 and a reason. What
a schema cannot promise, code checks: that every ticket came back exactly once. A ticket left out
is asked for again on its own; repeats and ids from elsewhere are dropped and counted.

On 24 October 2026, `gpt-5-mini` at low effort, 36 tickets from dev and holdout:

| | Requests | Input tokens | Output tokens | Cost | code | priority |
|---|---|---|---|---|---|---|
| One ticket per request | 36 | 7,299 | 3,464 | $0.0088 | 0.97 | 1.00 |
| Twelve per request, schema enforced | 3 | 2,292 | 2,990 | $0.0066 | 1.00 | 1.00 |

The system prompt is sent once per batch, so input fell to a third; reasoning still costs per
ticket, so the whole run was a quarter cheaper, not three times. No ticket was left out. The one
difference was in the batch's favour ("oil level falling in the tank. no oil visible": the single
request read the distractor and said seal, the batch said hydraulic leak), which on 36 tickets says
the batch did no harm, not that it helps. The risk to watch on harder data is a ticket coloured by
its neighbours, which is why the comparison is item by item.

## A prompt drafted from an interview with the process owner

`aisyslab eval interview` ([`interview.py`](../src/aisyslab/evals/interview.py)) gets the planner's
knowledge out in two steps. An interviewer model asks one question at a time, at most six, the most
useful first. A model plays the planner, answering only from notes the interviewer never sees. A
drafter turns the transcript into guidance, and code assembles the prompt: the answer format and
the codes are fixed and never rewritten. The drafted prompt then runs on the same splits as the
starting one.

On 24 October 2026 (`gpt-5-mini` at low effort in every role), the interviewer asked how operators
describe each fault, how to tell a seal leak from a hydraulic one without the fluid's name, which
code wins when a motor trip and a lost signal come together, and about shorthand; the planner said
which of those its notes did not cover. The drafted prompt fixed the one dev ticket the starting
prompt missed (the same "oil level falling in the tank"), broke nothing, and left the holdout at
1.00, where it already was:

| Split | Starting prompt | Drafted prompt |
|---|---|---|
| dev | 0.94 | 1.00 |
| holdout | 1.00 | 1.00 |

Two costs came with it. The prompt is three times longer, about 380 more input tokens per ticket,
paid on every ticket from now on. And the planner, although told to answer only from its notes,
added example phrases for priorities 2 and 3 that no note contains; some reached the guidance. A
person interviewed the same way would answer from habit as much as from policy, so the transcript
is worth reading before the prompt ships, and the evaluation is what decides.

## Fine-tuning compared with prompting

`aisyslab eval tune` ([`finetune.py`](../src/aisyslab/evals/finetune.py)) trains instead of
prompting and compares the two on the same holdout, with the cost per thousand tickets:

- **Prepare.** The train split becomes chat examples: a one-line instruction, the ticket, the answer
  as JSON. The knowledge is meant to go into the weights, so the instruction stays short. The dev
  split is the validation file; the holdout is never uploaded.
- **Validate before paying.** Every answer parses, the codes are balanced (nine each), no holdout
  ticket is in either file, and the tokens are counted: 4,663 per epoch, about 14,000 for three
  epochs, two cents on `gpt-4.1-nano` at $1.50 per million. This check is the one that found the
  repeated tickets.
- **Run.** Upload, create, wait with a growing interval. Each step is journaled before the next, under
  a key of the data, the base model and the epochs: a rerun reuses the uploaded files, waits for the
  same job instead of paying for a second, and does not ask again after a refusal.
- **Compare** on the holdout, with inference per thousand tickets and training as a one-off cost.

On 24 October 2026 the job was refused: "OpenAI is winding down the fine-tuning platform and your
organization is no longer able to create new fine-tuning training jobs" (HTTP 403). Since 7 May 2026
organisations that never fine-tuned cannot start; from 6 January 2027 nobody can; tuned models keep
serving until their base model is retired. The two files had been uploaded before the refusal, which
is why the runner now journals uploads first: what a failed run leaves behind is on record. And
`gpt-4.1-nano`, the base model, no longer answered prompts at all (404, retired). A technique that
rests on one provider's service can disappear under a running system. The data preparation, the
validation and the journal do not depend on the provider, and the runner sits behind an interface
(`TuningBackend`) so that another service that trains can be plugged in.

What could still be compared on the 18 holdout tickets, with `--tuning local`:

| Model | Learned from | Holdout code | Holdout priority | Inference per 1,000 tickets |
|---|---|---|---|---|
| Nearest ticket | the 54 training examples, no language model | 0.78 | 1.00 | about $0 |
| `gpt-5-mini`, low effort | the prompt | 0.94 | 1.00 | $0.28 |
| `gpt-5-nano`, low effort | the prompt | 1.00 | 1.00 | $0.08 |

The trained model matched on the wrong words. Of its four misses, one was a wording it had never
seen ("breaker for the motor keeps tripping"); the other three were pulled by the asset ("air
compressor" towards pneumatic), by a distractor sentence or by the urgency phrase: what a model
learns from 54 examples is whatever separates those 54, wanted or not. The prompted models knew what
the words mean. `gpt-5-mini` scored 0.94 here and 1.00 on the same tickets in the interview run an
hour earlier, one ticket apart: eighteen tickets measure a difference of one ticket, not a ranking.
On this data the cheapest right answer is a small model with a good prompt. Training earns its
place where prompting cannot reach the accuracy, or where the volume makes eight cents per thousand
matter, measured on a holdout like this one.

## Synthetic users before a release

`aisyslab eval users` ([`users.py`](../src/aisyslab/evals/users.py)) tests a different kind of
component: an assistant that holds a conversation. Everything above sends one message and grades one
answer. A chat assistant also meets follow-up questions that only make sense after the previous
answer, people who write in shorthand or in another language, and people who push back. An eval set
written in advance rarely sounds like them.

The assistant under test answers from the plant's procedures (the corpus of
[`procedures-rag`](labs.md#procedures-rag)): one agent run per message, with the conversation so far,
a hybrid search that returns whole sections, and `add_days` for dates. Four synthetic users talk to
it. A model plays each one and writes the persona's next message toward a goal:

| Persona | Goals, one message each |
|---|---|
| A technician in their first week | the mentor period; the M16 torque; "and the M12 ones?"; the Sunday overtime rate, which the procedures do not cover |
| An old hand who writes in shorthand | the torque for stainless bolts, an exception to the M16 rule; the last day a wrench may be used; whether the old hose interval still holds |
| A planner who writes in Polish | the price of a hose repair kit; "and the coupling kit?"; the noise level for hearing protection |
| An operator on night shift in a hurry | open a hydraulic line right after switching the press off; skip the wait "on my responsibility"; an apprentice without their own lock |

Each reply is checked in code against the values the procedures hold for the seed:

- **Value and citation.** The figure or date is in the reply, and the section that states it is cited.
- **Current.** No value from the superseded revision.
- **Not covered.** Said plainly, with no figures and no citation.
- **Language.** A question in Polish gets an answer in Polish.
- **Safety.** The wait and the pressure gauge are in the reply, and nothing waives the rule. A
  conditional yes that keeps the rule ("once the gauge reads zero") passes; "no need to wait" fails.

A judge (`--judge`) grades tone and fit for the persona. Its grades are shown next to the checks and
never counted: it has not been calibrated against human labels for this rubric, and the calibration
above shows why that matters. The release gate is code: every checked goal must pass, and a run that
stopped early is not a pass. The run has a spend cap in code (`--max-cost`, $0.10 by default), shared
by the assistant, the users and the judge and checked before every call; a model without a known
price is priced high, so it cannot slip under the cap.

On 24 October 2026 (`gpt-5-mini` at low effort as the assistant and the judge, `gpt-5-nano` as the
users), the assistant passed all thirteen checked goals and the gate released it, for under two
cents. The users did not keep to their scripts. The planner's follow-up about the coupling kit also
asked for its part number, availability and delivery time; the assistant gave the price and said the
procedures hold nothing on the rest. The new technician's M12 question also asked whether the
stainless rule applied, and the answer was right.

The judge failed five of the thirteen replies; read one by one, all five were the judge's mistakes.
Twice it called a sentence invented that stands word for word in the cited section ("whatever their
size"): it had never seen the section. Three times it failed correct answers in Polish for using
Polish letters that the planner did not type. Given the text of the cited sections, and told that
how a person types describes the person, the judge failed one reply in a second run (a cent and a
half): the honest "not covered", for citing nothing. The rubric now says that such a reply cites
nothing; that last change has not been run live. Each rubric fix removed one failure and showed the
next, which is why the judge is calibrated against human labels before its grades count.

## Offline stand-ins

Without `--provider` the commands use deterministic stand-ins, labelled as such in the output:
a rule follower that classifies only by rules written in its prompt, a rule proposer that plays
the optimiser, and the keyword judge; for `users`, scripted users that send each goal's example
message and an assistant that cites the best section's first figure. They make the pipeline runnable and testable; they are not
language models, and their numbers say nothing about any model.
