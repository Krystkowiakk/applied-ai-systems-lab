# Voice on the shop floor

Code: [`src/aisyslab/voice/`](../src/aisyslab/voice/). CLI: `aisyslab voice notes` (offline, it
replays recorded transcripts) and `aisyslab voice ask` (live).

## Speech-to-text grounded in the plant's vocabulary

The transcriber gets the vocabulary up front: every machine and part ID and every machine name, as a
prompt for models that read one (`gpt-4o-transcribe`) or as keywords for models that take a list
(`gpt-transcribe`). What comes back is still spoken language, so grounding in code follows
([`vocabulary.py`](../src/aisyslab/voice/vocabulary.py)):

- spoken numbers become digits ("P ten forty", "P one oh four oh"), and IDs are read in any of the
  ways a transcriber writes them (`MC14`, `M.C. 14`, `P 1,040`);
- every ID is checked against the registry; one it does not hold is tried with "-teen" and "-ty"
  swapped (fourteen and forty sound alike next to a press), and exactly one match is a correction;
- a part number whose registry name is not in the note, while another part's name is, becomes a
  question: the note says "relief valve", and the number heard belongs to a pressure sensor;
- machines named by what they are ("the transfer press") are found by name, and "the press on line
  one", when line one has two, becomes a question.

Nothing is guessed. Whether to ask the technician is decided in code: a request without a certain
machine cannot be filed. The model only writes the question.

## The evaluation: ten voice notes

[`notes.py`](../src/aisyslab/voice/notes.py) holds ten notes written the way technicians talk at a
machine, each with the request it should become (machine, part, urgency, whether to ask). Four
pipelines run on the same audio, each adding one step. The live run on 24 October 2026: notes spoken
by `gpt-4o-mini-tts` in five voices, transcribed by `gpt-4o-transcribe` with and without the
vocabulary, read by `gpt-5-mini` at low effort.

| Pipeline | Machine | Part | Urgency | Ask when unsure | All right |
|---|---|---|---|---|---|
| Transcript and patterns | 5/10 | 10/10 | 10/10 | 6/10 | 5/10 |
| + grounding in the registry | 9/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| + the vocabulary in speech-to-text | 9/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| + a model reading the note | 10/10 | 10/10 | 9/10 | 10/10 | 9/10 |

What it showed:

- On clean speech, `gpt-4o-transcribe` wrote every part number right without help. The vocabulary
  fixed three words ("NC 14", "Dear pump", "Asembly") and wrote every ID in the registry's own
  format, but grounding had already settled those notes, so it changed no request. The transcript's
  losses were machines named by what they are and the ambiguous press: grounding's job.
- Patterns take the first machine in "MC-15, sorry, MC-13"; the model takes the correction.
- Both transcripts of one note lost its last sentence, "It's still running". Without it the model
  read a knocking drive as "soon" instead of "routine". What speech-to-text loses, nothing after it
  recovers.
- Reading the outputs changed one rule. The model asked about a missing part although the machine
  was known; whether to ask is now derived in code, and the model writes only the question.
- The notes are synthetic speech without noise. A press running next to the technician is harder,
  so these numbers are a floor for the error rate, not a ceiling.

The run cost about four and a half cents: speech for 90 seconds of audio, two transcriptions per
note, ten extraction calls. The transcripts are recorded in
[`recorded/voice-notes.json`](../src/aisyslab/voice/recorded/voice-notes.json), so the pipelines
after speech-to-text run offline and in the tests.

## Speech that is paid for once and said once

[`delivery.py`](../src/aisyslab/voice/delivery.py): `SpeechCache` keeps audio under a key of the
synthesizer, the voice, the style instructions and the text. The same sentence in the same voice is
synthesized once, also across restarts; a changed voice or instruction is a new key, never a stale
hit. `DeliveryJournal` marks each utterance `delivering` before it plays and `delivered` after. A
spoken briefing is a job, and jobs are retried after a crash, so delivery would be at least once
unless something stops it: on a retry, delivered sentences are skipped, and a sentence that was
playing when the speaker failed is skipped too and listed as uncertain. On a shop-floor speaker a
repeated instruction is worse than a missing one, and the journal says what may be missing.

## A voice agent as a cascade, with a latency budget per stage

[`cascade.py`](../src/aisyslab/voice/cascade.py): speech-to-text, a model, text-to-speech. The
answer streams, and each sentence goes to speech as soon as it is complete; the first sentence is
cut once more at its first clause of four words or more, because a short clause is spoken sooner.
Each stage has a budget (transcription 1.5 s, the model's first sentence 2.0 s, the first sound 4.0
s). When the model misses its budget, a filler fills the silence. The finished answer is journaled,
so a retried turn speaks it from the first sentence that cannot have been heard instead of asking
the model again.

Two live runs, one each, so these are what happened, not averages:

- The first: transcription 1.06 s, first sentence 1.72 s, first sound 4.67 s, over budget because
  the first sentence had 25 words and its speech took about three seconds. Asking again with the same
  turn spoke nothing twice: three sentences "delivered before", no transcription, no model call.
- The fix: the first-clause cut, and the filler synthesized before the first turn.
- The second: the model took 2.9 s to its first sentence, so the filler played at 3.7 s, then
  "Line 2 is down:". The filler still had to be synthesized in that run; it is now prepared ahead.
