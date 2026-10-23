# Graph RAG: when the relations are the knowledge

Code: [`src/aisyslab/graph/`](../src/aisyslab/graph/) (store and extraction). Lab:
[`parts-graph`](labs.md#parts-graph).

## Why a graph here

A supplier delays a part. Which lines does that hold up, and which customer orders? The answer runs
across five documents: the bill of materials puts the part in assemblies, the installation register
and the logbooks put assemblies on machines, the layout and the changeover notes put machines on
lines, the line plan names each line's products, and the order book lists the orders for each
product. No document names both the part and an order it holds up; a test checks this on every
seed. Similarity search finds the documents that share words with the question (the delay notice,
a price letter, the bill of materials), and none of them holds the answer. The chain has to be
walked, not retrieved.

## Tables by code, notes by a model, both checked

What code can parse, code parses: every table row becomes nodes and facts, with no model involved.
Each note goes to one model call with a fresh context, which writes facts in a line format:

```
MC-14 RUNS_ON L3 since 2026-10-14
MC-14 RUNS_ON L2 until 2026-10-14
P-1015 ALTERNATE_FOR P-1513 in A-24 until 2026-09-08
```

A parser checks every line: the grammar, that the relation exists, the node types at both ends, the
dates, that every ID appears in the note and that it exists in the plant. A rejected line goes back
to the model once, with a reason it can act on ("leave out the type names and write ...", "the graph
holds no relation DELAYED_UNTIL; it holds ..."). What still fails is reported and never stored. The
example lines in the prompt use IDs no document uses, so a copied example fails the source check.

## Time on the edges

Facts carry the dates between which they hold. The schema marks `RUNS_ON` as functional, since a
machine stands on one line at a time: a dated move ends the fact it replaces, so a changeover note
supersedes the layout table without anyone deleting a row. A move planned for next month is stored
with its date and does not count today. An approval and its later withdrawal merge into one fact
with both dates, in whichever order they arrive. Two undated facts that cannot both hold are
reported as a conflict instead of being settled silently.

## Traversal tools

`walk` follows a chain of relations from a node and returns every path, with the documents behind
each hop; a `<` in front of a relation follows it backwards (`<FOR_PRODUCT` goes from a product to
its orders). `edges` lists one node's facts as current, ended and planned, with their documents.
`correct_graph` adds or removes facts a document states, through the same parser. The store is
SQLite in memory, small enough to read in one sitting; in production this is a graph database or
Postgres, and the tools stay the same.

## What the baseline shows

The baseline builds the same graph and reads the notes with patterns: a machine that "runs on line"
somewhere, every assembly a logbook names, every assembly an alternate is "approved" in, each dated
by the note. That holds for a plain changeover and fails on everything else: a move planned for next
month (dated today), an assembly sent to stores and one a machine "does not use" (both named in a
logbook), and an approval withdrawn in other words. It is accepted on 1 seed in 20, the one whose
only trap is a plain move, with a mean score of 0.22.

## What the live runs showed

The first run (24 October 2026, `gpt-5-mini` at low effort, seed 0 with three traps) failed for
about three cents. The prompt had shown each relation as `<Part> ALTERNATE_FOR <Part> in <Assembly>`,
and the extractor wrote `Part P-1015 ALTERNATE_FOR Part P-1513 in Assembly A-22 ...`. The parser
rejected every line; the repair call got a generic "not in the format" and wrote the same lines
again. Every note was lost. The build report listed the rejections inside each note, the agent did
not look, answered from the tables alone, scored zero and stopped with a summary in text.

The fixes: example lines instead of a notation, with IDs no document uses; a parser that says how
to fix a line it can read; a warning at the top of the build report that rejected lines are facts
missing from the graph; and a prompt that says so and that only `submit_impact` counts. In the
second run every note was extracted on the first call. The agent tried three times to store the
delivery date as a fact (the graph has no relation for it, and the parser now says so in those
words), then listed one order as both at risk and saved; the verifier said so, and the second
submission was accepted, for about a cent and a half.
