# Research: search, read, note, report

Code: [`src/aisyslab/research/`](../src/aisyslab/research/). Lab:
[`part-research`](labs.md#part-research). CLI: `aisyslab search "query"`.

## One search interface, two backends

A research loop calls `SearchAPI.search(query, k)` and gets URLs, titles and snippets. `OfflineSearch`
answers from a fixed set of pages, so a lab can be graded; `OpenAISearch` asks a search model on the
real web and turns the pages it cites into results. On 24 October 2026 one real search ("ISO 4414
pneumatic safety requirements") returned eleven cited pages in seven seconds, for about four cents.

## A crawler that behaves

`Crawler.fetch` reads robots.txt first and refuses a disallowed path; keeps a minimum interval per host
by waiting, instead of being told to slow down; leaves transient errors to the caller's retries;
stops at a page budget; and counts every URL once, in one canonical form.

## A deep research loop

The agent searches, fetches the pages that matter and notes each fact with the page it came from. A
note can only cite a page that was fetched. Then it names what is still missing (a current datasheet,
a current lead time, whether an alternative fits) and searches again.

## A long report, section by section

The report is not one long generation. The agent gives an outline; each section is written by a
separate call from the numbered notes, with every sentence citing its source; code assembles the
sections, lists the sources and refuses the report if a citation points to no source or to a page
that was never fetched.

## What the live runs showed

On seed 0, `gpt-5-mini` at low effort found the right answer on its first try: the maker's successor,
6 weeks, 10 bar from the current datasheet revision. The verifier rejected it, because the answer
named the part inside a sentence ("Valvetec VT-210 (24 V DC, G1/4), recommended replacement") and the
verifier compared whole strings. Its feedback, "not the fastest compatible replacement", was wrong, and
the agent spent its remaining steps searching for parts in stock. The verifier now reads part numbers
out of the text and says so when there is none or more than one. Feedback is the contract for the next
attempt, so wrong feedback misdirects a correct agent. In the second run the agent was accepted on its
fifth submission, after the feedback ("an incompatible part someone recommends was not rejected") sent
it to the forum thread it had not read.
