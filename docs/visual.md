# Visual work instructions

Code: [`src/aisyslab/visual/`](../src/aisyslab/visual/). CLI: `aisyslab card HP-07` (offline, with a
line drawing and no checks) or with `--images openai --provider openai` (live).

## The card is drawn in code

A work instruction card at the machine carries torques, pressures and part numbers. Image models
misspell and invent numbers, so nothing on the card that can be read is generated:
[`card.py`](../src/aisyslab/visual/card.py) draws the header, the safety box, the numbered steps and
the footer from the procedure's fields with Pillow. The only generated part is the illustration,
and it may not contain anything that reads as text.

## Two checks before the card is used

[`check.py`](../src/aisyslab/visual/check.py) asks a vision model narrow questions and lets code
decide:

- **The illustration.** The model lists every mark that resembles letters or digits, says what the
  image shows and whether anything is unsafe. Code decides what counts as text: digits, capitals
  and symbols always ("G5", "50 BAR", "Ø16"), a short lower-case token too; a shape described in
  lower-case words ("two small circles") does not. An answer that does not parse is a rejection.
  A rejected image is generated once more with the reason in the prompt; if that one fails too,
  the card goes out without an illustration, which is always safe.
- **The finished card.** The model copies the numbered steps as printed. Code compares the reading
  with the procedure: the number of steps, and each step's critical value (a torque, a pressure, a
  part number) as a value of its own, so "195 Nm" is not a reading of "95 Nm". A card that reads
  back differently is not released.

The model never says that a card is fine. It reports what it sees, and the rules are in code.

## What the live runs showed

On 24 October 2026, `gpt-image-1-mini` at low quality drew the illustration and `gpt-5-mini` at low
effort checked it, for about a cent per card.

- **The first check was a yes-or-no question, and it was wrong.** Asked "any letters, digits, labels
  or logos?", the model answered no for an image with letter-like marks on the valve block ("6S", or
  "G5") and a "G" on a gauge. The card was released.
- **The fix: list, then decide in code.** Asked to list every such mark, the model found "6" and "C"
  on the same image, and bolt heads as dots; asked for the characters only, "G5", "00" and "6", and
  code rejected the image. One of those calls returned nothing that parsed, and code rejected that
  too.
- **A new image passed for the right reason.** The next illustration had no marks; the check listed
  none, the card read back all six steps with their values, and it was released.
- **The checker has to know the generator's conventions.** The generation prompt asks for one amber
  highlight on the part being worked on; the check read it as wear on the hose. The check prompt
  now says what the highlight means.
