"""Voice notes from the shop floor, filed as work requests: an evaluation of the speech pipeline.

Ten notes, written the way technicians talk at a machine: part numbers in pairs of digits ("P ten
forty") or one digit at a time, machines by number or by what they are ("the transfer press"), a
correction in mid-sentence, a "press on line one" when line one has two, a note with no part.
Each note has the request it should become: machine, part, urgency, and whether a person has to
be asked first.

Four pipelines are compared on the same audio, each adding one step:

- ``plain``: speech-to-text without the vocabulary, then patterns over the IDs it wrote;
- ``grounding``: the same transcript grounded in the registry (spoken numbers, names, teen/ty);
- ``vocabulary``: speech-to-text told the plant's terms, then grounding;
- ``vocabulary+model``: the same, then a model reads the note and fills the request.

Audio comes from a text-to-speech model (cached), so the notes are clean speech: a floor with a
press running is harder, and these numbers are a floor for the error rate, not a ceiling.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .speech import Audio, Transcriber
from .vocabulary import Grounded, Term, Vocabulary

REGISTRY = (
    Term("MC-11", "machine", "press 1", "L1"),
    Term("MC-12", "machine", "press 2", "L1"),
    Term("MC-13", "machine", "trim press", "L2"),
    Term("MC-14", "machine", "transfer press", "L2"),
    Term("MC-15", "machine", "press 5", "L3"),
    Term("MC-16", "machine", "robot cell", "L3"),
    Term("MC-17", "machine", "welding cell", "L4"),
    Term("MC-18", "machine", "assembly cell", "L4"),
    Term("P-1040", "part", "seal kit"),
    Term("P-1218", "part", "proportional valve"),
    Term("P-1314", "part", "filter element"),
    Term("P-1515", "part", "relief valve"),
    Term("P-1550", "part", "pressure sensor"),
    Term("P-1660", "part", "gear pump"),
    Term("P-1716", "part", "encoder"),
    Term("L1", "line"),
    Term("L2", "line"),
    Term("L3", "line"),
    Term("L4", "line"),
)
VOCABULARY = Vocabulary(REGISTRY)
URGENCIES = ("stop", "soon", "routine")
STYLE = "A maintenance technician leaving a quick voice note on a noisy shop floor: informal, a little hurried."


@dataclass(frozen=True)
class VoiceNote:
    id: str
    voice: str
    script: str
    machine: str | None  # None when the note does not say which machine: ask
    part: str | None
    urgency: str
    ask: bool = False


NOTES = (
    VoiceNote(
        "note-01",
        "ash",
        "This is Marta at M C fourteen, the transfer press. The seal kit, P ten forty, is weeping oil. "
        "It will hold until Friday.",
        "MC-14",
        "P-1040",
        "soon",
    ),
    VoiceNote(
        "note-02",
        "coral",
        "Line two is down. The trim press, M C fifteen, sorry, M C thirteen, has a hydraulic pressure alarm. "
        "We need the relief valve, P fifteen fifteen, right now.",
        "MC-13",
        "P-1515",
        "stop",
    ),
    VoiceNote(
        "note-03",
        "echo",
        "Robot cell on line three keeps stopping on a position error. Probably the encoder, P seventeen "
        "sixteen. Not urgent, just log it.",
        "MC-16",
        "P-1716",
        "routine",
    ),
    VoiceNote(
        "note-04",
        "sage",
        "Press two on line one, the proportional valve P twelve eighteen is sticking. Cycle times are up, "
        "it should be fixed this week.",
        "MC-12",
        "P-1218",
        "soon",
    ),
    VoiceNote(
        "note-05",
        "verse",
        "The press on line one is leaking from the main cylinder. It needs fixing before the weekend.",
        None,
        None,
        "soon",
        ask=True,
    ),
    VoiceNote(
        "note-06",
        "ash",
        "M C seventeen, the welding cell. Filter element P thirteen fourteen is clogged again. The line will "
        "stop within the hour if we don't swap it.",
        "MC-17",
        "P-1314",
        "stop",
    ),
    VoiceNote(
        "note-07",
        "coral",
        "Assembly cell, M C eighteen. The pressure sensor P fifteen fifty reads zero. We're running on the "
        "backup, so it can wait for the day shift.",
        "MC-18",
        "P-1550",
        "soon",
    ),
    VoiceNote(
        "note-08",
        "echo",
        "Gear pump P sixteen sixty on M C eleven is noisy. Nothing urgent.",
        "MC-11",
        "P-1660",
        "routine",
    ),
    VoiceNote(
        "note-09",
        "sage",
        "This is Tomasz. The transfer press needs a look, something is knocking in the drive, but I "
        "couldn't see which part. It's still running.",
        "MC-14",
        None,
        "routine",
    ),
    VoiceNote(
        "note-10",
        "verse",
        "M C one four, the transfer press, line is stopped. Seal kit again, P one zero four zero.",
        "MC-14",
        "P-1040",
        "stop",
    ),
)

EXTRACT_SYSTEM = """You file a maintenance work request from a technician's voice note.

You get the transcript and what the plant's vocabulary found in it: IDs it is sure of, and what it
could not settle ("unsure", with candidates and the reason). Return JSON only:
{"machine": "MC-.." or null, "part": "P-...." or null, "urgency": "stop" | "soon" | "routine",
 "question": "what to ask the technician when the machine is not certain, else empty"}

- A correction ("sorry", "I mean", "no wait") replaces what was said before it.
- Use an ID the vocabulary found, or one of the candidates when the note's own words settle it
  (the part's name, the machine's line). Never make an ID up.
- When the note does not say which machine, set machine to null and write the question.
- urgency: stop if the line is down or will stop within hours; soon if it must be fixed within days
  or by the next shift; routine if it only needs to be logged."""


def code_request(grounded: Grounded) -> dict[str, Any]:
    """Patterns over the grounded note: the first machine and part found, urgency by keywords."""
    machines, parts = grounded.ids("machine"), grounded.ids("part")
    text = grounded.normalised.lower()
    if re.search(r"\b(down|stopped|right now|within the hour|will stop)\b", text):
        urgency = "stop"
    elif re.search(r"\b(this week|until|day shift|before|should be fixed|soon)\b", text):
        urgency = "soon"
    else:
        urgency = "routine"
    machine = machines[0] if machines else None
    return {"machine": machine, "part": parts[0] if parts else None, "urgency": urgency, "ask": machine is None}


def plain_request(transcript: str) -> dict[str, Any]:
    """Speech-to-text alone: IDs only where the transcript already wrote them as IDs."""
    machine = re.search(r"\bMC-?(\d{2})\b", transcript)
    part = re.search(r"\bP-?(\d{4})\b", transcript)
    grounded = Grounded(transcript, transcript)
    request = code_request(grounded)
    request.update(
        machine=f"MC-{machine[1]}" if machine else None, part=f"P-{part[1]}" if part else None, ask=machine is None
    )
    return request


def model_request(call: Callable[[str, str], str], grounded: Grounded) -> dict[str, Any]:
    text = call(EXTRACT_SYSTEM, json.dumps(grounded.as_dict(), ensure_ascii=False))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    kinds = {t.id: t.kind for t in REGISTRY}
    candidates = {c for u in grounded.unsure for c in u.get("candidates", [])}
    # A machine is certain only when grounding found it; one picked from an unsure list is a guess.
    machine = data.get("machine") if data.get("machine") in grounded.ids("machine") else None
    # A part may come from the candidates: the note's words (its name) can settle a number.
    named = data.get("part")
    part = named if isinstance(named, str) and kinds.get(named) == "part" else None
    part = part if part in grounded.ids("part") or part in candidates else None
    urgency = data.get("urgency") if data.get("urgency") in URGENCIES else "soon"
    question = str(data.get("question") or "") if machine is None else ""
    # Whether to ask is decided in code: a request without a certain machine cannot be filed.
    return {"machine": machine, "part": part, "urgency": urgency, "ask": machine is None, "question": question}


def grade(note: VoiceNote, request: dict[str, Any]) -> dict[str, bool]:
    return {
        "machine": request["machine"] == note.machine,
        "part": request["part"] == note.part,
        "urgency": request["urgency"] == note.urgency,
        "ask": request["ask"] == note.ask,
    }


def evaluate(
    transcriber: Transcriber,
    audio: Callable[[VoiceNote], Audio],
    model_call: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Run every pipeline on every note; returns per-note results and a summary per pipeline."""
    rows: list[dict[str, Any]] = []
    for note in NOTES:
        clip = audio(note)
        plain = transcriber.transcribe(clip)
        heard = transcriber.transcribe(clip, vocabulary=VOCABULARY.keywords())
        grounded = VOCABULARY.ground(heard)
        requests = {
            "plain": plain_request(plain),
            "grounding": code_request(VOCABULARY.ground(plain)),
            "vocabulary": code_request(grounded),
        }
        if model_call is not None:
            requests["vocabulary+model"] = model_request(model_call, grounded)
        rows.append(
            {
                "note": note.id,
                "transcripts": {"plain": plain, "vocabulary": heard},
                "grounded": grounded.as_dict(),
                "requests": requests,
                "grades": {name: grade(note, r) for name, r in requests.items()},
            }
        )
    summary: dict[str, dict[str, int]] = {}
    for name in rows[0]["requests"]:
        fields = {f: sum(r["grades"][name][f] for r in rows) for f in ("machine", "part", "urgency", "ask")}
        summary[name] = fields | {"all_right": sum(all(r["grades"][name].values()) for r in rows), "notes": len(rows)}
    return {"rows": rows, "summary": summary}
