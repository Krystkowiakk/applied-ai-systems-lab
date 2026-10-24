"""``aisyslab voice``: evaluate the voice-note pipeline, or ask the shift-lead assistant by voice."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from ..core.messages import Message, Usage
from ..core.model import ChatModel
from ..providers import PROVIDERS, make_model
from .cascade import VoiceCascade
from .delivery import DeliveryJournal, Speaker, SpeechCache
from .notes import NOTES, STYLE, VOCABULARY, VoiceNote, evaluate
from .speech import Audio, OpenAISynthesizer, OpenAITranscriber, RecordedTranscriber

RECORDED = Path(__file__).parent / "recorded" / "voice-notes.json"
ASK_SYSTEM = """You brief a shift lead by voice, from the night shift's voice notes below. Answer in two
to four short sentences that sound natural when spoken: no lists, no markdown, the most urgent first,
machine and part numbers said once."""


def add_parser(sub: Any) -> None:
    voice = sub.add_parser("voice", help="Voice on the shop floor: evaluate voice notes, or ask by voice.")
    voice.add_argument("action", choices=["notes", "ask"])
    voice.add_argument(
        "--speech", choices=["recorded", "openai"], default="recorded", help="Recorded transcripts, or live speech."
    )
    voice.add_argument("--stt-model", default="gpt-4o-transcribe")
    voice.add_argument("--tts-model", default="gpt-4o-mini-tts")
    voice.add_argument("--provider", choices=PROVIDERS, default=None, help="A model to read notes or answer.")
    voice.add_argument("--model", default=None)
    voice.add_argument("--effort", default=None)
    voice.add_argument("--record", type=Path, default=None, help="Save the live transcripts as a recording here.")
    voice.add_argument("--question", default="What do I need to know about line two before the morning meeting?")
    voice.add_argument(
        "--turn", default=None, help="Turn id (default: one per question); asking again with the same id resumes it."
    )
    voice.add_argument("--out", type=Path, default=Path("runs/voice"))


def run(args: argparse.Namespace) -> int:
    if (args.model or args.effort) and not args.provider:
        print("error: --model and --effort need --provider", file=sys.stderr)
        return 2
    model = make_model(args.provider, args.model, effort=args.effort) if args.provider else None
    if args.turn is None:  # a new question is a new turn; the same question asked again resumes it
        args.turn = "ask-" + hashlib.sha256(args.question.encode()).hexdigest()[:12]
    return _notes(args, model) if args.action == "notes" else _ask(args, model)


class _Metered:
    """Counts the tokens of the model calls made for the evaluation."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.usage = Usage()

    def __call__(self, system: str, user: str) -> str:
        turn = self.model.complete(system=system, messages=[Message(role="user", content=user)], tools=[])
        self.usage = self.usage + turn.usage
        return turn.text


def _notes(args: argparse.Namespace, model: ChatModel | None) -> int:
    if args.speech == "recorded":
        transcriber: Any = RecordedTranscriber(RECORDED)

        def audio(note: VoiceNote) -> Audio:
            return Audio(b"", "mp3", note.id)

    else:
        cache = SpeechCache(args.out / "speech-cache", OpenAISynthesizer(args.tts_model))
        transcriber = OpenAITranscriber(args.stt_model)

        def audio(note: VoiceNote) -> Audio:
            clip = cache.get(note.script, voice=note.voice, instructions=STYLE)
            return Audio(clip.data, clip.format, note.id)

    call = _Metered(model) if model else None
    started = time.perf_counter()
    result = evaluate(transcriber, audio, call)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "voice-notes.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.record and args.speech == "openai":
        transcripts = {r["note"]: r["transcripts"] for r in result["rows"]}
        recording = {"model": args.stt_model, "tts": args.tts_model, "recorded": date.today().isoformat()}
        args.record.write_text(json.dumps(recording | {"transcripts": transcripts}, indent=2) + "\n", encoding="utf-8")
    print(f"Transcriber: {transcriber.name}; {len(NOTES)} notes; {time.perf_counter() - started:.1f} s")
    print("| Pipeline | Machine | Part | Urgency | Ask when unsure | All right |")
    print("|---|---|---|---|---|---|")
    for name, s in result["summary"].items():
        n = s["notes"]
        print(
            f"| {name} | {s['machine']}/{n} | {s['part']}/{n} | {s['urgency']}/{n} | {s['ask']}/{n} | {s['all_right']}/{n} |"
        )
    if call:
        print(f"Model tokens: {call.usage.input_tokens} in, {call.usage.output_tokens} out")
    return 0


def _ask(args: argparse.Namespace, model: ChatModel | None) -> int:
    if args.speech != "openai" or model is None:
        print("error: voice ask needs --speech openai and --provider", flush=True)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    cache = SpeechCache(args.out / "speech-cache", OpenAISynthesizer(args.tts_model))
    played: list[Path] = []

    def play(audio: Audio) -> None:  # stands in for the shop-floor speaker: each sentence as a file, in order
        path = args.out / f"{args.turn}-{len(played) + 1:02d}.{audio.format}"
        path.write_bytes(audio.data)
        played.append(path)

    speaker = Speaker(cache, DeliveryJournal(args.out / "journal.sqlite"), play, voice="sage")
    cascade = VoiceCascade(
        OpenAITranscriber(args.stt_model), model, speaker, system=ASK_SYSTEM, vocabulary=VOCABULARY.keywords()
    )
    cascade.warm()
    question = cache.get(args.question, voice="ash", instructions=STYLE)
    context = "\n".join(f"- {note.script}" for note in NOTES)
    report = cascade.turn(Audio(question.data, question.format, "question"), turn_id=args.turn, context=context)
    print(f"Heard: {report.question}")
    for sentence, outcome in report.spoken:
        print(f"  [{outcome}] {sentence}")
    print("Seconds: " + ", ".join(f"{k} {v:.2f}" for k, v in report.seconds.items()))
    print(f"Over budget: {', '.join(report.over_budget) or 'none'}; filler: {report.filler}")
    print(f"Speech cache: {cache.hits} hits, {cache.misses} syntheses; audio in {args.out}")
    return 0
