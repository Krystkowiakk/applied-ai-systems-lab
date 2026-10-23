"""Speaking without paying twice and without saying anything twice.

``SpeechCache`` keeps synthesized audio under a key made of the synthesizer, the voice, the style
instructions and the text: the same sentence in the same voice is synthesized once, ever. A
different voice or a changed instruction is a different key, never a stale hit.

``DeliveryJournal`` records each utterance before it is played and after: ``delivering``, then
``delivered``. A briefing job that crashes and is retried (the runtime retries jobs, so delivery is
at least once unless something stops it) asks the journal first. Delivered is skipped; delivering
means the speaker may have said it, and it is skipped too: on a shop-floor speaker, a repeated
instruction is worse than a missing one, and the journal says what may be missing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from .speech import Audio, Synthesizer


class SpeechCache:
    def __init__(self, directory: Path, synthesizer: Synthesizer) -> None:
        self.directory = directory
        self.synthesizer = synthesizer
        self.hits = 0
        self.misses = 0
        directory.mkdir(parents=True, exist_ok=True)

    def key(self, text: str, *, voice: str, instructions: str = "") -> str:
        material = json.dumps([self.synthesizer.name, voice, instructions, text], ensure_ascii=False)
        return hashlib.sha256(material.encode()).hexdigest()[:32]

    def get(self, text: str, *, voice: str, instructions: str = "") -> Audio:
        key = self.key(text, voice=voice, instructions=instructions)
        for path in self.directory.glob(f"{key}.*"):
            self.hits += 1
            return Audio(path.read_bytes(), path.suffix[1:], key)
        audio = self.synthesizer.synthesize(text, voice=voice, instructions=instructions)
        (self.directory / f"{key}.{audio.format}").write_bytes(audio.data)
        self.misses += 1
        return Audio(audio.data, audio.format, key)


class DeliveryJournal:
    def __init__(self, path: Path | str = ":memory:") -> None:
        # Autocommit, so a state is on disk before playback starts; one lock, since the model's
        # answer is saved from the thread that streams it.
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS utterances (id TEXT PRIMARY KEY, text TEXT, state TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (turn TEXT PRIMARY KEY, text TEXT)")

    def state(self, utterance_id: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT state FROM utterances WHERE id = ?", (utterance_id,)).fetchone()
        return row[0] if row else None

    def begin(self, utterance_id: str, text: str) -> bool:
        """Claim an utterance for playing; False when it was ever claimed before."""
        with self.lock:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO utterances (id, text, state) VALUES (?, ?, 'delivering')", (utterance_id, text)
            )
        return cursor.rowcount == 1

    def delivered(self, utterance_id: str) -> None:
        with self.lock:
            self.db.execute("UPDATE utterances SET state = 'delivered' WHERE id = ?", (utterance_id,))

    def uncertain(self) -> list[str]:
        """Utterances that may or may not have reached the listener."""
        with self.lock:
            rows = self.db.execute("SELECT text FROM utterances WHERE state = 'delivering' ORDER BY rowid").fetchall()
        return [r[0] for r in rows]

    def save_answer(self, turn: str, text: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO answers (turn, text) VALUES (?, ?)", (turn, text))

    def answer(self, turn: str) -> str | None:
        """A turn's whole answer, once the model has finished it: a retry speaks this, not a new one."""
        with self.lock:
            row = self.db.execute("SELECT text FROM answers WHERE turn = ?", (turn,)).fetchone()
        return row[0] if row else None


def utterance_id(scope: str, text: str) -> str:
    """The same sentence in the same briefing or turn is the same utterance, on every retry."""
    return f"{scope}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"


class Speaker:
    def __init__(
        self,
        cache: SpeechCache,
        journal: DeliveryJournal,
        play: Callable[[Audio], None],
        *,
        voice: str = "alloy",
        instructions: str = "",
    ) -> None:
        self.cache = cache
        self.journal = journal
        self.play = play
        self.voice = voice
        self.instructions = instructions

    def prepare(self, text: str) -> None:
        """Synthesize ahead of time (a filler, a fixed phrase), so saying it later costs no wait."""
        self.cache.get(text, voice=self.voice, instructions=self.instructions)

    def say(self, scope: str, text: str) -> str:
        """Play one sentence at most once. Returns what happened: played, or why it was skipped."""
        uid = utterance_id(scope, text)
        state = self.journal.state(uid)
        if state == "delivered":
            return "skipped: delivered before"
        if state == "delivering":
            return "skipped: may have been delivered"
        audio = self.cache.get(text, voice=self.voice, instructions=self.instructions)  # before the claim
        if not self.journal.begin(uid, text):
            return "skipped: claimed by another speaker"
        self.play(audio)
        self.journal.delivered(uid)
        return "played"
