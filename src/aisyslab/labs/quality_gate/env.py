"""A quality gate at the end of the shift: learn the lab's rule from inspected batches.

The quality lab has judged 240 earlier batches OK or NOK; its rule was never written down.
Tonight's batches need a decision. The rule hides in the numbers (a torque window and a minimum
cure temperature), which is code's territory: a decision tree finds the thresholds exactly.

Traps for a solver that lets a model read the numbers, or trains on everything:

- ``rework_ticket`` is filled in only after the lab rejects a batch. It predicts the history
  perfectly and tonight not at all (target leakage);
- two of tonight's batches lost their cure-temperature reading and must be held;
- the torque window has two edges, and failures above the upper edge are rare.

Tonight's batches are clear of the thresholds, so a rule learned from the history decides them
the way the lab would.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict
from . import tree as cart

HISTORY_SIZE = 240
TONIGHT_SIZE = 24
NUMERIC = ("batch_seq", "torque_nm", "cure_temp_c", "cure_time_s", "humidity_pct")
TEXT = ("line", "shift", "rework_ticket")
FEATURES = NUMERIC + TEXT
API_ROW_LIMIT = 500
TORQUE_MARGIN, CURE_MARGIN = 0.15, 1.5


class QualityGateEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        r = self.rng
        self.torque_low = round(r.uniform(4.0, 4.4), 2)
        self.torque_high = round(r.uniform(5.6, 6.0), 2)
        self.cure_min = round(r.uniform(150.0, 156.0), 1)
        self._ids = iter(r.sample(range(260000, 269999), HISTORY_SIZE + TONIGHT_SIZE))
        self.history = self._history()
        self.tonight = self._tonight()
        self._by_id = {b["batch_id"]: b for b in self.tonight}
        self.codes = {
            col: {value: float(k) for k, value in enumerate(sorted({str(b[col]) for b in self.history}))}
            for col in TEXT
        }
        self.expected = {
            b["batch_id"] for b in self.tonight if b["cure_temp_c"] is not None and self.verdict(b) == "OK"
        }
        self.models: dict[str, cart.Node] = {}
        self.instruction = (
            f"Tonight's shift produced {TONIGHT_SIZE} batches. Decide which of them pass the quality gate. "
            f"The quality lab has judged {HISTORY_SIZE} earlier batches OK or NOK; its rule was never written "
            "down. A batch with a missing measurement is held. Answer with the IDs of the batches that pass."
        )

    # ----- data ----------------------------------------------------------------------------
    def verdict(self, batch: dict[str, Any]) -> str:
        t, c = batch["torque_nm"], batch["cure_temp_c"]
        return "NOK" if t < self.torque_low or t > self.torque_high or c < self.cure_min else "OK"

    def _batch(self, torque: float | None = None, cure: float | None = None) -> dict[str, Any]:
        r = self.rng
        return {
            "batch_id": f"B-{next(self._ids)}",
            "batch_seq": 0,
            "line": r.choice(["L1", "L2"]),
            "shift": r.choice(["A", "B", "C"]),
            "torque_nm": round(r.gauss(5.0, 0.55), 2) if torque is None else torque,
            "cure_temp_c": round(r.gauss(163.0, 6.0), 1) if cure is None else cure,
            "cure_time_s": r.randint(280, 340),
            "humidity_pct": round(r.uniform(30.0, 60.0), 1),
            "rework_ticket": "",
        }

    def _history(self) -> list[dict[str, Any]]:
        r = self.rng
        # Batches close to each threshold, on both sides, as a real history would have.
        edges = [
            *(self._batch(torque=round(self.torque_low + d, 2), cure=165.0) for d in (-0.03, -0.02, 0.02, 0.03)),
            *(self._batch(torque=round(self.torque_high + d, 2), cure=165.0) for d in (-0.03, -0.02, 0.02, 0.03)),
            *(self._batch(torque=5.0, cure=round(self.cure_min + d, 1)) for d in (-0.3, -0.2, 0.2, 0.3)),
        ]
        batches = edges + [self._batch() for _ in range(HISTORY_SIZE - len(edges))]
        r.shuffle(batches)
        tickets = iter(r.sample(range(1000, 9999), HISTORY_SIZE))
        for seq, batch in enumerate(batches, start=1):
            batch["batch_seq"] = seq
            batch["lab_verdict"] = self.verdict(batch)
            if batch["lab_verdict"] == "NOK":
                batch["rework_ticket"] = f"RW-{next(tickets)}"  # written after the lab's decision
        return batches

    def _clear(self, torque: float, cure: float) -> bool:
        return (
            abs(torque - self.torque_low) > TORQUE_MARGIN
            and abs(torque - self.torque_high) > TORQUE_MARGIN
            and abs(cure - self.cure_min) > CURE_MARGIN
        )

    def _sample(self, draw: Callable[[], float], keep: Callable[[float], bool]) -> float:
        while True:
            value = draw()
            if keep(value):
                return value

    def _tonight(self) -> list[dict[str, Any]]:
        r = self.rng
        torque = lambda: round(r.gauss(5.0, 0.55), 2)  # noqa: E731
        cure = lambda: round(r.gauss(163.0, 6.0), 1)  # noqa: E731
        good_torque = lambda: self._sample(  # noqa: E731
            torque, lambda t: self.torque_low + TORQUE_MARGIN < t < self.torque_high - TORQUE_MARGIN
        )
        good_cure = lambda: self._sample(cure, lambda c: c > self.cure_min + CURE_MARGIN)  # noqa: E731
        batches = [
            self._batch(torque=round(self.torque_low - r.uniform(0.2, 0.6), 2), cure=good_cure()),
            self._batch(torque=round(self.torque_high + r.uniform(0.2, 0.6), 2), cure=good_cure()),
            self._batch(torque=good_torque(), cure=round(self.cure_min - r.uniform(2.0, 6.0), 1)),
        ]
        while len(batches) < TONIGHT_SIZE:
            t, c = torque(), cure()
            if self._clear(t, c):
                batches.append(self._batch(torque=t, cure=c))
        for i in r.sample(range(3, TONIGHT_SIZE), 2):
            batches[i]["cure_temp_c"] = None  # the sensor dropped out
        r.shuffle(batches)
        for seq, batch in enumerate(batches, start=HISTORY_SIZE + 1):
            batch["batch_seq"] = seq
        return batches

    def _dataset(self, name: str) -> list[dict[str, Any]]:
        if name == "history":
            return self.history
        if name == "tonight":
            return self.tonight
        raise ToolError(f"No dataset '{name}'.", hint="Datasets: history, tonight.")

    def _encode(self, batch: dict[str, Any]) -> dict[str, float | None]:
        row: dict[str, float | None] = {col: batch[col] for col in NUMERIC}
        for col in TEXT:
            row[col] = self.codes[col].get(str(batch[col]), -1.0)
        return row

    # ----- the API --------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        self._api("describe")
        by = {lab: [b for b in self.history if b["lab_verdict"] == lab] for lab in ("OK", "NOK")}
        return {
            "history": {
                "rows": len(self.history),
                "verdicts": cart.label_counts([b["lab_verdict"] for b in self.history]),
                "columns": [*NUMERIC, *TEXT, "lab_verdict"],
            },
            "tonight": {
                "rows": len(self.tonight),
                "columns": [*NUMERIC, *TEXT],
                "missing_values": {col: n for col in NUMERIC if (n := sum(b[col] is None for b in self.tonight))},
            },
            "numeric_by_verdict": {
                col: {lab: _summary([b[col] for b in rows]) for lab, rows in by.items()} for col in NUMERIC
            },
            "text_by_verdict": {
                col: {
                    "distinct_values": len({b[col] for b in self.history}),
                    "filled_share": {
                        lab: round(sum(bool(b[col]) for b in rows) / len(rows), 2) for lab, rows in by.items()
                    },
                }
                for col in TEXT
            },
        }

    def rows(self, dataset: str, offset: int = 0, limit: int = 25) -> dict[str, Any]:
        self._api("rows")
        data = self._dataset(dataset)
        offset, limit = max(0, offset), max(1, min(limit, API_ROW_LIMIT))
        out: dict[str, Any] = {
            "dataset": dataset,
            "total": len(data),
            "offset": offset,
            "rows": [dict(b) for b in data[offset : offset + limit]],
        }
        if offset + limit < len(data):
            out["next_offset"] = offset + limit
        return out

    def train(self, features: list[str], max_depth: int = 3) -> dict[str, Any]:
        self._api("train")
        unknown = [f for f in features if f not in FEATURES]
        if unknown or not features:
            raise ToolError(
                f"Unknown feature(s): {', '.join(unknown) or 'none given'}.", hint=f"Features: {', '.join(FEATURES)}."
            )
        if not 1 <= max_depth <= 5:
            raise ToolError("max_depth must be between 1 and 5.")
        rows = [self._encode(b) for b in self.history]
        labels = [b["lab_verdict"] for b in self.history]
        node = cart.fit(rows, labels, features, max_depth)
        model_id = f"M{len(self.models) + 1}"
        self.models[model_id] = node
        hits = sum(cart.predict(node, row) == label for row, label in zip(rows, labels, strict=True))
        return {
            "model_id": model_id,
            "rules": cart.rules(node),
            "features_used": cart.used_features(node),
            "train_accuracy": round(hits / len(rows), 3),
            "cv_accuracy": round(cart.cross_validate(rows, labels, features, max_depth), 3),
        }

    def predict(self, model_id: str) -> dict[str, Any]:
        self._api("predict")
        node = self.models.get(model_id)
        if node is None:
            raise ToolError(f"No model '{model_id}'.", hint="Fit one with train_tree first.")
        predictions = []
        for batch in self.tonight:
            try:
                predictions.append(
                    {"batch_id": batch["batch_id"], "prediction": cart.predict(node, self._encode(batch))}
                )
            except cart.MissingValue as exc:
                predictions.append(
                    {"batch_id": batch["batch_id"], "prediction": "unknown", "reason": f"missing {exc.feature}"}
                )
        return {"model_id": model_id, "predictions": predictions}

    def submit(self, passing_batch_ids: list[str]) -> Verdict:
        self._api("submit")
        given = {b.strip().upper() for b in passing_batch_ids if b.strip()}
        unknown = sorted(given - set(self._by_id))
        known = given & set(self._by_id)
        missing_passed = [b for b in known if self._by_id[b]["cure_temp_c"] is None]
        bad_passed = [b for b in known if b not in self.expected and self._by_id[b]["cure_temp_c"] is not None]
        good_held = self.expected - given
        problems: list[str] = []
        if bad_passed:
            problems.append(_count(len(bad_passed), "passed batch fails", "passed batches fail") + " the lab's rule")
        if missing_passed:
            problems.append(
                _count(len(missing_passed), "passed batch has", "passed batches have")
                + " a missing measurement and must be held"
            )
        if good_held:
            problems.append(_count(len(good_held), "held batch would", "held batches would") + " pass")
        if unknown:
            problems.append(_count(len(unknown), "ID is", "IDs are") + " not among tonight's batches")
        hits = len(given & self.expected)
        score = 2 * hits / (len(given) + len(self.expected)) if given or self.expected else 1.0
        accepted = not problems
        return self._record(
            Verdict(
                accepted,
                "OK" if accepted else "REJECTED",
                "Gate decisions match the lab." if accepted else "; ".join(problems) + ".",
                {
                    "score": round(score, 3),
                    "false_passes": len(bad_passed) + len(missing_passed),
                    "false_holds": len(good_held),
                    "missing_passed": len(missing_passed),
                },
            )
        )


def _summary(values: list[float | None]) -> dict[str, float]:
    present = [float(v) for v in values if v is not None]
    return {"min": round(min(present), 2), "mean": round(fmean(present), 2), "max": round(max(present), 2)}


def _count(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"
