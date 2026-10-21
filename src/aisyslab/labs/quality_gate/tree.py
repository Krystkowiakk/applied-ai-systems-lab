"""A small decision tree (CART, Gini impurity) in plain Python.

This is what code should do with labelled numbers: find the thresholds exactly, the same way
every time, in milliseconds. The baseline uses it directly; the agent reaches it through the
``train_tree`` tool and decides which columns go in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Row = Mapping[str, float | None]


class MissingValue(Exception):
    """A row lacks a value the tree needs to decide."""

    def __init__(self, feature: str) -> None:
        super().__init__(feature)
        self.feature = feature


@dataclass
class Node:
    label: str
    size: int
    feature: str | None = None
    threshold: float | None = None
    below: Node | None = None  # value < threshold
    above: Node | None = None  # value >= threshold


def _gini(ok: int, nok: int) -> float:
    n = ok + nok
    return 0.0 if n == 0 else 1.0 - (ok / n) ** 2 - (nok / n) ** 2


def fit(rows: Sequence[Row], labels: Sequence[str], features: Sequence[str], max_depth: int = 3) -> Node:
    """Grow a tree on labels "OK"/"NOK"; ties at a leaf go to "NOK" (hold when unsure)."""

    def grow(idx: list[int], depth: int) -> Node:
        ok = sum(labels[i] == "OK" for i in idx)
        nok = len(idx) - ok
        node = Node("OK" if ok > nok else "NOK", len(idx))
        if depth >= max_depth or ok == 0 or nok == 0:
            return node
        parent = _gini(ok, nok)
        best: tuple[float, str, float, int] | None = None
        for feature in features:
            order = sorted(idx, key=lambda i: _value(rows[i], feature))
            left_ok = left_nok = 0
            for k in range(len(order) - 1):
                if labels[order[k]] == "OK":
                    left_ok += 1
                else:
                    left_nok += 1
                a, b = _value(rows[order[k]], feature), _value(rows[order[k + 1]], feature)
                if a == b:
                    continue
                n_left = k + 1
                n_right = len(order) - n_left
                impurity = (n_left * _gini(left_ok, left_nok) + n_right * _gini(ok - left_ok, nok - left_nok)) / len(
                    order
                )
                gain = parent - impurity
                if best is None or gain > best[0] + 1e-12:
                    best = (gain, feature, (a + b) / 2, k + 1)
        if best is None or best[0] <= 1e-12:
            return node
        _gain, feature, threshold, _cut = best
        node.feature, node.threshold = feature, threshold
        node.below = grow([i for i in idx if _value(rows[i], feature) < threshold], depth + 1)
        node.above = grow([i for i in idx if _value(rows[i], feature) >= threshold], depth + 1)
        return node

    return grow(list(range(len(rows))), 0)


def predict(node: Node, row: Row) -> str:
    while node.feature is not None and node.threshold is not None and node.below and node.above:
        node = node.below if _value(row, node.feature) < node.threshold else node.above
    return node.label


def rules(node: Node, path: tuple[str, ...] = ()) -> list[str]:
    """Each leaf as a readable rule: 'torque_nm < 4.21 and cure_temp_c >= 152.6 -> OK (180)'."""
    if node.feature is None or node.threshold is None or node.below is None or node.above is None:
        condition = " and ".join(path) or "always"
        return [f"{condition} -> {node.label} ({node.size})"]
    t = f"{node.threshold:.2f}"
    return rules(node.below, (*path, f"{node.feature} < {t}")) + rules(node.above, (*path, f"{node.feature} >= {t}"))


def used_features(node: Node) -> list[str]:
    if node.feature is None or node.below is None or node.above is None:
        return []
    found = [node.feature, *used_features(node.below), *used_features(node.above)]
    return list(dict.fromkeys(found))


def cross_validate(
    rows: Sequence[Row], labels: Sequence[str], features: Sequence[str], max_depth: int, folds: int = 5
) -> float:
    """Accuracy over k folds split by position; deterministic, no shuffling."""
    hits = 0
    for fold in range(folds):
        train = [i for i in range(len(rows)) if i % folds != fold]
        test = [i for i in range(len(rows)) if i % folds == fold]
        tree = fit([rows[i] for i in train], [labels[i] for i in train], features, max_depth)
        hits += sum(predict(tree, rows[i]) == labels[i] for i in test)
    return hits / len(rows)


def label_counts(labels: Sequence[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {"OK": counts["OK"], "NOK": counts["NOK"]}


def _value(row: Row, feature: str) -> float:
    value = row.get(feature)
    if value is None:
        raise MissingValue(feature)
    return float(value)
