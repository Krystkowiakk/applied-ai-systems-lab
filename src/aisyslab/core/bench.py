"""Aggregate run reports into one comparison table per lab and solver."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean

from .runner import RunReport


@dataclass
class Row:
    lab: str
    solver: str
    model: str
    runs: int
    accepted: int
    score: float | None
    submissions: float
    api_calls: float
    tool_calls: float
    model_calls: float
    input_tokens: float
    output_tokens: float
    seconds: float
    statuses: dict[str, int]

    def cost(self, price_in: float | None, price_out: float | None) -> float | None:
        """Mean USD per run, given prices per million tokens."""
        if price_in is None or price_out is None:
            return None
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


def summarize(reports: Iterable[RunReport]) -> list[Row]:
    groups: dict[tuple[str, str, str], list[RunReport]] = defaultdict(list)
    for r in reports:
        groups[(r.lab, r.solver, r.model or "-")].append(r)
    rows = []
    for (lab, solver, model), group in groups.items():
        statuses: dict[str, int] = defaultdict(int)
        for r in group:
            statuses[r.status] += 1
        rows.append(
            Row(
                lab=lab,
                solver=solver,
                model=model,
                runs=len(group),
                accepted=sum(r.accepted for r in group),
                score=mean(r.score for r in group if r.score is not None)
                if any(r.score is not None for r in group)
                else None,
                submissions=mean(r.submissions for r in group),
                api_calls=mean(r.api_calls for r in group),
                tool_calls=mean(r.tool_calls for r in group),
                model_calls=mean(r.model_calls for r in group),
                input_tokens=mean(r.input_tokens for r in group),
                output_tokens=mean(r.output_tokens for r in group),
                seconds=mean(r.seconds for r in group),
                statuses=dict(statuses),
            )
        )
    return rows


def to_markdown(rows: Sequence[Row], *, price_in: float | None = None, price_out: float | None = None) -> str:
    with_cost = price_in is not None and price_out is not None
    header = [
        "Lab",
        "Solver",
        "Model",
        "Accepted",
        "Score",
        "Submissions",
        "API calls",
        "Tool calls",
        "Model calls",
        "Tokens in",
        "Tokens out",
        "Seconds",
    ]
    if with_cost:
        header.append("USD / run")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [
            row.lab,
            row.solver,
            row.model,
            f"{row.accepted}/{row.runs}",
            f"{row.score:.2f}" if row.score is not None else "-",
            f"{row.submissions:.1f}",
            f"{row.api_calls:.1f}",
            f"{row.tool_calls:.1f}",
            f"{row.model_calls:.1f}",
            f"{row.input_tokens:,.0f}",
            f"{row.output_tokens:,.0f}",
            f"{row.seconds:.2f}",
        ]
        if with_cost:
            cost = row.cost(price_in, price_out)
            cells.append(f"{cost:.4f}" if cost is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
