"""``aisyslab eval``: score the triage prompt, optimise it, or calibrate the judge."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from ..core.model import ChatModel
from ..providers import make_model
from .graders import DETERMINISTIC, judged_reason
from .judge import KeywordJudge, LLMJudge, calibrate
from .optimize import optimize_prompt
from .runner import run_eval, to_markdown
from .triage import BASE_PROMPT, RuleFollowerModel, RuleProposerModel, generate_tickets

OFFLINE = "offline stand-ins (rule follower, rule proposer, keyword judge), not language models"


def _model(args: argparse.Namespace, model_id: str | None) -> ChatModel:
    return make_model(args.provider, model_id, base_url=args.base_url, effort=args.effort)


def run(args: argparse.Namespace) -> int:
    offline = args.provider is None
    if offline:
        print(f"[{OFFLINE}]")
    tickets = generate_tickets(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.what == "judge":
        judge = LLMJudge(KeywordJudge() if offline else _model(args, args.model))
        calibration = calibrate(judge)
        print(json.dumps(calibration, indent=2, ensure_ascii=False))
        (args.out / f"judge-{stamp}.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        return 0

    if args.what == "triage":
        ids = [m.strip() for m in (args.models or args.model or "").split(",") if m.strip()] or [None]
        models = [RuleFollowerModel()] if offline else [_model(args, m) for m in ids]
        graders = list(DETERMINISTIC)
        if args.judge:
            graders.append(judged_reason(LLMJudge(KeywordJudge() if offline else _model(args, args.model))))
        reports = [run_eval(m, BASE_PROMPT, tickets, graders) for m in models]
        table = to_markdown(reports)
        print(table)
        (args.out / f"triage-{stamp}.md").write_text(table + "\n", encoding="utf-8")
        (args.out / f"triage-{stamp}.json").write_text(
            json.dumps([r.summary() | {"confusion": r.confusion()} for r in reports], indent=2), encoding="utf-8"
        )
        return 0

    executor = RuleFollowerModel() if offline else _model(args, args.model)
    optimizer = RuleProposerModel() if offline else _model(args, args.optimizer_model or args.model)
    result = optimize_prompt(executor, optimizer, BASE_PROMPT, tickets, rounds=args.rounds)
    for r in result.rounds:
        print(f"round {r.number}: train {r.train:.2f} {'kept' if r.kept else 'discarded'} {r.note}")
    for split, scores in result.final.items():
        print(f"{split:<8} start {scores['start']:.2f} -> best {scores['best']:.2f}")
    (args.out / f"optimize-{stamp}.json").write_text(
        json.dumps(
            {"rounds": [asdict(r) for r in result.rounds], "final": result.final, "best_prompt": result.best_prompt},
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0
