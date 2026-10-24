"""``aisyslab eval``: score the triage prompt, optimise it, or calibrate the judge."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime

from ..core.model import ChatModel, ModelError
from ..providers import make_model
from .graders import DETERMINISTIC, judged_reason
from .judge import KeywordJudge, LLMJudge, calibrate
from .optimize import optimize_prompt
from .runner import EvalReport, compare, run_eval, to_markdown
from .triage import BASE_PROMPT, RuleFollowerModel, RuleProposerModel, Ticket, generate_tickets

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

    if args.what in ("batch", "interview", "tune"):
        return _more(args, tickets, offline, stamp)

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
        by_id = {t.id: t for t in tickets}
        details = [
            r.summary()
            | {
                "confusion": r.confusion(),
                # every item, so disagreements can be read, not just counted
                "items": [
                    {"ticket": by_id[i.ticket_id].text, "priority": by_id[i.ticket_id].priority} | asdict(i)
                    for i in r.items
                ],
            }
            for r in reports
        ]
        (args.out / f"triage-{stamp}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
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


def _effort(args: argparse.Namespace, model_id: str) -> str | None:
    """Reasoning effort only for models that reason; the others reject the parameter."""
    return args.effort if model_id.startswith(("gpt-5", "o")) else None


def _more(args: argparse.Namespace, tickets: list[Ticket], offline: bool, stamp: str) -> int:
    from .batch import OpenAIJsonModel, RuleFollowerBatch, classify_batches
    from .finetune import (
        PRICES,
        TRAINING_PRICE,
        TUNED_SYSTEM,
        NearestTicketModel,
        OpenAITuning,
        SimulatedTuning,
        per_thousand,
        run_job,
        validate,
    )
    from .interview import NotesOwner, RuleDrafter, ScriptedInterviewer, draft_prompt, interview

    chosen = [t for t in tickets if t.split in args.split.split(",")]
    model_id = args.model or "gpt-5-mini"
    executor = RuleFollowerModel() if offline else make_model(args.provider, model_id, effort=_effort(args, model_id))
    saved: dict[str, object] = {}

    if args.what == "batch":
        if not offline and args.provider != "openai":
            print("error: a provider-enforced JSON schema is implemented for OpenAI here")
            return 2
        single = run_eval(executor, BASE_PROMPT, chosen)
        batch_model = (
            RuleFollowerBatch(drop=1) if offline else OpenAIJsonModel(model_id, effort=_effort(args, model_id))
        )
        batched = classify_batches(batch_model, BASE_PROMPT, chosen, size=args.batch_size)
        print(to_markdown([single, batched.report]))
        print(
            f"Requests: {len(chosen)} one by one, {batched.calls} in batches "
            f"({batched.missing_first} tickets left out and asked again, {batched.repeated} repeats dropped)"
        )
        for grader in ("code", "priority"):
            diff = compare(single, batched.report, grader)
            print(f"{grader}: batching fixed {diff['fixed'] or 'none'}, broke {diff['regressed'] or 'none'}")
        saved = {
            "single": single.summary(),
            "batch": batched.report.summary(),
            "calls": batched.calls,
            "missing_first": batched.missing_first,
            "repeated": batched.repeated,
            "items": [
                {"ticket": t.text, "single": a.output, "batch": b.output}
                for t, a, b in zip(chosen, single.items, batched.report.items, strict=True)
            ],
        }

    elif args.what == "interview":
        parts = (ScriptedInterviewer(), NotesOwner(), RuleDrafter()) if offline else (executor, executor, executor)
        transcript, used = interview(parts[0], parts[1], questions=args.questions)
        drafted, drafting = draft_prompt(parts[2], transcript)
        for q, a in transcript:
            print(f"Q: {q}\nA: {a}\n")
        print(drafted.split("RULES:", 1)[1].strip(), "\n")
        reports: list[EvalReport] = []
        for split in args.split.split(","):
            items = [t for t in chosen if t.split == split]
            before, after = run_eval(executor, BASE_PROMPT, items), run_eval(executor, drafted, items)
            reports += [before, after]
            for grader in ("code", "priority"):
                diff = compare(before, after, grader)
                fixed, broke = diff["fixed"], diff["regressed"]
                assert isinstance(fixed, list) and isinstance(broke, list)
                print(
                    f"{split} {grader}: {diff['before']:.2f} -> {diff['after']:.2f}, "
                    f"fixed {len(fixed)}, broke {len(broke)}"
                )
        print(to_markdown(reports))
        print(
            f"Interview and drafting: {(used + drafting).input_tokens} tokens in, {(used + drafting).output_tokens} out"
        )
        saved = {"transcript": transcript, "prompt": drafted, "reports": [r.summary() for r in reports]}

    else:
        train = [t for t in tickets if t.split == "train"]
        dev = [t for t in tickets if t.split == "dev"]
        holdout = [t for t in tickets if t.split == "holdout"]
        check = validate(train, dev, holdout, epochs=args.epochs)
        print(
            "Validation: "
            + json.dumps({k: check[k] for k in ("train", "validation", "per_code", "leaks", "training_tokens")})
        )
        if not check["ok"]:
            print("error: the training data did not validate: " + "; ".join(check["problems"] + check["leaks"]))
            return 1
        tuning = args.tuning or ("local" if offline else "openai")
        backend = OpenAITuning() if tuning == "openai" else SimulatedTuning()
        job = run_job(
            backend,
            args.out / f"tuning-journal-{tuning}.json",
            train=train,
            dev=dev,
            model=args.base_model,
            epochs=args.epochs,
            workdir=args.out / "tuning",
            max_wait_s=args.wait,
            sleep=time.sleep if tuning == "openai" else (lambda _s: None),
        )
        print(f"Job {job.get('job', '-')}: {job['status']}" + (f" ({job.get('error')})" if job.get("error") else ""))
        if job["status"] not in ("succeeded", "refused", "failed", "cancelled"):
            print("The comparison waits for the job; run the same command again to resume it.")
            saved = {"job": job, "validation": check}
        else:
            if tuning == "openai" and job["status"] == "succeeded":
                tuned: ChatModel = make_model("openai", str(job["fine_tuned_model"]))
            else:
                if tuning == "openai":
                    print("Compared instead: a model trained on the same examples without a language model.")
                tuned = NearestTicketModel(train)
            prompted_ids = [m.strip() for m in (args.models or "gpt-5-mini,gpt-5-nano").split(",")]
            prompted = (
                [RuleFollowerModel()]
                if offline
                else [make_model(args.provider, m, effort=_effort(args, m)) for m in prompted_ids]
            )
            reports = [run_eval(tuned, TUNED_SYSTEM, holdout)]
            for model in prompted:
                try:
                    reports.append(run_eval(model, BASE_PROMPT, holdout))
                except ModelError as exc:  # a retired model is a finding, not a reason to lose the others
                    print(f"{model.name}: not compared ({exc})")
            print(to_markdown(reports))
            remote = tuning == "openai" and job["status"] == "succeeded"
            trained = int(job.get("trained_tokens") or check["training_tokens"])
            base = next((k for k in TRAINING_PRICE if args.base_model.startswith(k)), None)
            print("| Model | Holdout code | Holdout priority | Inference per 1,000 tickets | Training, once |")
            print("|---|---|---|---|---|")
            for report in reports:
                key = max((k for k in PRICES if report.model.startswith(k)), key=len, default=None)
                if report.model == tuned.name:
                    key = f"ft:{base}" if remote and base else None
                cost = per_thousand(report, PRICES[key]) if key else None
                training = (
                    trained * TRAINING_PRICE[base] / 1e6 if remote and base and report.model == tuned.name else 0.0
                )
                inference = (
                    f"${cost.per_thousand:.4f}" if cost else ("$0 (local)" if report.model == tuned.name else "-")
                )
                once = f"${training:.4f}" if training else "-"
                print(
                    f"| {report.model} | {report.rate('code'):.2f} | {report.rate('priority'):.2f} | {inference} | {once} |"
                )
            saved = {"job": job, "validation": check, "reports": [r.summary() for r in reports]}

    (args.out / f"{args.what}-{stamp}.json").write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
    return 0
