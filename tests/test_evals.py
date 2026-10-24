"""Evaluation: the dataset, graders, runs and comparisons, prompt optimisation and the judge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from aisyslab.cli import main
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn
from aisyslab.core.runner import run_once
from aisyslab.evals.graders import DETERMINISTIC, code, judged_reason, parse, priority, schema
from aisyslab.evals.judge import HUMAN_LABELS, KeywordJudge, LLMJudge, calibrate, cohen_kappa
from aisyslab.evals.optimize import optimize_prompt
from aisyslab.evals.runner import compare, fingerprint, run_eval, to_markdown
from aisyslab.evals.triage import (
    BASE_PROMPT,
    HOLDOUT_SYMPTOMS,
    RuleFollowerModel,
    RuleProposerModel,
    Ticket,
    generate_tickets,
)
from aisyslab.labs import LABS
from aisyslab.mcp.route import routed

TICKET = Ticket("T-1", "coolant pump 2: grinding noise at the drive end. line stopped.", "MECH-BRG", 1, "train")


def test_dataset_is_seeded_split_and_fingerprinted() -> None:
    tickets = generate_tickets(0)
    assert tickets == generate_tickets(0) and tickets != generate_tickets(1)
    counts = {s: sum(t.split == s for t in tickets) for s in ("train", "dev", "holdout")}
    assert counts == {"train": 54, "dev": 18, "holdout": 18}
    assert len({t.text for t in tickets}) == len(tickets)  # no ticket twice, so none crosses splits
    unseen = {w for words in HOLDOUT_SYMPTOMS.values() for w in words}
    assert not any(w in t.text for t in tickets if t.split != "holdout" for w in unseen)
    assert fingerprint(tickets) == fingerprint(generate_tickets(0)) != fingerprint(tickets[1:])


def test_graders_check_format_code_and_priority() -> None:
    good = '```json\n{"code": "MECH-BRG", "priority": 1, "reason": "Grinding at the drive end."}\n```'
    assert parse(good) == {"code": "MECH-BRG", "priority": 1, "reason": "Grinding at the drive end."}
    assert all(g(TICKET, good).passed for g in DETERMINISTIC)
    bad = schema(TICKET, '{"code": "MECH-XYZ", "priority": 5}')
    assert not bad.passed and "unknown code" in bad.note and "priority 5" in bad.note and "no reason" in bad.note
    assert schema(TICKET, "The bearing.").note == "not a JSON object"
    wrong = '{"code": "MECH-SEAL", "priority": 2, "reason": "x"}'
    assert code(TICKET, wrong).note == "expected MECH-BRG, got MECH-SEAL"
    assert priority(TICKET, wrong).note == "expected 1, got 2"


def test_run_eval_scores_the_starting_prompt_and_keeps_order() -> None:
    tickets = generate_tickets(0)
    report = run_eval(RuleFollowerModel(), BASE_PROMPT, tickets)
    assert [i.ticket_id for i in report.items] == [t.id for t in tickets]
    # The starting prompt only knows the default rule, so it gets one code in six right.
    assert report.rate("code") == 0.167 and report.rate("schema") == 1.0 and report.rate("priority") == 1.0
    assert report.confusion()["ELEC-MTR"] == {"MECH-BRG": 15}
    assert report.usage.input_tokens > 0
    assert report.summary()["dataset"] == fingerprint(tickets)


def test_compare_lists_fixed_and_regressed_items() -> None:
    tickets = generate_tickets(0)[:12]
    before = run_eval(RuleFollowerModel(), BASE_PROMPT, tickets)
    seal_rule = "- if the ticket mentions 'seal' -> MECH-SEAL\n"
    after = run_eval(RuleFollowerModel(), BASE_PROMPT.replace("RULES:\n", "RULES:\n" + seal_rule), tickets)
    result = compare(before, after)
    fixed = [t.id for t in tickets if t.code == "MECH-SEAL" and "seal" in t.text]
    regressed = [t.id for t in tickets if t.code == "MECH-BRG" and "seal" in t.text]
    assert result["fixed"] == sorted(fixed) and result["regressed"] == sorted(regressed)
    with pytest.raises(ValueError, match="different datasets"):
        compare(before, run_eval(RuleFollowerModel(), BASE_PROMPT, tickets[1:]))


def test_markdown_table_marks_graders_a_run_did_not_use() -> None:
    tickets = generate_tickets(0)[:6]
    plain = run_eval(RuleFollowerModel(), BASE_PROMPT, tickets)
    judged = run_eval(
        RuleFollowerModel("judged"), BASE_PROMPT, tickets, [*DETERMINISTIC, judged_reason(LLMJudge(KeywordJudge()))]
    )
    table = to_markdown([plain, judged]).splitlines()
    assert "| reason |" in table[0]
    assert table[2].split("|")[8].strip() == "-" and table[3].split("|")[8].strip() != "-"


def test_optimiser_improves_train_and_the_holdout_shows_the_gap() -> None:
    result = optimize_prompt(RuleFollowerModel(), RuleProposerModel(), BASE_PROMPT, generate_tickets(0))
    assert result.rounds[0].train == 0.167 and max(r.train for r in result.rounds) == 1.0
    assert result.rounds[-1].note == "no failures left on train"
    assert result.final["dev"]["best"] > result.final["dev"]["start"]
    # Rules learned from training wordings miss the wordings only the holdout has.
    assert result.final["holdout"]["start"] < result.final["holdout"]["best"] < 1.0
    assert result.best_prompt.count("- if the ticket mentions") > 0 and result.usage.output_tokens > 0


def test_optimiser_discards_a_candidate_that_does_not_help() -> None:
    same = ScriptedModel([text_turn(BASE_PROMPT)] * 2)
    result = optimize_prompt(RuleFollowerModel(), same, BASE_PROMPT, generate_tickets(0), rounds=2)
    assert [r.kept for r in result.rounds] == [True, False, False]
    assert result.best_prompt == BASE_PROMPT
    assert result.final["holdout"]["start"] == result.final["holdout"]["best"]


def test_cohen_kappa_measures_agreement_beyond_chance() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([True, True, True], [True, True, True]) == 1.0


def test_calibration_shows_how_far_a_judge_can_be_trusted() -> None:
    crude = calibrate(LLMJudge(KeywordJudge()))
    assert crude["n"] == len(HUMAN_LABELS) and crude["agreement"] == 0.583 and crude["kappa"] == 0.167
    assert any("Anna" in d["reason"] for d in crude["disagreements"])  # type: ignore[attr-defined]

    def human_like(_system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        item = json.loads(messages[-1].content)
        label = next(x for x in HUMAN_LABELS if (x.ticket, x.reason) == (item["ticket"], item["reason"]))
        return text_turn("PASS\nNames the symptom." if label.human_pass else "FAIL\nAdds facts.")

    judge = LLMJudge(ScriptedModel(human_like))
    assert calibrate(judge) == {"n": 12, "agreement": 1.0, "kappa": 1.0, "disagreements": []}
    assert judge.usage.input_tokens > 0


def test_eval_cli_runs_offline_and_writes_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "triage", "--judge", "--out", str(tmp_path)]) == 0
    assert "| rule-follower |" in capsys.readouterr().out
    assert main(["eval", "optimize", "--rounds", "2", "--out", str(tmp_path)]) == 0
    assert "holdout" in capsys.readouterr().out
    assert main(["eval", "judge", "--out", str(tmp_path)]) == 0
    saved = json.loads(next(tmp_path.glob("optimize-*.json")).read_text())
    assert saved["rounds"][0]["number"] == 0 and "best_prompt" in saved
    assert {p.suffix for p in tmp_path.iterdir()} == {".md", ".json"}


def test_promptless_routing_drops_only_the_system_prompt() -> None:
    seen: list[str] = []

    def policy(system: str, _messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        seen.append(system)
        assert tools, "the tools stay"
        return text_turn("Done.")

    lab = LABS["plant-database"]
    run_once(routed(lab, promptless=True), "agent", 0, model=ScriptedModel(policy))
    run_once(lab, "agent", 0, model=ScriptedModel(policy))
    assert seen[0] == "" and seen[-1] == lab.spec.system  # type: ignore[union-attr]
