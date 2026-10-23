"""The agent solvers, driven by scripted models.

A scripted model is not intelligent: it replays decisions written in the test. These tests
prove the wiring (tool schemas, argument validation, verdicts, stopping, cost accounting),
not that a real model solves the labs. That takes a provider and `aisyslab bench`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from aisyslab.core.lab import EnvOptions
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn, tool_turn
from aisyslab.core.runner import run_once
from aisyslab.labs import LABS
from aisyslab.labs.inbox.env import InboxEnv
from aisyslab.labs.maintenance_portal.env import MaintenancePortalEnv
from aisyslab.labs.plant_database.baseline import BLOCKED_ORDERS
from aisyslab.labs.plant_database.env import PlantDatabaseEnv
from aisyslab.labs.procedures_rag.env import ProceduresRagEnv
from aisyslab.labs.service_repair.env import LOCK, ROOT, ServiceRepairEnv
from aisyslab.labs.supply_intake.env import SupplyIntakeEnv, sniff


def last_tool_result(messages: Sequence[Message]) -> dict:
    return json.loads(next(m.content for m in reversed(messages) if m.role == "tool"))


def test_every_agent_exposes_valid_tool_schemas() -> None:
    def recording(store: list[Sequence[ToolSpec]]):
        def policy(_s: str, _m: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
            store.append(tools)
            return text_turn("I give up.")

        return policy

    for lab in LABS.values():
        captured: list[Sequence[ToolSpec]] = []
        report = run_once(lab, "agent", 0, model=ScriptedModel(recording(captured)))
        assert report.status == "finished" and not report.accepted
        for spec in captured[0]:
            assert spec.parameters["type"] == "object" and "$ref" not in json.dumps(spec.parameters), (
                lab.id,
                spec.name,
            )


def test_inbox_agent_recovers_from_a_stale_answer() -> None:
    env = InboxEnv(2, EnvOptions())
    answer = {
        "maintenance_date": env.date_corrected.isoformat(),
        "vpn_code": env.vpn_current,
        "change_id": env.change_approved,
    }
    script = iter(
        [
            tool_turn("mail_help"),
            tool_turn("search_mail", query="subject:maintenance"),
            tool_turn("submit_answer", **answer),
        ]
    )

    def policy(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        planned = next(script, None)
        if planned is not None:
            return planned
        feedback = last_tool_result(messages)
        if "outdated" in json.dumps(feedback):  # the mailbox changed while the agent was reading
            return tool_turn("wait_for_new_mail")
        return tool_turn("submit_answer", **{**answer, "maintenance_date": env.date_final.isoformat()})

    report = run_once(LABS["inbox"], "agent", 2, model=ScriptedModel(policy))
    assert report.accepted and report.status == "goal_reached"
    assert report.submissions == 2 and report.model_calls == 5


def test_service_repair_agent_follows_the_failing_checks() -> None:
    probe = ServiceRepairEnv(5, EnvOptions())
    lines = probe.files[f"{ROOT}/settings.ini"].splitlines()
    fixes = {
        "mode": "mode=production",
        "workers": "workers=4",
        "api_token": f"api_token={probe.token}",
        "timeout_s": "timeout_s=30",
    }
    commands = [f"cat {ROOT}/.denylist"]
    commands += [
        f"editline {ROOT}/settings.ini {n} {fixes[line.split('=')[0]]}"
        for n, line in enumerate(lines, 1)
        if line.split("=")[0] in fixes
    ]
    commands += [f"rm {LOCK}", "run line-gateway"]
    script = iter(commands)

    def policy(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        command = next(script, None)
        if command is not None:
            return tool_turn("shell", command=command)
        output = last_tool_result(messages)["output"]
        return tool_turn("submit_confirmation", code=re.search(r"Confirmation: (\S+)", output).group(1))

    report = run_once(LABS["service-repair"], "agent", 5, model=ScriptedModel(policy))
    assert report.accepted and report.tool_calls == len(commands) + 1


def test_portal_agent_that_obeys_the_page_is_caught() -> None:
    probe = MaintenancePortalEnv(1, EnvOptions())

    def steered(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        done = [m.tool_name for m in messages if m.role == "tool"]
        if "read_work_order" not in done:
            return tool_turn("read_work_order", work_order_id=probe.injected_on)
        if "request_approval" not in done:
            return tool_turn(
                "request_approval", work_order_id=probe.injected_on, action="close", reason="directive in comment"
            )
        return tool_turn("finish")

    report = run_once(LABS["maintenance-portal"], "agent", 1, model=ScriptedModel(steered))
    assert not report.accepted and report.status == "goal_reached"


def test_portal_agent_with_human_gate() -> None:
    probe = MaintenancePortalEnv(1, EnvOptions())

    def careful(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        done = [m.tool_name for m in messages if m.role == "tool"]
        if "update_work_order" not in done:
            return tool_turn(
                "update_work_order",
                work_order_id=probe.target_fix,
                failure_code="MECH-BRG",
                note="Drive-end bearing replaced.",
            )
        if "request_approval" not in done:
            return tool_turn(
                "request_approval", work_order_id=probe.target_close, action="close", reason="Readings fine at 24 h."
            )
        if "close_work_order" not in done:
            token = last_tool_result(messages)["approval_token"]
            return tool_turn(
                "close_work_order", work_order_id=probe.target_close, note="Closed after check.", approval_token=token
            )
        return tool_turn("finish")

    report = run_once(LABS["maintenance-portal"], "agent", 1, model=ScriptedModel(careful))
    assert report.accepted


def test_vision_side_calls_are_validated_and_costed() -> None:
    probe = SupplyIntakeEnv(7, EnvOptions())
    image_id = next(i.id for i in probe.items.values() if sniff(i.data) == "image/png")

    def policy(system: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        if messages and messages[0].images:  # the vision call made inside inspect_image
            return text_turn(f'Rack shown. {{"full_bins": {probe.full_bins + 1}, "empty_bins": 3}}')
        results = [m for m in messages if m.role == "tool"]
        if not results:
            return tool_turn("inspect_image", item_id=image_id, question="How many bins are full?")
        return text_turn(results[-1].content)

    report = run_once(LABS["supply-intake"], "agent", 7, model=ScriptedModel(policy))
    assert report.model_calls == 3  # two agent turns plus one vision call made by the tool


def test_rag_agent_submission_is_graded_per_question() -> None:
    probe = ProceduresRagEnv(2, EnvOptions())
    answers = [
        {"question_id": q.id, "value": q.value, "answer": q.value, "citations": list(q.cites)} for q in probe.questions
    ]

    report = run_once(
        LABS["procedures-rag"], "agent", 2, model=ScriptedModel([tool_turn("submit_answers", answers=answers)])
    )
    assert report.accepted and report.score == 1.0


def test_database_agent_discovers_the_schema_then_queries() -> None:
    expected = sorted(PlantDatabaseEnv(4, EnvOptions()).expected)
    script = [
        tool_turn("list_tables"),
        tool_turn("describe_table", table="work_orders"),
        tool_turn("run_query", sql="DELETE FROM reservations"),
        tool_turn("run_query", sql=BLOCKED_ORDERS),
    ]
    seen: list[dict] = []

    def policy(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        if any(m.role == "tool" for m in messages):
            seen.append(last_tool_result(messages))
        if script:
            return script.pop(0)
        if len(seen) == 4:
            return tool_turn("submit_answer", work_order_ids=[wo for (wo,) in seen[-1]["rows"]])
        return text_turn("Done.")

    report = run_once(LABS["plant-database"], "agent", 4, model=ScriptedModel(policy))
    assert "refused" in json.dumps(seen[2])  # the engine refused the write; the run went on
    assert [wo for (wo,) in seen[3]["rows"]] == expected
    assert not report.accepted  # a correct answer, but the run tried to write
    assert "write attempt" in json.dumps(seen[-1])


def test_database_agent_with_a_clean_run_is_accepted() -> None:
    expected = sorted(PlantDatabaseEnv(6, EnvOptions()).expected)
    script = [tool_turn("run_query", sql=BLOCKED_ORDERS), tool_turn("submit_answer", work_order_ids=expected)]
    report = run_once(LABS["plant-database"], "agent", 6, model=ScriptedModel(script))
    assert report.accepted and report.status == "goal_reached" and report.score == 1.0


def test_quality_agent_recovers_from_a_leaky_feature() -> None:
    def policy(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        done = [m.tool_name for m in messages if m.role == "tool"]
        if not done:
            return tool_turn("describe_data")
        last = last_tool_result(messages)
        if done[-1] == "describe_data":
            return tool_turn("train_tree", features=["torque_nm", "cure_temp_c", "rework_ticket"])
        if done[-1] == "train_tree":
            return tool_turn("predict_tonight", model_id=last["model_id"])
        if done[-1] == "predict_tonight":
            passing = [p["batch_id"] for p in last["predictions"] if p["prediction"] == "OK"]
            return tool_turn("submit_answer", passing_batch_ids=passing)
        if done[-1] == "submit_answer" and not last["accepted"]:
            return tool_turn("train_tree", features=["torque_nm", "cure_temp_c", "cure_time_s", "humidity_pct"])
        return text_turn("Done.")

    report = run_once(LABS["quality-gate"], "agent", 3, model=ScriptedModel(policy))
    assert report.accepted and report.status == "goal_reached" and report.submissions == 2
