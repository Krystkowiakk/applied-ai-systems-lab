from __future__ import annotations

from pathlib import Path

import pytest

from aisyslab.cli import _seeds, main


def test_list_and_show(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    assert "procedures-rag" in capsys.readouterr().out
    assert main(["show", "maintenance-portal"]) == 0
    assert "Techniques" in capsys.readouterr().out


def test_run_baseline_writes_a_trace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "service-repair", "--seed", "1", "--quiet", "--trace-dir", str(tmp_path)]) == 0
    traces = list(tmp_path.glob("*.jsonl"))
    assert len(traces) == 1
    assert main(["trace", str(traces[0])]) == 0
    assert "ACCEPTED" in capsys.readouterr().out


def test_agent_without_provider_is_a_clear_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "inbox", "--solver", "agent"]) == 2
    assert "--provider" in capsys.readouterr().err


def test_bench_writes_results(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert (
        main(
            [
                "bench",
                "--labs",
                "inbox,procedures-rag",
                "--seeds",
                "0-1",
                "--out",
                str(out),
                "--trace-dir",
                str(tmp_path / "t"),
            ]
        )
        == 0
    )
    table = next(out.glob("bench-*.md")).read_text()
    assert "| inbox | baseline | - | 2/2 |" in table
    assert "| procedures-rag | baseline | - | 0/2 |" in table


def test_seed_specs() -> None:
    assert _seeds("0-3") == [0, 1, 2, 3]
    assert _seeds("2,5") == [2, 5]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["bench", "--labs", "nope"], "Unknown lab 'nope'"),
        (["bench", "--labs", "inbox", "--seeds", "x"], "use a range such as 0-4"),
        (["bench", "--labs", "inbox", "--solvers", "agnet"], "unknown solver agnet"),
        (["trace", "no-such-trace.jsonl"], "no trace file"),
    ],
)
def test_mistyped_arguments_end_in_a_message_not_a_traceback(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_bench_prices_cached_input_as_cached_when_asked() -> None:
    from aisyslab.core.bench import Row

    row = Row("lab", "agent", "m", 1, 1, 1.0, 1, 0, 0, 1, 1_000_000, 100_000, 1.0, {}, cached_input_tokens=800_000)
    assert row.cost(1.0, 10.0) == pytest.approx(2.0)  # cached priced as ordinary input
    assert row.cost(1.0, 10.0, 0.1) == pytest.approx(0.2 + 0.08 + 1.0)


def test_one_failed_model_call_does_not_end_a_paid_evaluation() -> None:
    from aisyslab.core.model import ModelError
    from aisyslab.evals.runner import run_eval
    from aisyslab.evals.triage import BASE_PROMPT, RuleFollowerModel, generate_tickets

    class Flaky(RuleFollowerModel):
        def complete(self, **kwargs: object):  # type: ignore[override]
            if "seal" in kwargs["messages"][-1].content:  # type: ignore[index]
                raise ModelError("503 after retries")
            return super().complete(**kwargs)  # type: ignore[arg-type]

    tickets = generate_tickets(0)[:12]
    report = run_eval(Flaky(), BASE_PROMPT, tickets)
    failed = [i for i in report.items if i.error]
    assert len(report.items) == 12 and failed and all(not i.passed("schema") for i in failed)
