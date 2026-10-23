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
