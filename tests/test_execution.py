"""Entry validation precedes every recorded outcome and every spending decision."""

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from tests.test_clients import REQUEST, response
from tests.test_repairs import change
from tests.test_repairs import live as live
from tests.test_run import COMMIT
from tests.test_run import runner as runner
from zoon.pilot import clients, run


def over_cap(runner: run.Run) -> None:
    runner.session.cap = Decimal("0.001")
    runner.session.transcripts["second"].append(
        REQUEST, replace(response("second"), input_tokens=36000, output_tokens=0)
    )
    assert runner.session.spent > runner.session.cap


@pytest.mark.parametrize("step", run.STEPS)
@pytest.mark.parametrize("fault", ["dirty", "registration"])
def test_over_cap_still_refuses_changed_inputs(live: run.Run, step: str, fault: str) -> None:
    over_cap(live)
    if fault == "dirty":
        (live.root / "notes.txt").write_text("pending")
    else:
        change(live, "prereg/h1.md")
    with pytest.raises(clients.GuardRefusal):
        live.execute(step, live.registered_commit)
    assert not live.directory.exists()
    assert not (live.root / "docs/pilot/result.md").exists()


@pytest.mark.parametrize("step", run.STEPS[:4])
def test_cap_stop_writes_report_after_guard(runner: run.Run, step: str) -> None:
    for previous in run.STEPS[: run.STEPS.index(step)]:
        runner.save(previous, {})
    over_cap(runner)
    assert "stopped" in runner.execute(step, COMMIT)[0]
    lines = runner.execute("report")
    assert "no verdict: spending cap reached" in lines[0]
    assert runner.state("report")["status"] == "completed"
    assert "\n".join(lines) in (runner.directory / "result.md").read_text()


@pytest.mark.parametrize("status", ["stopped", "failed", "unrun"])
@pytest.mark.parametrize("step", run.STEPS[1:4])
def test_cap_cannot_skip_prerequisite(runner: run.Run, step: str, status: str) -> None:
    earlier = run.STEPS[: run.STEPS.index(step)]
    for previous in earlier[:-1]:
        runner.save(previous, {})
    if status != "unrun":
        runner.save(earlier[-1], {}, status=status)
    over_cap(runner)
    before = {p.name: p.read_bytes() for p in runner.directory.glob("*")}
    with pytest.raises(clients.GuardRefusal):
        runner.execute(step, COMMIT)
    assert before == {p.name: p.read_bytes() for p in runner.directory.glob("*")}


def test_cap_report_refuses_any_failed_step(runner: run.Run) -> None:
    runner.save("judge-check", {}, status="failed")
    over_cap(runner)
    with pytest.raises(clients.GuardRefusal, match="repeat step"):
        runner.execute("report")
    assert not (runner.directory / "report.json").exists()


def test_report_terminal_exception_is_failed_and_resumable(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.save("judge-check", {}, status="stopped", reason=clients.StopReason.CAP.value)
    from zoon.pilot import result

    original = result.render

    def stop(*args: object) -> list[str]:
        raise clients.TerminalStop(clients.StopReason.RECORDS)

    monkeypatch.setattr(result, "render", stop)
    assert "failed" in runner.execute("report")[0]
    assert runner.state("report")["status"] == "failed"
    monkeypatch.setattr(result, "render", original)
    assert "no verdict" in runner.execute("report")[0]
    assert runner.state("report")["status"] == "completed"


@pytest.mark.parametrize(
    "field,value", [("step", "judge-check"), ("mode", "other"), ("previous", {})]
)
def test_state_identity_and_prerequisite_set(runner: run.Run, field: str, value: object) -> None:
    runner.save("judge-check", {})
    runner.save("judge", {})
    path = runner.directory / "judge.json"
    document = json.loads(path.read_text())
    document[field] = value
    path.write_bytes(run.encoded(document))
    with pytest.raises(clients.GuardRefusal):
        runner.state("judge")


def test_state_copied_to_another_step_refuses(runner: run.Run) -> None:
    runner.save("judge-check", {})
    (runner.directory / "judge.json").write_bytes(
        (runner.directory / "judge-check.json").read_bytes()
    )
    with pytest.raises(clients.GuardRefusal):
        runner.state("judge")


def test_state_save_survives_interrupted_replace(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.save("judge-check", {"value": "old"}, status="failed")
    original = Path.replace

    def killed(path: Path, target: Path) -> Path:
        if target.name == "judge-check.json":
            raise KeyboardInterrupt
        return original(path, target)

    monkeypatch.setattr(Path, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        runner.save("judge-check", {"value": "new"})
    assert clients.obj(runner.state("judge-check")["payload"])["value"] == "old"


def test_refused_own_checkpoint_cannot_record_repair(live: run.Run) -> None:
    live.save("judge-check", {})
    path = live.directory / "judge-check.json"
    document = json.loads(path.read_text())
    document["transcripts"] = {}
    path.write_bytes(run.encoded(document))
    change(live, "src/zoon/pilot/clients.py")
    with pytest.raises(clients.GuardRefusal):
        live.execute("judge-check", repair="Correct request handling")
    assert live.binding()["repairs"] == []


@pytest.mark.parametrize("field", ["registered_commit", "mode"])
def test_transcript_stamp_prevents_reuse(runner: run.Run, field: str) -> None:
    runner.session.ask_many("decision", [REQUEST])
    path = runner.session.transcripts["decision"].path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0] == {"registered_commit": runner.registered_commit, "mode": runner.mode}
    rows[0][field] = "different"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(clients.GuardRefusal, match="identity"):
        runner.execute("judge-check")


def test_live_state_rejects_copied_stub_state(live: run.Run) -> None:
    synthetic = run.Run(live.root, synthetic=True)
    synthetic.save("judge-check", {})
    live.directory.mkdir(parents=True)
    (live.directory / "judge-check.json").write_bytes(
        (synthetic.directory / "judge-check.json").read_bytes()
    )
    with pytest.raises(clients.GuardRefusal, match="step or mode"):
        live.state("judge-check")
