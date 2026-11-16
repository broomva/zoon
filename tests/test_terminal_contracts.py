"""Offline regression cases for request and terminal-state contracts."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_clients import REQUEST, response, vertex_shape
from tests.test_run import COMMIT, finish
from tests.test_run import runner as runner
from zoon.pilot import clients, cohorts, materials, prompts, result, run


@pytest.mark.parametrize("fault", ["blocked", "empty", "limited"])
def test_recognition_unreadable_is_local(runner: run.Run, fault: str) -> None:
    runner.execute("judge-check")
    original = runner.session.clients["second"]

    class Second(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if not isinstance(request, str):
                return original.ask(request)
            shape = vertex_shape()
            candidate = clients.obj(clients.items(shape["candidates"])[0])
            candidate["finishReason"] = "MAX_TOKENS" if fault == "limited" else "STOP"
            candidate["content"] = {"parts": [{"text": "unknown" if fault == "limited" else ""}]}
            if fault == "blocked":
                candidate["finishReason"] = "SAFETY"
            return clients.VertexClient().read(request, shape)

    runner.session.clients["second"] = Second("second")
    assert "completed" in runner.execute("judge")[0]
    judged = clients.obj(runner.state("judge")["payload"])
    assert clients.obj(judged["second"])["usable"]
    recognition = clients.obj(judged["recognition"])
    assert bool(recognition) == (fault == "limited")
    runner.session.check_family("second")
    runner.save("calibrate", {}, status="stopped", reason="terminal stop")
    lines = runner.execute("report")
    recognition_lines = [line for line in lines if line.startswith("[stub recognition]")]
    assert len(recognition_lines) == 13
    assert all(("not run" in line) == (fault != "limited") for line in recognition_lines)


def test_cap_wins_over_retryable_window_failure(runner: run.Run) -> None:
    failed_key = min(
        clients.key(prompts.evaluative(record))
        for record in runner.records
        if record["set"] == "outside"
    )

    class Expensive(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if clients.key(request) == failed_key:
                raise clients.StepFailure("try again")
            return replace(super().ask(request), input_tokens=4_000_000_000)

    runner.session.clients["decision"] = Expensive("decision")
    assert "stopped" in runner.execute("judge-check")[0]
    assert "cap reached" in str(runner.state("judge-check")["reason"])


@pytest.mark.parametrize("step", run.STEPS[:4])
def test_every_step_checks_recorded_cap_before_work(runner: run.Run, step: str) -> None:
    runner.session.transcripts["decision"].append(
        REQUEST, replace(response(), input_tokens=4_000_000_000)
    )
    assert "stopped" in runner.execute(step, COMMIT)[0]
    assert "cap reached" in str(runner.state(step)["reason"])
    assert "cap reached" in runner.execute("report")[0]


def test_cap_report_overrides_completed_state(runner: run.Run) -> None:
    states = finish(runner)
    runner.session.transcripts["decision"].append(
        REQUEST, replace(response(), input_tokens=4_000_000_000)
    )
    assert "no verdict: spending cap reached" in result.render(runner, states)[0]
    assert "no verdict: spending cap reached" in runner.execute("report")[0]


def test_failed_step_report_refuses_without_result(runner: run.Run) -> None:
    runner.save("judge-check", {}, status="failed", reason="interrupted")
    with pytest.raises(clients.StopRun, match=r"repeat.*judge-check"):
        runner.execute("report")
    assert not (runner.directory / "result.md").exists()


@pytest.mark.parametrize("fault_step", ["calibrate", "held-out"])
def test_stopped_report_suppresses_unusable_second_numbers(
    runner: run.Run, fault_step: str
) -> None:
    for step in run.STEPS[: run.STEPS.index(fault_step)]:
        runner.execute(step)
    runner.session.transcripts["second"].append(
        replace(REQUEST, state="bad share"), replace(response("second"), share=0.4)
    )
    runner.save(fault_step, {}, status="stopped", reason="spending cap reached; no verdict")
    lines = runner.execute("report")
    assert any(
        "second availability = second family declared-answer share" in line for line in lines
    )
    assert not any(
        "second" in line
        and line.startswith(
            (
                "[stub reversal]",
                "[stub agreement with norm]",
                "[stub strict map]",
                "[stub judge calibration]",
                "[stub fitted]",
                "[stub parameter count]",
            )
        )
        for line in lines
    )


@pytest.mark.parametrize(
    "reason", ["decision model version changed; no verdict", "decision model withdrawn; no verdict"]
)
def test_decision_availability_names_terminal_fault(runner: run.Run, reason: str) -> None:
    runner.execute("judge-check")
    runner.save("judge", {}, status="stopped", reason=reason)
    assert f"[stub family] decision availability = {reason}" in runner.execute("report")


def test_completed_step_reason_matches_first_execution(runner: run.Run) -> None:
    assert runner.execute("judge-check") == runner.execute("judge-check")


@pytest.mark.parametrize("step", run.STEPS[:4])
@pytest.mark.parametrize("fault", ["rewrites", "texts", "cohorts", "fixtures"])
def test_live_material_guards_precede_any_request(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, step: str, fault: str
) -> None:
    runner.synthetic = False

    def guarded(*args: object) -> None:
        pass

    monkeypatch.setattr(runner, "guard", guarded)
    people = run.stub.people(48, 41)
    document = cohorts.generate(people, calibration_size=8)
    monkeypatch.setattr(runner, "training", lambda: (people, document))
    target = runner.root / "prereg" / "requests"
    for old in target.iterdir():
        if old.name != "README.md":
            old.unlink()
    made = prompts.fixtures(people, document, materials.load(), runner.records)
    for name, content in made.items():
        (target / name).write_bytes(content)
    (runner.root / "prereg" / "cohorts.json").write_bytes(cohorts.serialise(document))
    if fault == "rewrites":
        (runner.root / "pilot" / "rewrites.sha256").write_text("wrong\n")
    elif fault == "texts":
        (runner.root / "pilot" / "task-texts.sha256").write_text("wrong\n")
    elif fault == "cohorts":
        (runner.root / "prereg" / "cohorts.json").write_text("{}\n")
    else:
        (target / next(iter(made))).write_bytes(b"wrong")
    runner.deps = run.dependencies(runner.root)
    with pytest.raises(clients.StopRun, match=r"materials|cohorts|fixtures|digest"):
        runner.execute(step, COMMIT)
    assert not runner.session.transcripts["decision"].entries


def test_recognition_keys_use_supplied_records() -> None:
    from zoon.pilot import recognition, rewrites

    records = copy.deepcopy(rewrites.load())
    next(r for r in records if r["id"] == "Medicine")["removed"] = ["alternate marker"]
    assert recognition.keys("Medicine", records)[0] == "alternate marker"


def test_stub_steps_ignore_dirty_tree(runner: run.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    def dirty(root: Path, *args: str) -> str:
        return " M tracked" if args[0] == "status" else COMMIT

    monkeypatch.setattr(run, "git", dirty)
    finish(runner)
    assert runner.execute("report")[0].startswith("[stub verdict]")


def test_interrupted_final_window_cannot_resume_past_cap(runner: run.Run) -> None:
    recorded = run.stub.client("second", runner.records).recorded
    runner.session.workers = 1
    calls: list[int] = []

    class Expensive(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            calls.append(1)
            reply = super().ask(request)
            return replace(reply, input_tokens=2_000_000_000) if len(calls) == 25 else reply

    runner.session.clients["second"] = Expensive("second", recorded)
    assert "stopped" in runner.execute("judge-check")[0]
    assert len(runner.session.transcripts["second"].entries) == 25
    for name in ("judge-check.json", "judge-check-values.json"):
        (runner.directory / name).unlink()
    replay = run.Run(runner.root, synthetic=True, supplied_clients=[])
    assert "stopped" in replay.execute("judge-check")[0]
    assert "cap reached" in str(replay.state("judge-check")["reason"])
    assert "no verdict: spending cap reached" in replay.execute("report")[0]


def test_stub_repair_accepts_dirty_tree(runner: run.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    runner.execute("judge-check")
    changed = runner.root / "src" / "zoon" / "pilot" / "clients.py"
    changed.write_text(changed.read_text() + "\n")
    replay = run.Run(runner.root, synthetic=True)

    def dirty(root: Path, *args: str) -> str:
        return " M tracked" if args[0] == "status" else COMMIT

    monkeypatch.setattr(run, "git", dirty)
    assert "completed" in replay.execute("judge", repair="Restore reply handling")[0]


@pytest.mark.parametrize("step", run.STEPS[:4])
def test_live_material_guards_accept_regenerated_inputs(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    def guarded(*args: object) -> None:
        pass

    monkeypatch.setattr(runner, "guard", guarded)
    people = run.stub.people(48, 41)
    document = cohorts.generate(people, calibration_size=8)
    monkeypatch.setattr(runner, "training", lambda: (people, document))
    target = runner.root / "prereg" / "requests"
    for old in target.iterdir():
        if old.name != "README.md":
            old.unlink()
    for name, content in prompts.fixtures(
        people, document, materials.load(), runner.records
    ).items():
        (target / name).write_bytes(content)
    (runner.root / "prereg" / "cohorts.json").write_bytes(cohorts.serialise(document))
    runner.deps = run.dependencies(runner.root)
    for earlier in run.STEPS[: run.STEPS.index(step)]:
        runner.save(earlier, {})

    def complete() -> run.State:
        return runner.save(step, {})

    def held_out(registered: str | None) -> run.State:
        return complete()

    method = {
        "judge-check": "judge_check",
        "judge": "judge_all",
        "calibrate": "calibrate",
        "held-out": "held_out",
    }[step]
    monkeypatch.setattr(runner, method, held_out if step == "held-out" else complete)
    runner.synthetic = False
    assert "completed" in runner.execute(step, COMMIT)[0]
    assert not runner.session.transcripts["decision"].entries


@pytest.mark.parametrize("step", run.STEPS[:4])
def test_small_session_cap_stops_before_any_step_work(
    runner: run.Run, step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from decimal import Decimal

    runner.session.cap = Decimal("0.001")
    runner.session.transcripts["second"].append(
        REQUEST, replace(response("second"), input_tokens=36000, output_tokens=0)
    )
    assert runner.session.spent == Decimal("0.0036")

    def forbidden() -> None:
        raise AssertionError("work started above the cap")

    monkeypatch.setattr(runner, "preflight", forbidden)
    assert "stopped" in runner.execute(step, COMMIT)[0]
    assert "cap reached" in runner.execute("report")[0]


def test_small_session_cap_overrides_completed_and_failed_state(runner: run.Run) -> None:
    from decimal import Decimal

    states = finish(runner)
    runner.session.cap = Decimal("0.001")
    assert "no verdict: spending cap reached" in result.render(runner, states)[0]
    assert "no verdict: spending cap reached" in runner.execute("report")[0]
    runner.save("judge-check", {}, status="failed", reason="interrupted")
    assert "no verdict: spending cap reached" in runner.execute("report")[0]


def test_small_session_cap_report_without_any_saved_step(runner: run.Run) -> None:
    from decimal import Decimal

    runner.session.cap = Decimal("0.001")
    runner.session.transcripts["second"].append(
        REQUEST, replace(response("second"), input_tokens=36000, output_tokens=0)
    )
    assert "no verdict: spending cap reached" in runner.execute("report")[0]
