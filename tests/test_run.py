"""The command order, sealed fifth step and full synthetic report."""

import copy
import json
import shutil
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from tests.synthetic import write_raw
from tests.tracked import copy_tracked
from zoon.pilot import clients, cohorts, fetch, prompts, result, run, split, stub

COMMIT = "b" * 40


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> run.Run:
    copy_tracked(tmp_path)

    def git(root: Path, *arguments: str) -> str:
        return (
            COMMIT
            if arguments in (("rev-parse", "HEAD"), ("rev-parse", "refs/tags/registered^{commit}"))
            else ""
        )

    def sealed(raw_dir: Path) -> object:
        raise AssertionError("the real held-out loader is unreachable in a stub run")

    monkeypatch.setattr(run, "git", git)
    monkeypatch.setattr(fetch, "load_registered_held_out", sealed)
    return run.Run(tmp_path, synthetic=True, workers=3)


def finish(runner: run.Run) -> dict[str, run.State]:
    for step in run.STEPS[:-1]:
        lines = runner.execute(step, COMMIT)
        assert all(line.startswith("[stub ") for line in lines)
        assert "status = completed" in lines[0], lines
    return {step: runner.state(step) for step in run.STEPS[:-1]}


def test_full_synthetic_sequence_restart_and_constructed_verdicts(runner: run.Run) -> None:
    live_result = runner.root / "docs" / "pilot" / "result.md"
    live_before = live_result.read_bytes() if live_result.exists() else None
    states = finish(runner)
    lines = runner.execute("report")
    assert lines[0].startswith("[stub verdict] outcome = ")
    assert all(line.startswith("[stub ") for line in lines)
    target = runner.directory / "result.md"
    assert "\n".join(lines) in target.read_text()
    assert (live_result.read_bytes() if live_result.exists() else None) == live_before
    before = {f: t.digest() for f, t in runner.session.transcripts.items()}
    for step in run.STEPS:
        runner.execute(step, COMMIT)
    assert before == {f: t.digest() for f, t in runner.session.transcripts.items()}
    assert runner.state("report")["status"] == "completed"
    held = clients.obj(states["held-out"]["payload"])
    base = clients.obj(held["logs"])
    shape = run.floats(base["decision/zoon-evaluative"]).shape

    def rendered(mean: float, spread: float = 0.0) -> list[str]:
        changed = copy.deepcopy(states)
        payload = clients.obj(changed["held-out"]["payload"])
        logs = clients.obj(payload["logs"])
        for name in logs:
            logs[name] = np.full(shape, -1.0).tolist()
        improvement = np.full(shape, mean)
        improvement[::2] += spread
        improvement[1::2] -= spread
        logs["decision/zoon-evaluative"] = (-1 + improvement).tolist()
        return result.render(runner, changed)

    assert rendered(0.01)[0] == "[stub verdict] outcome = pass"
    assert rendered(0.001)[0] == "[stub verdict] outcome = fail"
    assert rendered(0.01, 0.2)[0] == "[stub verdict] outcome = fail"
    broken = copy.deepcopy(states)
    del clients.obj(clients.obj(broken["held-out"]["payload"])["logs"])["decision/prompted"]
    assert (
        result.render(runner, broken)[0]
        == "[stub verdict] outcome = no verdict: complete-records failure; no verdict"
    )
    for label in (
        "H1-strict",
        "reference",
        "trials 2-13",
        "dilemma interval",
        "reading share",
        "recall",
        "recognition",
        "recognition filtered",
        "fitted",
        "declared answers",
        "spending",
        "transcript",
        "run",
    ):
        assert any(line.startswith(f"[stub {label}]") for line in lines), label
    # Dropping all targets gives no estimate and does not change the original verdict.
    filtered = copy.deepcopy(states)
    recognition = clients.obj(clients.obj(filtered["judge"]["payload"])["recognition"])
    for value in recognition.values():
        clients.obj(value)["flagged"] = True
    repeated = result.render(runner, filtered)
    assert repeated[0] == lines[0]
    assert all(
        "not estimable" in line
        for line in repeated
        if line.startswith("[stub recognition filtered]")
    )
    # A second-family record failure removes only those results.
    removed = copy.deepcopy(states)
    payload = clients.obj(removed["held-out"]["payload"])
    payload["logs"] = {n: v for n, v in base.items() if not n.startswith("second/")}
    payload["unavailable"] = {"second": "complete-records failure"}
    assert result.render(runner, removed)[0] == lines[0]


@pytest.mark.parametrize("step", ["judge", "calibrate", "held-out", "report"])
def test_out_of_order_refuses(runner: run.Run, step: str) -> None:
    with pytest.raises(clients.GuardRefusal, match="has not completed"):
        runner.execute(step, COMMIT)
    assert not runner.session.transcripts["decision"].entries


def test_stub_steps_skip_registration_baselines(runner: run.Run) -> None:
    runner.execute("judge-check")
    target = runner.root / "prereg" / "h1.md"
    target.write_text(target.read_text() + "\n")
    restarted = run.Run(runner.root, synthetic=True)
    assert "completed" in restarted.execute("judge")[0]


def test_changed_transcript_and_previous_state_refuse(runner: run.Run) -> None:
    runner.execute("judge-check")
    path = runner.session.transcripts["decision"].path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["response"]["input_tokens"] = 50
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(clients.GuardRefusal, match="checkpoint changed"):
        runner.execute("judge")


def test_stop_rule_reports_and_never_unseals(runner: run.Run) -> None:
    records = [r for r in runner.records if r["set"] == "outside"]
    recorded = {
        clients.key(prompts.evaluative(r)): clients.Response(
            dict.fromkeys(map(str, range(1, 8)), 1 / 7), None, "", "jev-1.14.0", 10, 0
        )
        for r in records
    }
    runner.session.clients["decision"] = clients.StubClient("decision", recorded)
    assert "status = stopped" in runner.execute("judge-check")[0]
    with pytest.raises(clients.GuardRefusal, match="did not pass"):
        runner.execute("judge")
    assert (
        runner.execute("report")[0] == "[stub verdict] outcome = the judge could not be calibrated"
    )


def test_second_family_unusable_does_not_change_decision_pipeline(runner: run.Run) -> None:
    client = stub.client("second", runner.records)
    recorded = {k: replace(r, share=0.49) for k, r in client.recorded.items()}
    runner.session.clients["second"] = clients.StubClient("second", recorded)
    finish(runner)
    lines = runner.execute("report")
    assert "no verdict" not in lines[0]
    assert any("second family declared-answer share below one half" in line for line in lines)
    assert len(clients.obj(clients.obj(runner.state("judge")["payload"])["recognition"])) == 13
    assert not any(
        n.startswith("second/")
        for n in clients.obj(clients.obj(runner.state("held-out")["payload"])["logs"])
    )


def test_cap_reached_reports_no_verdict(runner: run.Run) -> None:
    runner.session.cap = Decimal(0)
    lines = runner.execute("judge-check")
    assert "status = stopped" in lines[0]
    assert (
        runner.execute("report")[0]
        == "[stub verdict] outcome = no verdict: spending cap reached; no verdict"
    )
    assert runner.session.spent == 0


def test_real_held_out_loader_only_on_synthetic_raw_files(tmp_path: Path) -> None:
    intended = write_raw(tmp_path, 40)
    table = fetch.load_registered_held_out(tmp_path)
    expected = {
        (e, split.participant_index(e, raw))
        for e, raw in intended
        if split.assign(e, raw) == "held_out"
    }
    people = cohorts.participants(table)
    assert {(p.experiment, p.participant) for p in people} == expected
    assert all(len(p.trials) == 13 for p in people)
    assert not expected & {
        (p.experiment, p.participant) for p in cohorts.participants(fetch.load(tmp_path))
    }


def test_live_clients_are_lazy_and_smoke_is_not_run(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> str:
        raise AssertionError("credentials must not be read")

    monkeypatch.setattr(clients, "access_token", forbidden)
    live = run.Run(runner.root)
    assert not live.session.transcripts["decision"].entries
    assert not live.session.transcripts["second"].entries
    assert run.smoke.__name__ == "smoke"


def test_stub_records_are_synthetic_and_reproducible() -> None:
    people = stub.people(12, 4)
    assert people == stub.people(12, 4)
    assert len(people) == 12 and all(p.participant.startswith("synthetic-") for p in people)
    assert len({(p.experiment, p.participant) for p in people}) == 12


def test_complete_record_failure_in_scoring_stops_decision_only(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    for step in run.STEPS[:3]:
        runner.execute(step)
    original = run.arms.predict

    def broken(
        rotations: list[run.arms.Rotation], data: run.models.Sequences
    ) -> dict[str, list[run.models.Floats]]:
        predicted = original(rotations, data)
        predicted["decision/zoon-evaluative"][0][:] = np.nan
        return predicted

    monkeypatch.setattr(run.arms, "predict", broken)
    assert "status = stopped" in runner.execute("held-out", COMMIT)[0]
    assert "complete-records" in runner.execute("report")[0]


def test_bad_registration_flag_does_not_write_failed_state(runner: run.Run) -> None:
    for step in run.STEPS[:3]:
        runner.execute(step)
    with pytest.raises(clients.GuardRefusal, match="must match"):
        runner.execute("held-out", "wrong")
    assert not (runner.directory / "held-out.json").exists()


def test_report_uses_no_client_even_after_restart(runner: run.Run) -> None:
    finish(runner)
    restarted = run.Run(runner.root, synthetic=True, supplied_clients=[])
    assert restarted.execute("report")[0].startswith("[stub verdict]")


def test_manifest_is_checked_without_loading_records(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(raw, 3)
    manifest = tmp_path / "manifest.json"
    fetch.fetch(raw, manifest)
    run.verify_data(raw, manifest)
    changed = raw / fetch.SOURCES[1].filename
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(clients.GuardRefusal, match="manifest"):
        run.verify_data(raw, manifest)


def test_changed_step_output_refuses(runner: run.Run) -> None:
    runner.execute("judge-check")
    path = runner.directory / "judge-check.json"
    state = json.loads(path.read_text())
    state["payload"]["changed"] = True
    path.write_bytes(run.encoded(state))
    with pytest.raises(clients.GuardRefusal, match="output digest"):
        runner.execute("judge")


def test_withdrawn_second_family_still_attempts_recognition(runner: run.Run) -> None:
    attempts: list[bool] = []

    class Unavailable(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            attempts.append(isinstance(request, str))
            raise clients.FamilyUnusable(clients.FamilyCause.WITHDRAWN)

    runner.session.clients["second"] = Unavailable("second")
    finish(runner)
    assert sum(attempts) == 13
    lines = runner.execute("report")
    assert "no verdict" not in lines[0]
    assert any("second family withdrawn" in line for line in lines)
    assert all(
        "not estimable" in line for line in lines if line.startswith("[stub recognition filtered]")
    )


def test_failed_step_can_resume_without_resending_answers(runner: run.Run) -> None:
    counts: dict[str, int] = {}
    faults = [1]
    recorded = stub.client("decision", runner.records).recorded

    class Flaky(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            digest = clients.key(request)
            counts[digest] = counts.get(digest, 0) + 1
            if faults:
                faults.pop()
                raise clients.StepFailure("transport interrupted")
            return super().ask(request)

    runner.session.clients["decision"] = Flaky("decision", recorded)
    assert "status = failed" in runner.execute("judge-check")[0]
    answered = set(runner.session.transcripts["decision"].entries)
    assert answered
    with pytest.raises(clients.GuardRefusal, match="did not pass"):
        runner.execute("judge")
    assert "status = completed" in runner.execute("judge-check")[0]
    assert all(counts[digest] == 1 for digest in answered)
    assert len(runner.session.transcripts["decision"].entries) == 25


@pytest.mark.parametrize("fault_step", run.STEPS[:4])
def test_malformed_second_transport_preserves_decision_results(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, fault_step: str
) -> None:
    baseline = finish(runner)
    expected = runner.session.transcripts["decision"].path.read_bytes()
    verdict = runner.execute("report")[0]
    shutil.rmtree(runner.root / "data")
    replay = run.Run(runner.root, synthetic=True)
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")

    def transport(*args: object) -> tuple[int, dict[str, object]]:
        return 200, {"candidates": []}

    class RecordedTransport(clients.VertexClient):
        def preflight(self) -> None:
            pass

    for step in run.STEPS[:4]:
        if step == fault_step:
            replay.session.clients["second"] = RecordedTransport(lambda: "fixture-token", transport)
        assert "status = completed" in replay.execute(step, COMMIT)[0]
        assert clients.obj(replay.state(step)["payload"]).get("decision") == clients.obj(
            baseline[step]["payload"]
        ).get("decision")
    assert replay.session.transcripts["decision"].path.read_bytes() == expected
    actual = clients.obj(clients.obj(replay.state("held-out")["payload"])["logs"])
    original = clients.obj(clients.obj(baseline["held-out"]["payload"])["logs"])
    assert {k: v for k, v in actual.items() if k.startswith("decision/")} == {
        k: v for k, v in original.items() if k.startswith("decision/")
    }
    assert replay.execute("report")[0] == verdict


@pytest.mark.parametrize("status", [401, 403, "missing"])
@pytest.mark.parametrize("step", run.STEPS[:4])
def test_preflight_operator_fault_precedes_every_step_request(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, status: int | str, step: str
) -> None:
    for earlier in run.STEPS[: run.STEPS.index(step)]:
        runner.execute(earlier)
    recorded = {family: t.digest() for family, t in runner.session.transcripts.items()}
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    if status == "missing":
        monkeypatch.delenv("ZOON_VERTEX_PROJECT")
    calls: list[int] = []
    refreshes: list[int] = []

    def token() -> str:
        refreshes.append(1)
        return "fixture-token"

    def transport(*args: object) -> tuple[int, dict[str, object]]:
        calls.append(1)
        return int(status), {}

    runner.session.clients["second"] = clients.VertexClient(
        token, transport, preflight_transport=transport
    )
    assert "status = failed" in runner.execute(step, COMMIT)[0]
    assert runner.state(step)["reason"] == "StepFailure"
    assert recorded == {family: t.digest() for family, t in runner.session.transcripts.items()}
    assert len(calls) == (0 if status == "missing" else 2 if status == 401 else 1)
    assert len(refreshes) == len(calls)


def test_held_out_checks_stopped_prerequisite_before_loader(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.save("judge-check", {}, status="completed")
    runner.save("judge", {}, status="completed")
    runner.save("calibrate", {}, status="stopped", reason=clients.StopReason.CAP.value)
    reached: list[bool] = []

    def loader(directory: Path) -> None:
        reached.append(True)

    monkeypatch.setattr(fetch, "load_registered_held_out", loader)
    with pytest.raises(clients.GuardRefusal, match="calibrate did not pass"):
        runner.execute("held-out", COMMIT)
    assert not reached


def test_smoke_covers_seven_and_nine_answers_without_materials() -> None:
    lines = run.smoke(True)
    assert len(lines) == 6
    for family in clients.budget.FAMILIES:
        selected = [line for line in lines if f"smoke {family}]" in line]
        assert len(selected) == 3
        assert any("answers = 7" in line for line in selected)
        assert any("answers = 9" in line for line in selected)
        assert all("sum before division = " in line and "share = " in line for line in selected)


def test_report_share_summary_by_family_and_kind(runner: run.Run) -> None:
    finish(runner)
    lines = runner.execute("report")
    for family in clients.budget.FAMILIES:
        for kind in ("evaluative", "strict", "prompted"):
            assert any(
                f"{family} {kind} share count/min/mean/max/below half" in line for line in lines
            )


def test_norm_agreement_uses_reflected_ratings(runner: run.Run) -> None:
    states = finish(runner)
    judged = clients.obj(clients.obj(states["judge"]["payload"])["decision"])
    means = clients.obj(clients.obj(states["judge-check"]["payload"])["human"])
    from zoon.pilot import judge, outside

    expected = np.array(
        [
            8 - float(str(means[original]))
            if outside.action_reversed()[name]
            else float(str(means[original]))
            for name, original in outside.originals().items()
        ]
    )
    assert judged["norm_correlation_evaluative"] == judge.correlation(
        run.floats(judged["evaluative"]), expected
    )


def test_smoke_preflights_before_any_question(monkeypatch: pytest.MonkeyPatch) -> None:
    questions: list[int] = []

    class Decision(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            questions.append(1)
            return super().ask(request)

    class Second(clients.VertexClient):
        def preflight(self) -> None:
            raise clients.StepFailure("ZOON_VERTEX_PROJECT")

    monkeypatch.setattr(clients, "DecisionClient", lambda: Decision("decision"))
    monkeypatch.setattr(clients, "VertexClient", Second)
    with pytest.raises(clients.StepFailure, match="ZOON_VERTEX_PROJECT"):
        run.smoke(False)
    assert not questions


def test_share_summary_uses_exact_groups_and_omits_recognition(runner: run.Run) -> None:
    from zoon.pilot import prices

    state = runner.save("judge-check", {}, status="failed", reason="interrupted")
    for family in clients.budget.FAMILIES:
        transcript = runner.session.transcripts[family]
        version = prices.DECISION.model if family == "decision" else prices.SECOND.model
        for i, (question, share) in enumerate(
            [
                ("How appropriate is it?", 0.2),
                ("How appropriate is it?", 0.8),
                (prompts.STRICT[0], 0.5),
                (prompts.PREDICT, 0.9),
            ]
        ):
            request = prompts.Decision(str(i), question, prompts.declared(("Yes", "No")))
            transcript.append(
                request, clients.Response({"Yes": 0.5, "No": 0.5}, share, "", version, 1, 1)
            )
        transcript.append("recognition", clients.Response({}, 0.01, "unknown", version, 1, 1))
    lines = result.render(runner, {"judge-check": state})
    for family in clients.budget.FAMILIES:
        for kind, summary in (
            ("evaluative", "2/0.20000000/0.50000000/0.80000000/1"),
            ("strict", "1/0.50000000/0.50000000/0.50000000/0"),
            ("prompted", "1/0.90000000/0.90000000/0.90000000/0"),
        ):
            assert (
                f"[stub declared answers] {family} {kind} "
                f"share count/min/mean/max/below half = {summary}"
            ) in lines
