"""Calibration is resumable, deterministic, bounded and records closed outcomes."""

import json
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.interaction_run_support import repository
from zoon.interaction import decisions, prompted, run, score
from zoon.interaction.guard import checkpoint, encoded
from zoon.pilot import clients, prices


class InterruptedClient(clients.StubClient):
    def __init__(self, *, fail_at: int = 5, shuffled: bool = False) -> None:
        super().__init__("decision")
        self.count = 0
        self.fail_at, self.shuffled = fail_at, shuffled
        self.lock = threading.Lock()

    def ask(self, request: clients.Request) -> clients.Response:
        with self.lock:
            self.count += 1
            failed = self.count == self.fail_at
        if self.shuffled:
            time.sleep(int(clients.key(request)[:2], 16) / 100_000)
        if failed:
            raise clients.StepFailure("stub transport interrupted")
        return super().ask(request)


def transcript(path: Path) -> clients.Transcript:
    identity = json.loads(path.read_bytes().splitlines()[0])
    return clients.Transcript(path.parent, "decision", identity)


def test_stub_calibration_interrupt_resume_and_refuse_repeat(tmp_path: Path) -> None:
    root = repository(tmp_path / "repo")
    resumed_storage, plain_storage = tmp_path / "resumed", tmp_path / "plain"
    client = InterruptedClient(shuffled=True)
    interrupted = run.Run(root, stub=True, storage=resumed_storage, client=client)
    with pytest.raises(clients.StepFailure):
        interrupted.execute()
    path = resumed_storage / "transcripts" / "stub" / "decision.jsonl"
    before = path.read_bytes()
    assert len(transcript(path).entries) == clients.WINDOW - 1
    state = json.loads((resumed_storage / "run" / "stub" / "calibrate.json").read_bytes())
    assert state["status"] == "interrupted" and state["mode"] == "stub"
    final = run.Run(root, stub=True, storage=resumed_storage, client=client, workers=3).execute()
    assert path.read_bytes().startswith(before)
    uninterrupted = run.Run(root, stub=True, storage=plain_storage, workers=1).execute()
    assert final == uninterrupted
    assert final["status"] == "completed"
    assert final["payload"]["calls"] == 2 * prompted.SUBSAMPLE_COUNT
    assert len(final["payload"]["answers"]["keys"]) == prompted.SUBSAMPLE_COUNT
    assert len(final["payload"]["disguise_answers"]["keys"]) == prompted.SUBSAMPLE_COUNT
    assert client.count == 2 * prompted.SUBSAMPLE_COUNT + 1
    assert final["payload"]["versions"] == [prices.DECISION.model]
    assert final["payload"]["input_tokens"] == 12 * 1024
    assert final["payload"]["output_tokens"] == 1024
    assert final["payload"]["spending"] == str(prices.DECISION.dollars(12 * 1024, 1024))
    assert len(final["payload"]["recalibration"]) == 5
    before_state = (resumed_storage / "run" / "stub" / "calibrate.json").read_bytes()
    with pytest.raises(clients.GuardRefusal, match="completed"):
        run.Run(root, stub=True, storage=resumed_storage).execute()
    assert (resumed_storage / "run" / "stub" / "calibrate.json").read_bytes() == before_state
    assert not (root / "data" / "run").exists() and not (root / "data" / "transcripts").exists()


def test_worker_count_and_completion_order_do_not_change_results(tmp_path: Path) -> None:
    root = repository(tmp_path / "repo")
    first = run.Run(root, stub=True, storage=tmp_path / "one", workers=1).execute()
    second = run.Run(
        root,
        stub=True,
        storage=tmp_path / "many",
        client=InterruptedClient(fail_at=0, shuffled=True),
        workers=5,
    ).execute()
    assert first == second
    assert (tmp_path / "one" / "transcripts" / "stub" / "decision.jsonl").read_bytes() == (
        tmp_path / "many" / "transcripts" / "stub" / "decision.jsonl"
    ).read_bytes()


def test_disguise_uses_the_saved_fit_and_registered_scoring_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path / "repo")
    fitted: list[Any] = []

    def fit(pairs: Any) -> prompted.Recalibration:
        fitted.append(pairs)
        return prompted.Recalibration(20.0, (0.0, 0.0, 0.0, 0.0))

    monkeypatch.setattr(prompted, "fit", fit)
    runner = run.Run(root, stub=True)
    state = runner.execute()
    assert len(fitted) == 1 and len(fitted[0]) == 512
    assert runner.session is not None
    assert runner.session.cap == Decimal(17)
    mapping = runner.session.transcript.entries
    sample = prompted.subsample(runner.guard.verify_materials())
    plain: list[list[float]] = []
    disguised: list[list[float]] = []
    recalibration = prompted.Recalibration(
        state["payload"]["recalibration"][0], tuple(state["payload"]["recalibration"][1:])
    )
    bands: list[int] = []
    for decision in sample:
        view = prompted.training_view(decision.group, decision.seat, decision.round)
        bands.append(
            run.measure.band(
                decision.group.rounds[decision.round - 1][decision.seat],
                view.game.endowments[decision.seat],
            )
        )
        for label, multiplier, destination in (
            ("original", 1, plain),
            ("disguise", prompted.DISGUISE, disguised),
        ):
            request = prompted.request(view, multiplier)
            response = mapping[
                decisions.request_key((label + ":" + decision.digest, request))
            ].response
            destination.append([response.probabilities[a] for a in prompted.ANSWERS])
    assert [band for _, band in fitted[0]] == bands
    np.testing.assert_array_equal(np.array([p for p, _ in fitted[0]]), np.array(plain))
    p = recalibration.apply(np.array(plain))
    q = recalibration.apply(np.array(disguised))
    logs = score.log_scores(q, np.array(bands)) - score.log_scores(p, np.array(bands))
    p, q = np.maximum(p, score.SCORING_FLOOR), np.maximum(q, score.SCORING_FLOOR)
    p, q = p / p.sum(axis=1, keepdims=True), q / q.sum(axis=1, keepdims=True)
    assert state["payload"]["disguise"] == {
        "mean_log_probability_change": float(logs.mean()),
        "mean_total_variation": float(np.mean(np.sum(0.5 * np.abs(q - p), axis=1))),
    }


@pytest.mark.parametrize("outcome", ["WITHDRAWN", "VERSION", "CAP", "FIT"])
def test_run_ending_faults_are_terminal_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    root = repository(tmp_path / "repo")

    class StoppingClient(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if outcome == "WITHDRAWN":
                raise clients.TerminalStop(clients.StopReason.WITHDRAWN)
            response = super().ask(request)
            return (
                replace(response, version="another-version") if outcome == "VERSION" else response
            )

    if outcome == "FIT":

        def failed_fit(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("fit did not converge")

        monkeypatch.setattr(prompted, "fit", failed_fit)
    runner = run.Run(
        root,
        stub=True,
        client=StoppingClient("decision"),
        cap=Decimal(0) if outcome == "CAP" else None,
    )
    state = runner.execute()
    assert state["status"] == "terminal"
    assert state["payload"]["outcome"] == {
        "H2": "no verdict",
        "H3": "no verdict",
        "reason": outcome,
    }
    if outcome == "VERSION":
        assert state["payload"]["calls"] > 0
        assert state["payload"]["versions"] == ["another-version"]
        assert Decimal(state["payload"]["spending"]) > 0
    with pytest.raises(clients.GuardRefusal, match="terminal"):
        runner.execute()
    with runner.guard.lock():
        runner.guard.before("report")


def test_disguise_interruption_keeps_the_fit_and_never_refits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path / "repo")
    runner = run.Run(root, stub=True, client=InterruptedClient(fail_at=520))
    with pytest.raises(clients.StepFailure):
        runner.execute()
    state = json.loads((runner.guard.directory / "calibrate.json").read_bytes())
    numbers = state["payload"]["recalibration"]

    def no_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("saved fit must be reused")

    monkeypatch.setattr(prompted, "fit", no_fit)
    completed = run.Run(root, stub=True).execute()
    assert completed["status"] == "completed"
    assert "outcome" not in completed["payload"]
    assert completed["payload"]["recalibration"] == numbers


def test_transcript_identity_damage_and_checkpoint_are_refused(tmp_path: Path) -> None:
    root = repository(tmp_path / "repo")
    runner = run.Run(root, stub=True, client=InterruptedClient())
    with pytest.raises(clients.StepFailure):
        runner.execute()
    path = runner.guard.transcript_directory / "decision.jsonl"
    original = path.read_bytes()
    lines = original.splitlines()
    identity = json.loads(lines[0])
    for field, value in (("mode", "live"), ("registered_commit", "b" * 40)):
        changed = {**identity, field: value}
        path.write_bytes(
            encoded(changed).replace(b"\n", b"") + b"\n" + b"\n".join(lines[1:]) + b"\n"
        )
        with pytest.raises(clients.GuardRefusal, match="identity"):
            runner.execute()
    path.write_bytes(original[:-2])
    with pytest.raises(clients.GuardRefusal, match="incomplete"):
        runner.execute()
    assert path.read_bytes() == original[:-2]
    entry = json.loads(lines[1])
    entry["response"]["input_tokens"] += 1
    path.write_bytes(
        lines[0] + b"\n" + json.dumps(entry).encode() + b"\n" + b"\n".join(lines[2:]) + b"\n"
    )
    with pytest.raises(clients.GuardRefusal, match="checkpoint"):
        runner.execute()


def test_budget_admission_includes_reservations_and_allows_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = prompted.fixtures()["2-AI-type-0-round-1.json"]
    estimate = clients.estimate("decision", request)
    client = InterruptedClient(fail_at=0)
    session = decisions.Session(
        tmp_path, client, 4, estimate, {"mode": "stub", "registered_commit": "a" * 40}
    )
    # A second call cannot be admitted in the same window. Its reported cost is
    # above the estimate so no next window can send it either.
    original = client.ask

    def costly(request: clients.Request) -> clients.Response:
        response = original(request)
        return replace(response, input_tokens=10_000_000)

    monkeypatch.setattr(client, "ask", costly)
    with pytest.raises(clients.TerminalStop) as stopped:
        session.ask_many([("one", request), ("two", request)])
    assert stopped.value.reason == clients.StopReason.CAP
    assert client.count == 1 and len(session.transcript.entries) == 1
    assert session.transcript.spent == prices.DECISION.dollars(10_000_000, 1)
    assert checkpoint(session.transcript)["digest"]


def test_retry_is_the_pilots_bounded_retry_with_registered_pauses() -> None:
    attempts: list[int] = []
    pauses: list[float] = []

    def failed() -> clients.Response:
        attempts.append(1)
        raise clients.TransientError

    with pytest.raises(clients.StepFailure):
        clients.retry(failed, pause=pauses.append)
    assert len(attempts) == 4 and pauses == [1, 2, 4]


def test_cli_accepts_only_implemented_steps_and_prints_no_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for step in ("closed-loop", "recall", "held-out", "report", "reset"):
        assert run.main([step, "--stub"]) == 1
        error = capsys.readouterr().err
        assert error.startswith("GuardRefusal: ") and "invalid choice" in error

    def failed(*args: Any, **kwargs: Any) -> Any:
        credential_in_frame = "not for output"
        assert credential_in_frame
        raise clients.StepFailure("bounded request failed")

    monkeypatch.setattr(run.Run, "execute", failed)
    assert run.main(["calibrate", "--stub"]) == 1
    output = capsys.readouterr()
    assert output.err == "StepFailure: bounded request failed\n"
    assert output.out == ""


@pytest.mark.parametrize("cap_calls,usable", [(1, False), (2, True), (4, False)])
def test_each_paid_retry_is_recorded_and_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap_calls: int, usable: bool
) -> None:
    from collections.abc import Callable
    from types import SimpleNamespace

    from zoon.pilot import budget as pilot_budget

    request = prompted.fixtures()["2-AI-type-0-round-1.json"]
    calls: list[int] = []
    pauses: list[float] = []
    bounded = clients.retry

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return bounded(call, pause=pauses.append)

    monkeypatch.setattr(clients, "retry", retry)

    class SDK:
        def Choice(self, **kwargs: Any) -> None:
            pass

        def TypeSafeClient(self, **kwargs: Any) -> Any:
            return self

        def system_one(self, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            probabilities = dict.fromkeys(
                prompted.ANSWERS, 0.2 if usable and len(calls) == 2 else 0.24
            )
            return SimpleNamespace(
                model=prices.DECISION.model,
                usage=SimpleNamespace(
                    input_tokens=pilot_budget.input_tokens(request, "decision"), output_tokens=1
                ),
                answers={"q": SimpleNamespace(probabilities=probabilities)},
            )

    estimate = clients.estimate("decision", request)
    session = decisions.Session(
        tmp_path,
        decisions.LiveClient(SDK()),
        4,
        cap_calls * estimate,
        {"mode": "stub", "registered_commit": "a" * 40},
    )
    if usable:
        responses = session.ask_many([("decision", request)])
        assert responses[0].probabilities == dict.fromkeys(prompted.ANSWERS, 0.2)
        assert pauses == [1]
    else:
        with pytest.raises(clients.TerminalStop if cap_calls == 1 else clients.StepFailure):
            session.ask_many([("decision", request)])
        assert pauses == ([1] if cap_calls == 1 else [1, 2, 4])
    assert len(calls) == cap_calls
    assert len(session.transcript.entries) == cap_calls
    assert session.transcript.spent == cap_calls * estimate
    reloaded = clients.Transcript(tmp_path, "decision", session.transcript.identity)
    assert checkpoint(reloaded) == checkpoint(session.transcript)
    assert next(iter(reloaded.entries.values())).response.returned_probabilities == dict.fromkeys(
        prompted.ANSWERS, 0.24
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing-version",
        "boolean-input",
        "boolean-output",
        "missing-input",
        "withdrawn",
        "mixed-withdrawal",
        "version",
    ],
)
def test_injected_decision_transport_obeys_metadata_and_retry_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from collections.abc import Callable
    from types import SimpleNamespace

    calls: list[int] = []
    pauses: list[float] = []
    bounded = clients.retry

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return bounded(call, pause=pauses.append)

    monkeypatch.setattr(clients, "retry", retry)

    class Missing(Exception):
        status_code = 404

    class SDK:
        def Choice(self, **kwargs: Any) -> None:
            pass

        def TypeSafeClient(self, **kwargs: Any) -> Any:
            return self

        def system_one(self, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            if fault in ("withdrawn", "mixed-withdrawal"):
                if fault == "mixed-withdrawal" and len(calls) == 1:
                    raise clients.TransientError
                raise Missing
            usage = SimpleNamespace(
                input_tokens=True if fault == "boolean-input" else 12,
                output_tokens=True if fault == "boolean-output" else 1,
            )
            if fault == "missing-input":
                del usage.input_tokens
            return SimpleNamespace(
                model=""
                if fault == "missing-version"
                else "another-version"
                if fault == "version"
                else prices.DECISION.model,
                usage=usage,
                answers={"q": SimpleNamespace(probabilities=dict.fromkeys(prompted.ANSWERS, 0.2))},
            )

    session = decisions.Session(
        tmp_path,
        decisions.LiveClient(SDK()),
        3,
        Decimal(17),
        {"mode": "stub", "registered_commit": "a" * 40},
    )
    with pytest.raises(
        clients.TerminalStop if fault in ("withdrawn", "version") else clients.StepFailure
    ):
        session.ask_many([("decision", prompted.fixtures()["2-AI-type-0-round-1.json"])])
    assert len(calls) == (1 if fault == "version" else 4)
    assert pauses == ([] if fault == "version" else [1, 2, 4])
    assert len(session.transcript.entries) == (1 if fault == "version" else 0)


def test_nonfinite_summary_interrupts_without_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path / "repo")

    def nonfinite(probabilities: Any, chosen: Any) -> Any:
        return np.full(len(chosen), np.nan)

    monkeypatch.setattr(score, "log_scores", nonfinite)
    runner = run.Run(root, stub=True)
    with pytest.raises(clients.StepFailure, match="non-finite"):
        runner.execute()
    state = json.loads((runner.guard.directory / "calibrate.json").read_bytes())
    assert state["status"] == "interrupted"
    assert "outcome" not in state["payload"]
    assert state["payload"]["calls"] == 1024


def test_record_fault_is_a_terminal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path / "repo")

    def unreadable(*args: Any, **kwargs: Any) -> Any:
        raise run.data.RecordFault("unreadable group")

    monkeypatch.setattr(run.Run, "calibrate", unreadable)
    state = run.Run(root, stub=True).execute()
    assert state["status"] == "terminal"
    assert state["payload"]["outcome"] == {
        "H2": "no verdict",
        "H3": "no verdict",
        "reason": "RECORDS",
    }


def test_incomplete_calibration_sample_interrupts_before_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path / "repo")
    original = prompted.subsample

    def incomplete(groups: Sequence[run.data.Group]) -> list[prompted.TrainingDecision]:
        return original(groups)[:-1]

    monkeypatch.setattr(prompted, "subsample", incomplete)
    runner = run.Run(root, stub=True)
    with pytest.raises(clients.StepFailure, match="exactly 512"):
        runner.execute()
    state = json.loads((runner.guard.directory / "calibrate.json").read_bytes())
    assert state["status"] == "interrupted" and state["payload"]["calls"] == 0


def test_cli_interrupt_prints_only_the_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(run.Run, "execute", interrupted)
    try:
        status = run.main(["calibrate", "--stub"])
    except KeyboardInterrupt:
        pytest.fail("command propagated an interrupt traceback")
    assert status == 1
    assert capsys.readouterr().err == "KeyboardInterrupt: \n"
