"""Resumption and admission use the same boundaries for every command."""

import http.client
import json
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from tests.test_clients import REQUEST, vertex_shape
from tests.test_run import COMMIT
from tests.test_run import runner as runner
from zoon.pilot import clients, run


@pytest.mark.parametrize(
    "step,method",
    list(
        zip(run.STEPS[:4], ("_judge_check", "_judge_all", "_calibrate", "_held_out"), strict=True)
    ),
)
@pytest.mark.parametrize(
    "fault", [ValueError, KeyError, TypeError, RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_each_body_fault_is_resumable_and_blocks_report(
    runner: run.Run,
    monkeypatch: pytest.MonkeyPatch,
    step: str,
    method: str,
    fault: type[BaseException],
) -> None:
    for previous in run.STEPS[: run.STEPS.index(step)]:
        runner.save(previous, {})

    def fail(*args: object) -> run.State:
        raise fault("private diagnostic")

    def complete(*args: object) -> run.State:
        return runner.save(step, {})

    monkeypatch.setattr(runner, method, fail)
    assert "failed" in runner.execute(step, COMMIT)[0]
    assert runner.state(step)["reason"] == fault.__name__
    with pytest.raises(clients.GuardRefusal, match="repeat step"):
        runner.execute("report")
    monkeypatch.setattr(runner, method, complete)
    assert "completed" in runner.execute(step, COMMIT)[0]


@pytest.mark.parametrize("outcome", [clients.TerminalStop, clients.FamilyUnusable])
def test_closed_outcomes_reject_unregistered_reasons(outcome: type[Exception]) -> None:
    with pytest.raises(TypeError):
        cast(Callable[[object], Exception], outcome)("unregistered reason")
    assert not issubclass(outcome, ValueError)
    for other in (
        clients.GuardRefusal,
        clients.StepFailure,
        clients.FamilyUnusable,
        clients.TerminalStop,
    ):
        if outcome is not other:
            assert not issubclass(outcome, other)


@pytest.mark.parametrize("status", ["completed", "stopped"])
def test_recorded_outcome_still_requires_guard(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    runner.save("judge-check", {}, status=status)
    before = (runner.directory / "judge-check.json").read_bytes()

    def refuse(*args: object) -> None:
        raise clients.GuardRefusal("changed inputs")

    monkeypatch.setattr(runner, "guard", refuse)
    runner.session.cap = Decimal("-1")
    with pytest.raises(clients.GuardRefusal, match="changed inputs"):
        runner.execute("judge-check")
    assert (runner.directory / "judge-check.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["reset", "incomplete", "json"])
@pytest.mark.parametrize("recover", [False, True])
def test_second_transport_faults_retry_without_family_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, recover: bool
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    original = clients.retry
    attempts: list[int] = []
    delays: list[float] = []

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return original(call, pause=delays.append)

    class Reply:
        status = 200

        def __enter__(self) -> "Reply":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            attempts.append(1)
            if not recover or len(attempts) == 1:
                if fault == "reset":
                    raise ConnectionResetError("private diagnostic")
                if fault == "incomplete":
                    raise http.client.IncompleteRead(b"{", 10)
                return b'{"candidates":'
            return json.dumps(vertex_shape()).encode()

    def urlopen(*args: object, **kwargs: object) -> Reply:
        return Reply()

    monkeypatch.setattr(clients, "retry", retry)
    monkeypatch.setattr(clients.urllib.request, "urlopen", urlopen)
    session = clients.Session(tmp_path, [clients.VertexClient(lambda: "fixture-token")])
    if recover:
        assert session.ask_many("second", [REQUEST])[0].share == pytest.approx(0.8)
        assert len(attempts) == 2 and delays == [1]
        session.check_family("second")
    else:
        with pytest.raises(clients.StepFailure):
            session.ask_many("second", [REQUEST])
        assert len(attempts) == 4 and delays == [1, 2, 4]
        assert not session.transcripts["second"].entries


def test_cap_admission_reserves_every_in_flight_request(tmp_path: Path) -> None:
    requests = [replace(REQUEST, state=f"request {i:02d}") for i in range(clients.WINDOW)]
    estimate = clients.estimate("decision", requests[0])
    assert all(clients.estimate("decision", r) == estimate for r in requests)
    recorded = {
        clients.key(r): replace(
            clients.StubClient("decision").ask(r),
            input_tokens=clients.budget.input_tokens(r, "decision"),
            output_tokens=1,
        )
        for r in requests
    }
    session = clients.Session(
        tmp_path, [clients.StubClient("decision", recorded)], cap=3 * estimate
    )
    with pytest.raises(clients.TerminalStop, match="cap reached"):
        session.ask_many("decision", requests)
    assert len(session.transcripts["decision"].entries) == 3
    assert session.spent == session.cap


def test_outcome_types_are_unrelated() -> None:
    for outcome in (
        clients.GuardRefusal,
        clients.StepFailure,
        clients.FamilyUnusable,
        clients.TerminalStop,
    ):
        assert outcome.__bases__ == (Exception,)


@pytest.mark.parametrize(
    "family,recognition", [("decision", False), ("second", False), ("second", True)]
)
def test_cap_admission_uses_each_unequal_estimate(
    tmp_path: Path, family: clients.Family, recognition: bool
) -> None:
    # Select three canonical keys in order with a small first estimate and larger next ones.
    candidates: list[clients.Request] = [
        ("recognition " + str(i) + " " + "x" * length)
        if recognition
        else replace(REQUEST, state=str(i) + " " + "x" * length)
        for length in (1, 3000, 6000)
        for i in range(20)
    ]
    ordered = sorted(candidates, key=lambda r: clients.key(clients.wire(r, family)))
    first = next(
        r
        for r in ordered
        if clients.estimate(family, r) == min(clients.estimate(family, c) for c in candidates)
    )
    later = [
        r
        for r in ordered[ordered.index(first) + 1 :]
        if clients.estimate(family, r) > 3 * clients.estimate(family, first)
    ]
    assert len(later) >= 2
    requests = [first, *later[:2]]
    estimates = [clients.estimate(family, r) for r in requests]
    assert estimates[0] < estimates[1] and estimates[0] < estimates[2]
    called: list[str] = []

    class Metered(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            called.append(clients.key(clients.wire(request, family)))
            return replace(
                super().ask(request),
                input_tokens=clients.budget.input_tokens(clients.wire(request, family), family),
                output_tokens=40 if recognition else 1,
            )

    session = clients.Session(tmp_path, [Metered(family)], workers=1)
    # Existing spending includes both families and a recognition request.
    session.transcripts["decision"].append(REQUEST, clients.StubClient("decision").ask(REQUEST))
    session.transcripts["second"].append(
        "earlier recognition",
        clients.Response({}, None, "unknown", clients.prices.SECOND.model, 100, 40),
    )
    before = session.spent
    session.cap = before + estimates[0] + estimates[1]
    with pytest.raises(clients.TerminalStop, match="cap reached"):
        session.ask_many(family, requests, recognition=recognition)
    assert called == [clients.key(clients.wire(r, family)) for r in requests[:2]]
    assert session.spent == session.cap
