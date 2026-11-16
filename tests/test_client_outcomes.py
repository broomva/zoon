"""Request failures, unreadable replies and withdrawal have distinct outcomes."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.test_clients import REQUEST, vertex_shape
from tests.test_run import runner as runner
from zoon.pilot import clients, prices, run


def valid_transport(*args: object) -> tuple[int, dict[str, object]]:
    return 200, vertex_shape()


def empty_transport(*args: object) -> tuple[int, dict[str, object]]:
    return 200, {"candidates": []}


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    original = clients.retry

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return original(call, pause=lambda delay: None)

    monkeypatch.setattr(clients, "retry", retry)
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")


@pytest.mark.usefixtures("fast")
@pytest.mark.parametrize("family", clients.budget.FAMILIES)
@pytest.mark.parametrize(
    "statuses",
    [(404, 200), (410, 200), (404, 410, 404, 410), (404, 500, 410, 404), (500, 404, 410, 404)],
)
def test_withdrawal_requires_every_attempt(
    family: clients.Family, statuses: tuple[int, ...]
) -> None:
    calls: list[int] = []

    def status() -> int:
        value = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(value)
        return value

    class StatusError(Exception):
        def __init__(self, code: int) -> None:
            self.status_code = code

    class SDK:
        def Choice(self, **kwargs: object) -> object:
            return object()

        def TypeSafeClient(self, **kwargs: object) -> object:
            return self

        def system_one(self, *args: object) -> object:
            code = status()
            if code != 200:
                raise StatusError(code)
            return SimpleNamespace(
                model=prices.DECISION.model,
                usage=SimpleNamespace(input_tokens=1),
                answers={"q": SimpleNamespace(probabilities={"Yes": 0.5, "No": 0.5})},
            )

    def transport(*args: object) -> tuple[int, dict[str, object]]:
        return status(), vertex_shape()

    client = (
        clients.DecisionClient(cast(clients.DecisionSDK, SDK()))
        if family == "decision"
        else clients.VertexClient(lambda: "fixture-token", transport)
    )
    if 200 in statuses:
        client.ask(REQUEST)
        assert calls == list(statuses)
    else:
        expected = (
            (clients.TerminalStop if family == "decision" else clients.FamilyUnusable)
            if all(s in (404, 410) for s in statuses)
            else clients.StepFailure
        )
        with pytest.raises(expected):
            client.ask(REQUEST)
        assert calls == list(statuses)


@pytest.mark.usefixtures("fast")
def test_missing_decision_version_retries_without_recording(tmp_path: Path) -> None:
    calls: list[int] = []

    class SDK:
        def Choice(self, **kwargs: object) -> object:
            return object()

        def TypeSafeClient(self, **kwargs: object) -> object:
            return self

        def system_one(self, *args: object) -> object:
            calls.append(1)
            return SimpleNamespace(model=None, answers={}, usage=SimpleNamespace(input_tokens=1))

    session = clients.Session(tmp_path, [clients.DecisionClient(cast(clients.DecisionSDK, SDK()))])
    with pytest.raises(clients.StepFailure):
        session.ask_many("decision", [REQUEST])
    assert len(calls) == 4
    assert not session.transcripts["decision"].entries
    assert clients.Session(tmp_path, []).spent == 0


def test_transcript_rejects_non_string_version(tmp_path: Path) -> None:
    session = clients.Session(tmp_path, [clients.StubClient("decision")])
    session.ask_many("decision", [REQUEST])
    path = session.transcripts["decision"].path
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["response"]["version"] = None
    path.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(clients.GuardRefusal):
        clients.Session(tmp_path, [])


@pytest.mark.usefixtures("fast")
def test_metadata_fault_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "gemini")
    session = clients.Session(
        tmp_path,
        [clients.VertexClient(lambda: "fixture-token", valid_transport)],
    )
    with pytest.raises(clients.StepFailure):
        session.ask_many("second", [REQUEST])
    assert not session.transcripts["second"].entries
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    assert session.ask_many("second", [REQUEST])[0].share == pytest.approx(0.8)


@pytest.mark.usefixtures("fast")
def test_unreadable_reply_without_version_is_share_cause(tmp_path: Path) -> None:
    session = clients.Session(
        tmp_path,
        [clients.VertexClient(lambda: "fixture-token", empty_transport)],
    )
    with pytest.raises(clients.FamilyUnusable) as error:
        session.ask_many("second", [REQUEST])
    assert error.value.cause is clients.FamilyCause.SHARE


@pytest.mark.usefixtures("fast")
@pytest.mark.parametrize(
    "fault",
    [ValueError, KeyError, IndexError, OverflowError, TypeError, clients.StepFailure, RuntimeError],
)
def test_reader_program_fault_is_not_zero_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: type[Exception]
) -> None:
    def fail(*args: object) -> object:
        raise fault("diagnostic")

    monkeypatch.setattr(clients, "token_probabilities", fail)
    session = clients.Session(
        tmp_path,
        [clients.VertexClient(lambda: "fixture-token", valid_transport)],
    )
    with pytest.raises(clients.StepFailure):
        session.ask_many("second", [REQUEST])
    assert not session.transcripts["second"].entries


@pytest.mark.parametrize("cause", [clients.FamilyCause.WITHDRAWN, clients.FamilyCause.VERSION])
def test_recognition_exclusion_withholds_computed_numbers(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, cause: clients.FamilyCause
) -> None:
    runner.execute("judge-check")
    original = runner.request
    captured: list[run.State] = []
    save = runner.save

    def request(
        family: clients.Family,
        requests: clients.Sequence[clients.Request],
        *,
        recognition_only: bool = False,
    ) -> list[clients.Response]:
        if recognition_only:
            raise clients.FamilyUnusable(cause)
        return original(family, requests)

    def record(step: str, payload: run.State) -> run.State:
        captured.append(payload)
        return save(step, payload)

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "save", record)
    assert "completed" in runner.execute("judge")[0]
    assert clients.obj(captured[0]["second"]) == {"usable": False, "reason": cause.value}
    assert "recognition_error" not in captured[0]


def test_second_arm_record_failure_has_family_cause(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from tests.test_run import COMMIT

    for step in run.STEPS[:3]:
        runner.execute(step)
    original = run.arms.predict

    def broken(*args: object) -> object:
        predicted = cast(Callable[..., dict[str, list[run.models.Floats]]], original)(*args)
        predicted["second/zoon-evaluative"][0][:] = np.nan
        return predicted

    monkeypatch.setattr(run.arms, "predict", broken)
    assert "completed" in runner.execute("held-out", COMMIT)[0]
    unavailable = clients.obj(clients.obj(runner.state("held-out")["payload"])["unavailable"])
    assert unavailable["second"] == "second family complete-records failure"


@pytest.mark.usefixtures("fast")
def test_decision_connection_error_interrupts_without_withdrawal() -> None:
    attempts: list[int] = []

    class SDK:
        def Choice(self, **kwargs: object) -> object:
            return object()

        def TypeSafeClient(self, **kwargs: object) -> object:
            return self

        def system_one(self, *args: object) -> object:
            attempts.append(1)
            raise ConnectionError("connection interrupted")

    client = clients.DecisionClient(cast(clients.DecisionSDK, SDK()))
    with pytest.raises(clients.StepFailure):
        client.ask(REQUEST)
    assert len(attempts) == 4
