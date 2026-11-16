"""Observed outcomes for faults at requests and inside each calling step."""

from collections.abc import Callable

import pytest

from tests.test_run import COMMIT, finish
from tests.test_run import runner as runner
from tests.tracked import copy_tracked
from zoon.pilot import clients, fetch, run

# The expected outcome is explicit for both families, independent of exception ancestry.
FAULTS: list[tuple[str, Callable[[], BaseException], str, str]] = [
    ("request", clients.StepFailure, "failed", "failed"),
    ("refusal-in-body", clients.GuardRefusal, "failed", "failed"),
    ("value", ValueError, "failed", "failed"),
    ("key", KeyError, "failed", "failed"),
    ("index", IndexError, "failed", "failed"),
    ("overflow", OverflowError, "failed", "failed"),
    ("type", TypeError, "failed", "failed"),
    ("runtime", RuntimeError, "failed", "failed"),
    ("io", OSError, "failed", "failed"),
    ("interrupt", KeyboardInterrupt, "failed", "failed"),
    ("exit", SystemExit, "failed", "failed"),
    ("record-in-body", fetch.RecordFault, "failed", "failed"),
    ("share", lambda: clients.FamilyUnusable(clients.FamilyCause.SHARE), "failed", "excluded"),
    ("cap", lambda: clients.TerminalStop(clients.StopReason.CAP), "stopped", "stopped"),
]


@pytest.fixture(scope="module")
def payloads(tmp_path_factory: pytest.TempPathFactory) -> dict[str, run.State]:
    root = tmp_path_factory.mktemp("baseline")
    copy_tracked(root)
    with pytest.MonkeyPatch.context() as patch:

        def git(*args: object) -> str:
            return COMMIT

        def sealed(*args: object) -> None:
            raise AssertionError("real records are unreachable")

        patch.setattr(run, "git", git)
        patch.setattr(fetch, "load_registered_held_out", sealed)
        states = finish(run.Run(root, synthetic=True))
        return {step: clients.obj(state["payload"]) for step, state in states.items()}


@pytest.mark.parametrize("step", run.STEPS[:4])
@pytest.mark.parametrize("family", clients.budget.FAMILIES)
@pytest.mark.parametrize("layer", ["request", "body"])
@pytest.mark.parametrize("label,fault,decision,second", FAULTS, ids=[f[0] for f in FAULTS])
def test_calling_step_fault_table(
    runner: run.Run,
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, run.State],
    step: str,
    family: clients.Family,
    layer: str,
    label: str,
    fault: Callable[[], BaseException],
    decision: str,
    second: str,
) -> None:
    for previous in run.STEPS[: run.STEPS.index(step)]:
        runner.save(previous, payloads[previous])
    hit: list[str] = []
    active: list[clients.Family] = []

    def fail() -> None:
        hit.append(label)
        raise fault()

    if layer == "request":
        original_client = runner.session.clients[family]

        class Interrupted(clients.StubClient):
            def ask(self, request: clients.Request) -> clients.Response:
                if not isinstance(request, str):
                    fail()
                return original_client.ask(request)

        runner.session.clients[family] = Interrupted(family)
    elif step in ("calibrate", "held-out"):
        original_collect = runner.collect_prompted

        def collect(
            f: clients.Family,
            people: clients.Sequence[run.cohorts.Participant],
            rotation: run.cohorts.Rotation,
            training: clients.Sequence[run.cohorts.Participant],
            tasks: dict[str, run.materials.Task],
            *,
            held: bool,
        ) -> run.models.Floats:
            if f == family:
                fail()
            # Other-family computation is irrelevant to the injected fault.
            return original_collect(f, people, rotation, training, tasks, held=held)

        monkeypatch.setattr(runner, "collect_prompted", collect)
    else:
        original_request = runner.request

        def request(
            f: clients.Family,
            requests: clients.Sequence[clients.Request],
            *,
            recognition_only: bool = False,
        ) -> list[clients.Response]:
            active[:] = [f]
            return original_request(f, requests, recognition_only=recognition_only)

        monkeypatch.setattr(runner, "request", request)
        name = "calibration" if step == "judge-check" else "evaluative"
        original_body = getattr(run.judge, name)

        def body(*args: object) -> object:
            if active == [family]:
                fail()
            return original_body(*args)

        monkeypatch.setattr(run.judge, name, body)
    runner.execute(step, COMMIT)
    assert hit, "the injected fault was not reached"
    state = runner.state(step)
    expected = decision if family == "decision" else second
    assert state["status"] == ("completed" if expected == "excluded" else expected)
    if expected == "excluded":
        payload = clients.obj(state["payload"])
        if step == "held-out":
            assert clients.obj(payload["unavailable"])["second"] == clients.FamilyCause.SHARE.value
            assert not any(n.startswith("second/") for n in clients.obj(payload["logs"]))
        else:
            family_state = clients.obj(payload["second"])
            assert family_state.get("passed", family_state.get("usable")) is False
            assert family_state["reason"] == clients.FamilyCause.SHARE.value
    elif expected == "failed":
        assert state["reason"] in (type(fault()).__name__, "StepFailure")


@pytest.mark.parametrize("fault", [clients.StepFailure, ValueError, OSError, RuntimeError])
def test_recognition_fault_interrupts(
    runner: run.Run,
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, run.State],
    fault: type[Exception],
) -> None:
    runner.save("judge-check", payloads["judge-check"])
    original = runner.request

    def request(
        family: clients.Family,
        requests: clients.Sequence[clients.Request],
        *,
        recognition_only: bool = False,
    ) -> list[clients.Response]:
        if recognition_only:
            raise fault()
        return original(family, requests)

    monkeypatch.setattr(runner, "request", request)
    assert "failed" in runner.execute("judge")[0]


def test_final_recognition_gate_fault_interrupts(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, payloads: dict[str, run.State]
) -> None:
    runner.save("judge-check", payloads["judge-check"])
    original = runner.session.check_family
    reached: list[bool] = []

    def check(family: clients.Family) -> None:
        replies = [
            e
            for e in runner.session.transcripts["second"].entries.values()
            if isinstance(e.request, str)
        ]
        if family == "second" and len(replies) == 13:
            reached.append(True)
            raise clients.StepFailure()
        original(family)

    monkeypatch.setattr(runner.session, "check_family", check)
    assert "failed" in runner.execute("judge")[0]
    assert reached
