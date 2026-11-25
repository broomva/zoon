"""Interrupted attempts retain receipts and preserve the earliest chat cause."""

import os
from pathlib import Path
from typing import Any

import pytest

from tests.interaction_run_support import repository
from tests.test_interaction_endings import invalid_sync
from tests.test_interaction_step_faults import ChatFault, Factory, band_logs, state
from zoon.interaction import chat, guard, run, score
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)


def test_chat_transport_resume_matches_uninterrupted_and_skips_finished_arms(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = runs("plain", prepared="calibrated")
    expected = plain.execute("closed-loop")
    client = ChatFault("transport", at=45)
    resumed = runs("resumed", prepared="calibrated", chat_client=client)
    with pytest.raises(chat.TransportInterruption):
        resumed.execute("closed-loop")
    interrupted = state(resumed)
    assert interrupted["status"] == "interrupted"
    assert interrupted["payload"]["chat"]["answered_decisions"] == 44
    before = (resumed.guard.transcript_directory / "chat.jsonl").read_bytes()

    def no_simulate(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("finished arms must only be reread")

    monkeypatch.setattr(score, "simulate", no_simulate)
    original_ask = chat.StubClient.ask
    requested: list[str] = []

    def ask(client: chat.StubClient, text: str) -> chat.Response:
        requested.append(text)
        return original_ask(client, text)

    monkeypatch.setattr(chat.StubClient, "ask", ask)
    # New process, fresh stub: its random stream must be recovered from receipts.
    fresh = run.Run(
        resumed.guard.root, stub=True, storage=resumed.guard.storage, simulation_groups=1
    )
    completed = fresh.execute("closed-loop")
    assert {k: v for k, v in completed.items() if k not in ("digest", "payload_digest")} == {
        k: v for k, v in expected.items() if k not in ("digest", "payload_digest")
    }
    assert len(requested) == 440 - 44
    assert (fresh.guard.transcript_directory / "chat.jsonl").read_bytes().startswith(before)
    assert band_logs(fresh) == band_logs(plain)


def test_recall_transport_resume_equals_uninterrupted(runs: Factory) -> None:
    plain = runs("plain", chat_client=ChatFault("replies", at=1))
    plain.execute("closed-loop")
    resumed = runs("resumed", chat_client=ChatFault("replies", at=1))
    resumed.execute("closed-loop")
    expected = run.Run(plain.guard.root, stub=True, storage=plain.guard.storage).execute("recall")
    interrupted = run.Run(
        resumed.guard.root,
        stub=True,
        storage=resumed.guard.storage,
        chat_client=ChatFault("transport", at=3),
    )
    with pytest.raises(chat.TransportInterruption):
        interrupted.execute("recall")
    final = run.Run(resumed.guard.root, stub=True, storage=resumed.guard.storage).execute("recall")
    assert {k: v for k, v in final.items() if k not in ("digest", "payload_digest")} == {
        k: v for k, v in expected.items() if k not in ("digest", "payload_digest")
    }


@pytest.mark.parametrize(
    "fault",
    [
        "chat",
        "decision",
        "chat-storage",
        "decision-storage",
        "state",
        "coded-log",
        "incomplete",
        "replaced",
    ],
)
@pytest.mark.parametrize("handler_fails", [False, True])
def test_failure_handler_preserves_last_success_and_recovers(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, fault: str, handler_fails: bool
) -> None:
    from zoon.interaction import decisions, simulation

    runner = runs(prepared="calibrated")
    original_save, original_write = runner.steps.save, simulation.write
    original_ask, original_decision = chat.StubClient.ask, clients.StubClient.ask
    original_append = chat.Transcript.append
    previous: dict[str, Any] = {}
    fired = False
    handler_saves = 0

    def fail() -> None:
        nonlocal fired, previous
        previous = state(runner)
        fired = True
        if fault == "incomplete":
            with (runner.guard.transcript_directory / "chat.jsonl").open("ab") as handle:
                handle.write(b'{"partial":')
        if fault == "chat":
            raise chat.TransportInterruption("chat unavailable")
        if fault == "decision":
            raise clients.StepFailure("decision unavailable")
        if fault == "decision-storage":
            raise decisions.StorageFault()
        raise OSError("storage unavailable")

    def ask(client: chat.StubClient, text: str) -> chat.Response:
        if not fired:
            fail()
        return original_ask(client, text)

    def decision(client: clients.StubClient, request: clients.Request) -> clients.Response:
        if not fired:
            fail()
        return original_decision(client, request)

    def append(transcript: chat.Transcript, *args: Any, **kwargs: Any) -> Any:
        if not fired:
            fail()
        return original_append(transcript, *args, **kwargs)

    def write(path: Path, raw: bytes) -> None:
        if not fired:
            fail()
        original_write(path, raw)

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        nonlocal previous, fired, handler_saves
        if status == "interrupted":
            handler_saves += 1
        if status == "interrupted" and handler_fails:
            with monkeypatch.context() as inner:
                inner.setattr(os, "fsync", invalid_sync)
                return original_save(status, reason)
        if status == "running" and "simulations" in runner.steps.payload and not fired:
            if fault == "state":
                fail()
            if fault == "replaced":
                result = original_save(status, reason)
                previous, fired = result, True
                raise guard.StateReplacedError("directory sync unavailable")
        return original_save(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        if fault in ("chat", "incomplete"):
            patch.setattr(chat.StubClient, "ask", ask)
        if fault in ("decision", "decision-storage"):
            patch.setattr(clients.StubClient, "ask", decision)
        if fault == "chat-storage":
            patch.setattr(chat.Transcript, "append", append)
        if fault == "coded-log":
            patch.setattr(simulation, "write", write)
        with pytest.raises(
            (
                OSError,
                clients.StepFailure,
                clients.GuardRefusal,
                chat.GuardRefusal,
                chat.Interruption,
            )
        ):
            runner.execute("closed-loop")
    assert fired
    assert handler_saves == (0 if fault == "replaced" else 1)
    if fault != "replaced":
        assert runner.steps.session is None
    disk = state(runner)
    if handler_fails or fault in ("incomplete", "replaced"):
        assert disk == previous
    else:
        assert disk["status"] == "interrupted"
        assert disk["reason"] == {
            "chat": "TransportInterruption",
            "decision": "StepFailure",
            "decision-storage": "StorageFault",
            "chat-storage": "OSError",
        }.get(fault, "OSError")
    if fault == "incomplete":
        with pytest.raises((clients.GuardRefusal, chat.GuardRefusal), match="cut the file back to"):
            runner.execute("closed-loop")
        path = runner.guard.transcript_directory / "chat.jsonl"
        path.write_bytes(path.read_bytes().split(b'{"partial":')[0])
    actual = runner.execute("closed-loop")
    expected = runs("plain").execute("closed-loop")
    assert {k: v for k, v in actual.items() if k not in ("digest", "payload_digest")} == {
        k: v for k, v in expected.items() if k not in ("digest", "payload_digest")
    }


def test_step_commands_use_stub_storage_and_print_recognition_in_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace
    from typing import Any

    root = repository(tmp_path / "repo")
    storage = tmp_path / "storage"
    original = run.Run

    class Replies(chat.StubClient):
        def ask(self, text: str) -> chat.Response:
            response = super().ask(text)
            if "recognise this experiment" in text:
                return replace(response, text="" if "two players" in text else "WANG\nFull reply")
            return replace(response, text="unusable")

    class LocalRun(original):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(root, simulation_groups=1, chat_client=Replies(), **kwargs)

    monkeypatch.setenv("ZOON_INTERACTION_DATA", str(storage))
    monkeypatch.setattr(run, "Run", LocalRun)
    for command in ("calibrate", "closed-loop", "recall"):
        assert run.main([command, "--stub"]) == 0
        output = capsys.readouterr()
        assert output.err == ""
        assert f"[interaction stub] {command} = completed" in output.out
        if command == "closed-loop":
            assert "[simulation] logs = 24" in output.out
        if command == "recall":
            assert "[recall stopped by] None" in output.out
            assert output.out.count("[recall ") == 9
            assert "WANG\nFull reply" in output.out


@pytest.mark.parametrize("cause", ["REPLAY", "WITHDRAWN"])
@pytest.mark.parametrize("again", [False, True])
def test_unrecorded_ending_is_observed_again(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, cause: str, again: bool
) -> None:
    runner = runs(chat_client=ChatFault("withdrawn", at=1))
    original = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if runner.steps.payload.get("chat_endings"):
            raise OSError("ending storage unavailable")
        return original(status, reason)

    def replay(*args: Any, **kwargs: Any) -> Any:
        raise chat.Unusable(chat.Cause.REPLAY)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        if cause == "REPLAY":
            runner.steps.chat_client = None
            patch.setattr(chat, "replay", replay)
        with pytest.raises(OSError):
            runner.execute("closed-loop")
    assert not state(runner)["payload"].get("chat_endings")
    runner.steps.chat_client = ChatFault("withdrawn", at=1) if again else None
    with monkeypatch.context() as patch:
        if cause == "REPLAY":
            patch.setattr(chat, "replay", replay)
        done = runner.execute("closed-loop")
    assert done["payload"]["chat"]["cause"] == (cause if again or cause == "REPLAY" else None)


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("handler_fails", [False, True])
def test_withdrawal_ending_save_boundary(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str, handler_fails: bool
) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    runner.steps.chat_client = ChatFault("withdrawn", at=1)
    original = runner.steps.save
    failed = False

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        nonlocal failed
        events = runner.steps.payload.get("chat_endings", [])
        if (
            events
            and events[-1]["step"] == step
            and (
                (status == "running" and not failed) or (status == "interrupted" and handler_fails)
            )
        ):
            failed = True
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", invalid_sync)
                return original(status, reason)
        return original(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute(step)
    saved = state(runner, step)
    assert saved["status"] == ("running" if handler_fails else "interrupted")
    assert bool(saved["payload"].get("chat_endings")) == (not handler_fails)
    runner.steps.chat_client = None
    done = runner.execute(step)
    assert done["status"] == "completed"
    assert bool(done["payload"].get("chat_endings")) == (not handler_fails)
    if not handler_fails:
        assert done["payload"]["chat"]["cause"] == "WITHDRAWN"
        assert done["payload"]["chat_endings"] == saved["payload"]["chat_endings"]


def test_decision_transport_interrupts_without_chat_ending(runs: Factory) -> None:
    class Decision(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            raise clients.StepFailure("decision transport unavailable")

    runner = runs(client=Decision("decision"))
    with pytest.raises(clients.StepFailure):
        runner.execute("closed-loop")
    saved = state(runner)
    assert (saved["status"], saved["reason"]) == ("interrupted", "StepFailure")
    assert not saved["payload"].get("chat_endings")


@pytest.mark.parametrize("initial", ["usable", "version", "replies", "cap", "withdrawn", "replay"])
@pytest.mark.parametrize("later", ["none", "version", "cap", "withdrawn"])
def test_chat_cause_and_events_across_steps(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, initial: str, later: str
) -> None:
    if initial in ("version", "cap", "withdrawn") and later != "none":
        pytest.skip("the earlier version, cap or withdrawal prevents recall requests")
    runner = runs(chat_client=ChatFault(initial, at=5))
    with monkeypatch.context() as patch:
        if initial == "replay":

            def failed(*args: Any, **kwargs: Any) -> Any:
                raise chat.Unusable(chat.Cause.REPLAY)

            patch.setattr(chat, "replay", failed)
        closed = runner.execute("closed-loop")
    path = runner.guard.directory / "closed-loop.json"
    before = path.read_bytes()
    runner.steps.chat_client = ChatFault(later, at=1)
    recalled = runner.execute("recall")
    first, last = closed["payload"]["chat"], recalled["payload"]["chat"]
    assert path.read_bytes() == before
    assert first["cause"] == (None if initial == "usable" else initial.upper())
    assert (first["status"] == "unusable") == (first["cause"] is not None)
    assert last["cause"] == (first["cause"] or (None if later == "none" else later.upper()))
    assert last["endings"][: len(first["endings"])] == first["endings"]
    assert len(first["endings"]) == int(initial in ("withdrawn", "replay"))
    assert len(last["endings"]) == len(first["endings"]) + int(later == "withdrawn")
    assert (last["status"] == "unusable") == (last["cause"] is not None)
    assert recalled["status"] == "completed"
