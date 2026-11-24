"""Transcript receipts survive storage faults independently of derived game logs."""

import os
from pathlib import Path
from typing import Any, cast

import pytest

from tests.test_interaction_step_faults import (
    ChatFault,
    Factory,
    state,
)
from zoon.interaction import chat, run, simulation
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)
REAL_FSYNC = os.fsync


def invalid_fsync(fd: int) -> None:
    REAL_FSYNC(999999)


def fresh(runner: run.Run) -> run.Run:
    return run.Run(runner.guard.root, stub=True, storage=runner.guard.storage, simulation_groups=1)


def test_every_interruption_preserves_receipts(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = runs("plain")
    expected = plain.execute("closed-loop")
    paid: list[str] = []
    original_ask = chat.StubClient.ask

    def ask(client: chat.StubClient, text: str) -> chat.Response:
        paid.append(text)
        return original_ask(client, text)

    # Every save (initial, each committed arm, completion), and every tenth request.
    def exercise(kind: str, point: int) -> None:
        runner = runs(f"{kind}-{point}")
        paid.clear()
        calls = 0
        original_save = runner.steps.save

        def save(status: str = "running", reason: str = "") -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if kind == "save" and calls == point:
                with monkeypatch.context() as patch:
                    # Real fsync on a closed descriptor fails inside the real atomic save.
                    patch.setattr(os, "fsync", invalid_fsync)
                    return original_save(status, reason)
            return original_save(status, reason)

        def request(client: chat.StubClient, text: str) -> chat.Response:
            nonlocal calls
            if kind == "request":
                calls += 1
                if calls == point:
                    with monkeypatch.context() as patch:
                        patch.setattr(os, "fsync", invalid_fsync)
                        return ask(client, text)
            return ask(client, text)

        with monkeypatch.context() as patch:
            patch.setattr(runner.steps, "save", save)
            patch.setattr(chat.StubClient, "ask", request)
            if kind == "request":
                # Fail append fsync after the paid reply has been durably written.
                original_append = chat.Transcript.append

                def append(transcript: chat.Transcript, *args: Any, **kwargs: Any) -> Any:
                    if len(paid) == point:
                        with monkeypatch.context() as inner:
                            inner.setattr(os, "fsync", invalid_fsync)
                            return original_append(transcript, *args, **kwargs)
                    return original_append(transcript, *args, **kwargs)

                patch.setattr(chat.Transcript, "append", append)
            with pytest.raises((OSError, chat.Interruption)):
                runner.execute("closed-loop")
        with monkeypatch.context() as patch:
            patch.setattr(chat.StubClient, "ask", ask)
            actual = fresh(runner).execute("closed-loop")
        assert actual == expected, (kind, point)
        assert len(paid) == 440, (kind, point)
        for name in expected["payload"]["digests"] | expected["payload"]["chat_logs"]:
            assert (runner.guard.storage / name).read_bytes() == (
                plain.guard.storage / name
            ).read_bytes()
        for name in ("chat.jsonl", "decision.jsonl"):
            assert (runner.guard.transcript_directory / name).read_bytes() == (
                plain.guard.transcript_directory / name
            ).read_bytes()

    for kind, points in (("save", range(1, 6)), ("request", range(10, 441, 10))):
        for point in points:
            exercise(kind, point)


@pytest.mark.parametrize("stage", ["first", "completed", "resumed"])
def test_unreadable_chat_never_rewrites_checkpoint(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    runner = runs()
    runner.execute("closed-loop")
    if stage != "first":
        runner.steps.chat_client = ChatFault("transport", at=1)
        with pytest.raises(chat.TransportInterruption):
            runner.execute("recall")
    runner.steps.chat_client = None
    previous = state(runner, "recall") if stage != "first" else None
    original_save, original_open = runner.steps.save, Path.open
    active = False

    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        if active and path.name == "chat.jsonl":
            return original_open(path.parent, "rb")  # Real open raises IsADirectoryError.
        return cast(Any, original_open)(path, *args, **kwargs)

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        nonlocal active
        active = active or stage != "completed" or status == "completed"
        return original_save(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", opening)
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(chat.StorageInterruption):
            runner.execute("recall")
    path = runner.guard.directory / "recall.json"
    if previous is None:
        assert not path.exists()
    else:
        saved = state(runner, "recall")
        assert saved["status"] == ("running" if stage == "completed" else "interrupted")
        assert saved["transcripts"] == previous["transcripts"]
        payload = saved["payload"].copy()
        old_payload = previous["payload"].copy()
        assert payload == old_payload
    completed = fresh(runner).execute("recall")
    assert all(r["status"] == "judged" for r in completed["payload"]["recall"])


@pytest.mark.parametrize("damage", ["delete", "alter"])
@pytest.mark.parametrize("family", ["chat-log", "coded-log", "transcript", "state"])
def test_file_damage_between_attempts(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, family: str, damage: str
) -> None:
    runner = runs()
    original = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "completed":
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", invalid_fsync)
                return original(status, reason)
        return original(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute("closed-loop")
    logs = sorted(
        (
            runner.guard.directory / "closed-loop" / ("chat" if family == "chat-log" else "zoon")
        ).rglob("*.jsonl")
    )
    target = logs[0]
    before = target.read_bytes()
    if family == "transcript":
        target = runner.guard.transcript_directory / "chat.jsonl"
        target.write_bytes(target.read_bytes()[:-3])
    elif family == "state":
        target = runner.guard.directory / "closed-loop.json"
        if damage == "delete":
            target.unlink()
        else:
            target.write_bytes(b"{}")
    elif damage == "delete":
        target.unlink()
    else:
        target.write_bytes(b"altered")
    if family in ("chat-log",) or (family == "state" and damage == "delete"):
        actual = fresh(runner).execute("closed-loop")
        assert actual == runs("control").execute("closed-loop")
        assert logs[0].read_bytes() == before
    else:
        with pytest.raises((clients.GuardRefusal, FileNotFoundError)) as caught:
            fresh(runner).execute("closed-loop")
        if family == "transcript":
            assert "cut the file back to" in str(caught.value)


def test_final_status_comes_from_recall(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = runs()

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    assert run.main(["closed-loop", "--stub"]) == 0
    assert "[simulation] logs = 32" in capsys.readouterr().out
    runner.steps.chat_client = ChatFault("version", at=1)
    assert run.main(["recall", "--stub"]) == 0
    assert '"cause": "VERSION"' in capsys.readouterr().out
    assert state(runner)["payload"]["chat"]["status"] == "usable"
    with runner.guard.lock():
        assert runner.guard.final_chat_record()["status"] == "unusable"
        assert runner.guard.final_chat_record()["cause"] == "VERSION"


@pytest.mark.parametrize("fault", ["VERSION", "CAP"])
def test_terminal_save_failure_is_repeated_from_receipts(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    class Decision(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            from dataclasses import replace

            response = super().ask(request)
            return replace(response, version="changed")

    from decimal import Decimal

    runner = runs(client=Decision("decision"), cap=Decimal(0) if fault == "CAP" else None)
    original_save = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "terminal":
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", invalid_fsync)
                return original_save(status, reason)
        return original_save(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute("closed-loop")
    assert state(runner)["status"] == "running"
    assert "outcome" not in state(runner)["payload"]
    assert runner.execute("closed-loop")["payload"]["outcome"]["reason"] == fault


def test_completed_chat_logs_are_never_guard_inputs(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs()
    completed = runner.execute("closed-loop")
    for name in completed["payload"]["chat_logs"]:
        (runner.guard.storage / name).unlink()
    runner.execute("recall")
    original_open = Path.open

    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        if "closed-loop/chat" in path.as_posix():
            return original_open(path.parent, "rb")
        return cast(Any, original_open)(path, *args, **kwargs)

    with runner.guard.lock(), monkeypatch.context() as patch:
        patch.setattr(Path, "open", opening)
        saved = runner.guard.state("closed-loop")
        simulation.Store(runner.guard, completed["payload"], lambda: None).verify()
        runner.guard.mode, runner.guard.checked_step = "live", "held-out"

        def saved_state(step: str) -> dict[str, Any] | None:
            return saved

        patch.setattr(runner.guard, "state", saved_state)
        assert runner.guard.permit().authorized()


def test_final_replay_uses_transcript_and_keeps_ending(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    runner = runs()
    original_play, original_save = simulation.play_chat_game, runner.steps.save

    def play(*args: Any, **kwargs: Any) -> tuple[simulation.kernel.Run, ...]:
        games = original_play(*args, **kwargs)
        return tuple(replace(game, log=(*game.log, " \n")) for game in games)

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "completed":
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", invalid_fsync)
                return original_save(status, reason)
        return original_save(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(simulation, "play_chat_game", play)
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute("closed-loop")
    assert state(runner)["payload"]["chat_endings"] == [{"cause": "REPLAY", "step": "closed-loop"}]
    result = fresh(runner).execute("closed-loop")
    assert result["payload"]["chat"]["cause"] == "REPLAY"
    assert "chat_logs" not in result["payload"]
