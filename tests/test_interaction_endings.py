"""Attempt-local operator endings and the single-save failure boundary."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.test_interaction_step_faults import ChatFault, Factory, interrupt_chat, state
from zoon.interaction import chat, guard, run
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)


real_sync = os.fsync


def invalid_sync(fd: int) -> None:
    real_sync(999999)


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("point", ["first", "middle", "metadata", "none"])
@pytest.mark.parametrize("flag", [False, True])
def test_operator_flag_applies_only_to_chat_transport(
    runs: Factory, step: str, point: str, flag: bool
) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")

    class Fault(ChatFault):
        def preflight(self) -> None:
            if point == "metadata":
                raise chat.TransportInterruption("metadata transport unavailable")
            super().preflight()

    if flag:
        interrupt_chat(runner, step)
    runner.steps.chat_client = Fault(
        "none" if point == "none" else "transport", at=3 if point == "middle" else 1
    )
    if point != "none" and not flag:
        with pytest.raises(chat.TransportInterruption):
            runner.execute(step)
        saved = state(runner, step)
        assert saved["status"] == "interrupted"
        assert saved["reason"] == "TransportInterruption"
        assert not saved["payload"].get("chat_endings")
        return
    result = runner.execute(step, end_chat="Unavailable" if flag else None)
    assert result["status"] == "completed"
    events = result["payload"].get("chat_endings", [])
    if point == "none":
        assert not events
    else:
        assert events == [
            {
                "chat_transport_failures": 2,
                **({"answered_recall": 2 if point == "middle" else 0} if step == "recall" else {}),
                "cause": "OPERATOR",
                "reason": "Unavailable",
                "step": step,
                "answered_decisions": (440 if step == "recall" else 2 if point == "middle" else 0),
            }
        ]
        if step == "recall":
            assert result["payload"]["stopped_by"] == "OPERATOR"
            asked = 2 if point == "middle" else 0
            assert [r["status"] for r in result["payload"]["recall"]] == (
                ["judged"] * asked + ["not asked"] * (8 - asked)
            )
        else:
            assert "chat_logs" not in result["payload"]


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("cause", ["withdrawn", "operator"])
@pytest.mark.parametrize("handler_fails", [False, True])
@pytest.mark.parametrize("repeat_flag", [False, True])
def test_ending_save_boundary(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    step: str,
    cause: str,
    handler_fails: bool,
    repeat_flag: bool,
) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")
    runner.steps.chat_client = ChatFault("transport" if cause == "operator" else "withdrawn", at=3)
    if cause == "operator":
        interrupt_chat(runner, step)
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
            real_fsync = os.fsync

            def invalid_fsync(fd: int) -> None:
                real_fsync(999999)

            with monkeypatch.context() as inner:
                inner.setattr(os, "fsync", invalid_fsync)
                return original(status, reason)
        return original(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute(step, **({"end_chat": "Unavailable"} if cause == "operator" else {}))
    saved = state(runner, step)
    assert saved["status"] == ("running" if handler_fails else "interrupted")
    assert bool(saved["payload"].get("chat_endings")) == (not handler_fails)
    runner.steps.chat_client = None
    if repeat_flag:
        with pytest.raises(clients.GuardRefusal, match="only after an attempt"):
            runner.execute(step, end_chat="Unavailable")
    done = runner.execute(step)
    assert done["status"] == "completed"
    assert bool(done["payload"].get("chat_endings")) == (not handler_fails)
    if not handler_fails:
        assert done["payload"]["chat"]["cause"] == cause.upper()


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_tampered_state_is_never_signed(runs: Factory, step: str) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")

    class Tamper(ChatFault):
        def ask(self, text: str) -> chat.Response:
            if self.calls == 2:
                path = runner.guard.directory / f"{step}.json"
                document = json.loads(path.read_bytes())
                document["payload"]["forged"] = True
                path.write_text(json.dumps(document))
            return super().ask(text)

    runner.steps.chat_client = Tamper("transport", at=3)
    with pytest.raises(clients.GuardRefusal, match="digest changed"):
        runner.execute(step)
    with runner.guard.lock(), pytest.raises(clients.GuardRefusal, match="digest changed"):
        runner.guard.state(step)


@pytest.mark.parametrize("fault", ["VERSION", "CAP", "WITHDRAWN"])
def test_terminal_replace_is_never_followed_by_another_save(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from dataclasses import replace
    from decimal import Decimal

    class Decision(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if fault == "WITHDRAWN":
                raise clients.TerminalStop(clients.StopReason.WITHDRAWN)
            return replace(super().ask(request), version="changed")

    runner = runs(client=Decision("decision"), cap=Decimal(0) if fault == "CAP" else None)
    original = guard.atomic
    visible = b""

    def atomic(path: Path, document: dict[str, Any]) -> None:
        nonlocal visible
        original(path, document)
        if document.get("status") == "terminal":
            visible = path.read_bytes()
            raise guard.StateReplacedError("directory sync unavailable")

    monkeypatch.setattr(guard, "atomic", atomic)
    with pytest.raises(guard.StateReplacedError):
        runner.execute("closed-loop")
    assert visible == (runner.guard.directory / "closed-loop.json").read_bytes()
    assert state(runner)["status"] == "terminal"


@pytest.mark.parametrize("status", ["running", "interrupted", "terminal"])
def test_recall_chat_record_requires_completed_recall(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    runner = runs()

    def saved(step: str) -> dict[str, Any]:
        return {"status": status}

    monkeypatch.setattr(runner.guard, "state", saved)
    with pytest.raises(clients.GuardRefusal, match="completed recall"):
        runner.guard.recall_chat_record()


def test_failed_replay_publishes_no_chat_logs(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise chat.Unusable(chat.Cause.REPLAY)

    monkeypatch.setattr(chat, "replay", fail)
    done = runner.execute("closed-loop")
    assert done["payload"]["chat"]["cause"] == "REPLAY"
    assert not list((runner.guard.directory / "closed-loop/chat").rglob("*.jsonl"))


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_flag_on_recorded_ending_does_nothing(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], step: str
) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")
    runner.steps.chat_client = ChatFault("transport", at=3)
    interrupt_chat(runner, step)
    original = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "completed":
            raise OSError("completion unavailable")
        return original(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute(step, end_chat="First reason")
    events = state(runner, step)["payload"]["chat_endings"]
    capsys.readouterr()
    runner.steps.chat_client = ChatFault("transport", at=1)
    with pytest.raises(clients.GuardRefusal, match="only after an attempt"):
        runner.execute(step, end_chat="Second reason")
    done = runner.execute(step)
    assert done["payload"]["chat_endings"] == events
    assert runner.steps.chat_client.calls == 0
    assert "flag had no effect" not in capsys.readouterr().out


@pytest.mark.parametrize("flag", [False, True])
def test_decision_transport_cannot_end_chat(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, flag: bool
) -> None:
    class Decision(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            raise clients.StepFailure("decision transport unavailable")

    runner = runs(client=Decision("decision"))
    if flag:
        interrupt_chat(runner, "closed-loop")
        from zoon.interaction import simulation

        def bands(*args: Any, **kwargs: Any) -> None:
            raise clients.StepFailure("decision transport unavailable")

        monkeypatch.setattr(simulation.Store, "bands", bands)
    with pytest.raises(clients.StepFailure):
        runner.execute("closed-loop", end_chat="Unavailable" if flag else None)
    assert state(runner)["status"] == "interrupted"
    assert state(runner)["reason"] == "StepFailure"
    assert not state(runner)["payload"].get("chat_endings")


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

    runner = runs()
    original_save, original_write = runner.steps.save, simulation.write
    original_ask, original_decision = chat.StubClient.ask, clients.StubClient.ask
    original_append = chat.Transcript.append
    previous: dict[str, Any] = {}
    fired = False

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
        nonlocal previous, fired
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
    actual["payload"].pop("chat_transport_failures", None)
    assert {k: v for k, v in actual.items() if k not in ("digest", "payload_digest")} == {
        k: v for k, v in expected.items() if k not in ("digest", "payload_digest")
    }


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_printing_uses_the_returned_step_state(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], step: str
) -> None:
    runner = runs()
    done = runner.execute("closed-loop")
    if step == "recall":
        done = runner.execute("recall")

    def execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return done

    def lock() -> Any:
        pytest.fail("printing must use the returned document")

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner.guard, "lock", lock)
    monkeypatch.setattr(run, "Run", local)
    assert run.main([step, "--stub"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "[chat] " in output.out


@pytest.mark.parametrize("cause", ["REPLAY", "WITHDRAWN", "OPERATOR"])
@pytest.mark.parametrize("again", [False, True])
def test_unrecorded_ending_is_observed_again(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, cause: str, again: bool
) -> None:
    runner = runs(chat_client=ChatFault("transport" if cause == "OPERATOR" else "withdrawn", at=1))
    if cause == "OPERATOR":
        interrupt_chat(runner, "closed-loop")
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
            runner.execute("closed-loop", end_chat="Unavailable" if cause == "OPERATOR" else None)
    assert not state(runner)["payload"].get("chat_endings")
    runner.steps.chat_client = (
        ChatFault("transport" if cause == "OPERATOR" else "withdrawn", at=1) if again else None
    )
    with monkeypatch.context() as patch:
        if cause == "REPLAY":
            patch.setattr(chat, "replay", replay)
        if cause == "OPERATOR" and again:
            with pytest.raises(chat.TransportInterruption):
                runner.execute("closed-loop")
            runner.steps.chat_client = ChatFault("transport", at=1)
        done = runner.execute(
            "closed-loop", end_chat="Unavailable" if cause == "OPERATOR" and again else None
        )
    assert done["payload"]["chat"]["cause"] == (cause if again or cause == "REPLAY" else None)


@pytest.mark.parametrize("cause", ["VERSION", "CAP", "WITHDRAWN"])
@pytest.mark.parametrize("withdraws_again", [False, True])
def test_failed_terminal_save_leaves_state_and_redetects_outcome(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, cause: str, withdraws_again: bool
) -> None:
    from dataclasses import replace
    from decimal import Decimal

    class Decision(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if cause == "WITHDRAWN":
                raise clients.TerminalStop(clients.StopReason.WITHDRAWN)
            return replace(super().ask(request), version="changed")

    runner = runs(client=Decision("decision"), cap=Decimal(0) if cause == "CAP" else None)
    original = runner.steps.save
    previous: dict[str, Any] = {}

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        nonlocal previous
        if status == "terminal":
            previous = state(runner)
            with monkeypatch.context() as inner:
                inner.setattr(os, "fsync", invalid_sync)
                return original(status, reason)
        return original(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute("closed-loop")
    assert state(runner) == previous
    if cause != "WITHDRAWN" or not withdraws_again:
        runner.steps.client = clients.StubClient("decision")
    done = runner.execute("closed-loop")
    if cause == "WITHDRAWN" and not withdraws_again:
        assert done == runs("plain").execute("closed-loop")
    else:
        assert done["status"] == "terminal"
        assert done["payload"]["outcome"]["reason"] == cause


def test_completed_state_replace_is_never_rewritten(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs()
    original = guard.atomic
    visible = b""

    def atomic(path: Path, document: dict[str, Any]) -> None:
        nonlocal visible
        original(path, document)
        if document.get("status") == "completed":
            visible = path.read_bytes()
            raise guard.StateReplacedError("directory sync unavailable")

    monkeypatch.setattr(guard, "atomic", atomic)
    with pytest.raises(guard.StateReplacedError):
        runner.execute("closed-loop")
    assert (runner.guard.directory / "closed-loop.json").read_bytes() == visible


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("reason", ["", " \t\n"])
def test_blank_operator_reason_runs_nothing(runs: Factory, step: str, reason: str) -> None:
    runner = runs()
    with pytest.raises(clients.GuardRefusal, match="nonempty reason"):
        runner.execute(step, end_chat=reason)
    assert not (runner.guard.directory / f"{step}.json").exists()


@pytest.mark.parametrize("stub", [False, True])
@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_cli_routes_operator_flag_in_either_mode(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stub: bool,
    step: str,
) -> None:
    runner = runs()
    result = runner.execute("closed-loop")
    if step == "recall":
        result = runner.execute("recall")

    def execute(
        command: str, *, repair: str | None = None, end_chat: str | None = None
    ) -> dict[str, Any]:
        assert command == step and end_chat == "Unavailable"
        return result

    def local(**kwargs: Any) -> run.Run:
        assert kwargs["stub"] == stub
        return runner

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(run, "Run", local)
    assert run.main([step, "--end-chat", "Unavailable", *(["--stub"] if stub else [])]) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("metadata", [404, 410])
@pytest.mark.parametrize("generation", [200, 404, 410])
def test_missing_metadata_defers_withdrawal_to_generation(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str, metadata: int, generation: int
) -> None:
    from tests.test_interaction_chat import payload

    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    calls: list[Any] = []

    def post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        calls.append(url)
        return generation, payload(
            modelVersion=chat.MODEL + "-stub",
            candidates=[{"content": {"parts": [{"text": "3"}]}}],
        )

    def metadata_get(*args: Any) -> tuple[int, dict[str, Any]]:
        return metadata, {}

    runner.steps.chat_client = chat.VertexClient(
        post,
        token=lambda: "dummy",
        preflight_transport=metadata_get,
        pause=lambda seconds: None,
    )
    result = runner.execute(step)
    assert result["status"] == "completed"
    if generation != 200:
        assert len(calls) == 4
        assert result["payload"]["chat"]["cause"] == "WITHDRAWN"
        assert result["payload"]["chat_endings"][-1] == {"cause": "WITHDRAWN", "step": step}
    else:
        assert len(calls) > 4
        assert result["payload"]["chat"]["cause"] is None


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_other_metadata_statuses_keep_their_meaning(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    calls: list[Any] = []

    def metadata(*args: Any) -> tuple[int, dict[str, Any]]:
        calls.append(status)
        return status, {}

    client = chat.VertexClient(
        metadata, token=lambda: "dummy", preflight_transport=metadata, pause=lambda seconds: None
    )
    transport = status in (401, 429) or status >= 500
    with pytest.raises(chat.TransportInterruption if transport else chat.ConfigurationInterruption):
        client.preflight()
    assert len(calls) == (4 if transport else 1)


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize(
    "failure",
    [
        chat.StorageInterruption,
        chat.ConfigurationInterruption,
        chat.ConsistencyInterruption,
        chat.GuardRefusal,
        OSError,
    ],
)
def test_operator_flag_preserves_other_failures(
    runs: Factory, step: str, failure: type[Exception]
) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")
    interrupt_chat(runner, step)

    class Fault(ChatFault):
        def ask(self, text: str) -> chat.Response:
            raise failure("unavailable")

    runner.steps.chat_client = Fault("none")
    with pytest.raises((failure, chat.ConsistencyInterruption)):
        runner.execute(step, end_chat="Unavailable")
    saved = state(runner, step)
    assert saved["status"] == "interrupted"
    assert not saved["payload"].get("chat_endings")
    assert saved["payload"]["chat_transport_failures"] == 1


def test_first_group_failure_stops_following_group(runs: Factory) -> None:
    client = ChatFault("transport", at=3)
    runner = runs(chat_client=client, simulation_groups=2)
    with pytest.raises(chat.TransportInterruption):
        runner.execute("closed-loop")
    assert client.calls == 3


def test_recall_record_reads_saved_recall_only(runs: Factory) -> None:
    runner = runs()
    runner.execute("closed-loop")
    with runner.guard.lock(), pytest.raises(clients.GuardRefusal, match="completed recall"):
        runner.guard.recall_chat_record()
    done = runner.execute("recall")
    with runner.guard.lock():
        assert runner.guard.recall_chat_record() == done["payload"]["chat"]


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_flag_requires_saved_chat_transport_interruption(runs: Factory, step: str) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")
    with pytest.raises(clients.GuardRefusal, match="only after an attempt"):
        runner.execute(step, end_chat="Unavailable")
    assert not (runner.guard.directory / f"{step}.json").exists()


def test_calibration_refuses_operator_flag(runs: Factory) -> None:
    with pytest.raises(clients.GuardRefusal, match="only for closed-loop or recall"):
        runs().execute("calibrate", end_chat="Unavailable")


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("fault", ["none", "version", "cap", "replies"])
def test_command_prints_no_effect_once_after_completion(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    fault: str,
) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")
    interrupt_chat(runner, step)
    runner.steps.chat_client = ChatFault(fault, at=1)

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    capsys.readouterr()
    assert run.main([step, "--stub", "--end-chat", "Unavailable"]) == 0
    assert capsys.readouterr().out.count("flag had no effect") == 1


def test_numeric_recall_replies_do_not_count_as_decisions(runs: Factory) -> None:
    from dataclasses import replace

    class Numeric(ChatFault):
        def ask(self, text: str) -> chat.Response:
            return replace(super().ask(text), text="3")

    runner = runs()
    runner.execute("closed-loop")
    runner.steps.chat_client = Numeric("transport", at=3)
    with pytest.raises(chat.TransportInterruption):
        runner.execute("recall")
    runner.steps.chat_client = Numeric("transport", at=1)
    done = runner.execute("recall", end_chat="Unavailable")
    ending = done["payload"]["chat_endings"][-1]
    assert ending["answered_decisions"] == 440
    assert ending["answered_recall"] == 2
    assert ending["chat_transport_failures"] == 2
    assert done["payload"]["stopped_by"] == "OPERATOR"
