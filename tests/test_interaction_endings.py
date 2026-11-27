"""The single-save failure boundary."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.test_interaction_step_faults import ChatFault, Factory, state
from zoon.interaction import budget, chat, guard, run, steps
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)


real_sync = os.fsync


def invalid_sync(fd: int) -> None:
    real_sync(999999)


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_tampered_state_is_never_signed(runs: Factory, step: str) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")

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
@pytest.mark.parametrize("metadata", [404, 410])
@pytest.mark.parametrize("generation", [200, 404, 410])
def test_missing_metadata_defers_withdrawal_to_generation(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str, metadata: int, generation: int
) -> None:
    from tests.test_interaction_chat import payload

    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
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


def test_first_group_failure_stops_following_group(runs: Factory) -> None:
    client = ChatFault("transport", at=3)
    runner = runs(chat_client=client, simulation_groups=2)
    with pytest.raises(chat.TransportInterruption):
        runner.execute("closed-loop")
    assert client.calls == 3


def test_recall_record_reads_saved_recall_only(runs: Factory) -> None:
    runner = runs(prepared="closed-loop")
    with runner.guard.lock(), pytest.raises(clients.GuardRefusal, match="completed recall"):
        runner.guard.recall_chat_record()
    done = runner.execute("recall")
    with runner.guard.lock():
        assert runner.guard.recall_chat_record() == done["payload"]["chat"]


@pytest.mark.parametrize("damage", ["missing", "altered"])
def test_guarded_output_restoration(runs: Factory, damage: str) -> None:
    runner = runs()
    name = next(iter(state(runner)["payload"]["digests"]))
    path = runner.guard.storage / name
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"altered")
    for _ in range(2):
        with pytest.raises(clients.GuardRefusal) as caught:
            runner.execute("closed-loop")
        if damage == "missing":
            assert str(caught.value) == f"missing guarded output {name}"
            assert str(runner.guard.storage) not in str(caught.value)


def test_injected_chat_cap_stops_before_first_call(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from decimal import Decimal

    client = ChatFault("transport", at=1)
    monkeypatch.setattr(budget, "chat_cap", lambda: Decimal(0))
    runner = runs(chat_client=client)
    done = runner.execute("closed-loop")
    assert done["payload"]["chat"]["cause"] == "CAP"
    assert client.calls == 0


def test_calibration_records_only_decision_transcript(runs: Factory) -> None:
    runner = runs(prepared="calibrated")
    assert set(state(runner, "calibrate")["transcripts"]) == {"decision"}


@pytest.mark.parametrize("family", ["chat", "unknown"])
def test_unregistered_transcript_family_is_refused(runs: Factory, family: str) -> None:
    runner = runs(prepared="calibrated")
    path = runner.guard.directory / "calibrate.json"
    saved = json.loads(path.read_bytes())
    saved["transcripts"][family] = {"keys": [], "digest": guard.digest(guard.encoded([]))}
    saved.pop("digest")
    path.write_bytes(guard.encoded({**saved, "digest": guard.digest(guard.encoded(saved))}))
    original = path.read_bytes()
    gate = guard.Guard(runner.guard.root, stub=True, storage=runner.guard.storage)
    gate.register_transcripts("decision")
    with gate.lock(), pytest.raises(clients.GuardRefusal, match="transcript reader"):
        gate.before("closed-loop")
    assert path.read_bytes() == original


@pytest.mark.parametrize("after", ["closed-loop", "recall"])
def test_fresh_calibration_refuses_completed_calling_step(runs: Factory, after: str) -> None:
    runner = runs(prepared="closed-loop")
    if after == "recall":
        runner.execute("recall")
    fresh = run.Run(runner.guard.root, stub=True, storage=runner.guard.storage, simulation_groups=1)
    with pytest.raises(clients.GuardRefusal, match="a completed calling step cannot be repeated"):
        fresh.execute("calibrate")


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_chat_is_verified_before_each_step_attempt(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    if step == "recall":
        runner.execute("recall")
    calls = 0
    original = runner.steps.transcript

    def transcript(*, read_only: bool = True) -> chat.Transcript:
        nonlocal calls
        calls += 1
        return original(read_only=read_only)

    monkeypatch.setattr(runner.steps, "transcript", transcript)
    path = runner.guard.transcript_directory / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes() if path.exists() else b""
    path.write_bytes(raw + b"garbage")
    for attempt in (1, 2):
        with pytest.raises(clients.GuardRefusal, match="incomplete transcript line"):
            runner.execute(step)
        assert calls == attempt


def test_recall_repeat_keeps_withdrawal_after_replay_and_failed_completion(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise chat.Unusable(chat.Cause.REPLAY)

    with monkeypatch.context() as patch:
        patch.setattr(chat, "replay", fail)
        runner.execute("closed-loop")
    client = ChatFault("withdrawn", at=2)
    runner.steps.chat_client = client
    original = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "completed":
            raise OSError("completion unavailable")
        return original(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError):
            runner.execute("recall")
    assert client.calls == 2
    done = runner.execute("recall")
    assert client.calls == 2
    assert done["payload"]["stopped_by"] == "WITHDRAWN"
    assert done["payload"]["chat_endings"] == [
        {"cause": "REPLAY", "step": "closed-loop"},
        {"cause": "WITHDRAWN", "step": "recall"},
    ]


def test_chat_games_follow_sorted_cells(runs: Factory, monkeypatch: pytest.MonkeyPatch) -> None:
    from zoon.interaction import simulation

    runner = runs()
    order: list[str] = []
    monkeypatch.setattr(simulation, "GAMES", dict(reversed(sorted(chat.GAMES.items()))))
    original = simulation.play_chat_game

    def play(game: chat.Game, session: chat.Session, groups: int) -> Any:
        order.append(game.cell)
        return original(game, session, groups)

    monkeypatch.setattr(simulation, "play_chat_game", play)
    runner.execute("closed-loop")
    assert order == sorted(chat.GAMES)


@pytest.mark.parametrize("status", [429, 503])
def test_generating_retry_exhaustion_is_transport(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    from tests.test_interaction_http_boundary import connection

    client, methods, pauses = connection(
        monkeypatch, lambda method, count: (200, b"{}") if method == "GET" else (status, b"{}")
    )
    runner = runs(chat_client=client)
    with pytest.raises(chat.TransportInterruption):
        runner.execute("closed-loop")
    assert methods == ["GET"] + ["POST"] * 4 and pauses == [1, 2, 4]
    assert not state(runner)["payload"].get("chat_endings")


@pytest.mark.parametrize("fault", ["withdrawn", "invalid-version", "none", "empty"])
def test_recall_command_prints_stop_and_question_labels(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fault: str
) -> None:
    runner = runs(prepared="closed-loop")
    if fault == "empty":
        from dataclasses import replace

        class Empty(chat.StubClient):
            def ask(self, text: str) -> chat.Response:
                return replace(super().ask(text), text="")

        runner.steps.chat_client = Empty()
    elif fault != "none":
        runner.steps.chat_client = ChatFault(fault, at=2)

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    assert run.main(["recall", "--stub"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    stop = {
        "withdrawn": "WITHDRAWN",
        "invalid-version": "VERSION",
        "none": "None",
        "empty": "None",
    }[fault]
    assert f"[recall stopped by] {stop}\n" in output.out
    replies = state(runner, "recall")["payload"]["recall"]
    for index, reply in enumerate(replies):
        label = (
            "judged"
            if fault in ("none", "empty") or index == 0
            else (
                "not judged: version" if fault == "invalid-version" and index == 1 else "not asked"
            )
        )
        assert reply["status"] == label
        response = reply["response"]
        text = response["text"] if response and response["text"] else "no reply"
        assert (
            f"[recall {reply['game']}] {label} | flagged = {reply['flagged']} | {text}\n"
            in output.out
        )


@pytest.mark.parametrize("damage", ["missing", "directory", "permission"])
def test_unreadable_guarded_output_command_uses_relative_name(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    runner = runs()
    name = next(iter(state(runner)["payload"]["digests"]))
    path = runner.guard.storage / name
    if damage in ("missing", "directory"):
        path.unlink()
        if damage == "directory":
            path.mkdir()
    else:
        original = Path.read_bytes

        def read_bytes(current: Path) -> bytes:
            if current == path:
                raise PermissionError(str(current))
            return original(current)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    assert run.main(["closed-loop", "--stub"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    message = (
        f"missing guarded output {name}"
        if damage == "missing"
        else f"guarded output {name} cannot be read"
    )
    assert output.err == f"GuardRefusal: {message}\n"
    assert str(runner.guard.storage) not in output.err
    assert state(runner)["status"] == "interrupted"
    assert state(runner)["reason"] == "GuardRefusal"


@pytest.mark.parametrize("constructor", ["run", "steps"])
def test_chat_cap_cannot_be_passed_to_constructor(runs: Factory, constructor: str) -> None:
    from decimal import Decimal

    runner = runs()
    options: dict[str, Any] = {"chat_cap": Decimal(0)}
    with pytest.raises(TypeError, match="chat_cap"):
        if constructor == "run":
            run.Run(runner.guard.root, stub=True, storage=runner.guard.storage, **options)
        else:
            steps.Steps(runner.guard, runner.steps.client, **options)


@pytest.mark.parametrize("damage", ["missing", "altered", "dangling"])
def test_guarded_copy_restores_same_result(runs: Factory, damage: str) -> None:
    baseline = runs("baseline").execute("closed-loop")["payload"]
    runner = runs("damaged")
    name = next(iter(state(runner)["payload"]["digests"]))
    path = runner.guard.storage / name
    raw = path.read_bytes()
    if damage == "altered":
        path.write_bytes(b"wrong")
    else:
        path.unlink()
        if damage == "dangling":
            path.symlink_to(runner.guard.storage / "absent.jsonl")
    for _ in range(2):
        with pytest.raises(clients.GuardRefusal) as caught:
            runner.execute("closed-loop")
        expected = (
            "recorded simulation digest differs from its output"
            if damage == "altered"
            else f"missing guarded output {name}"
        )
        assert str(caught.value) == expected
    if path.is_symlink():
        path.unlink()
    path.write_bytes(b"wrong copy")
    with pytest.raises(clients.GuardRefusal, match="digest differs"):
        runner.execute("closed-loop")
    path.write_bytes(raw)
    assert runner.execute("closed-loop")["payload"] == baseline


@pytest.mark.parametrize("number", [5, 24])
def test_guarded_storage_fault_interrupts_and_repeats(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, number: int
) -> None:
    runner = runs()
    path = runner.guard.storage / next(iter(state(runner)["payload"]["digests"]))
    original = Path.read_bytes

    def read(current: Path) -> bytes:
        if current == path:
            raise OSError(number, "storage unavailable")
        return original(current)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read)
        with pytest.raises(OSError) as caught:
            runner.execute("closed-loop")
        assert caught.value.errno == number
    assert state(runner)["reason"] == "OSError"
    assert state(runner)["status"] == "interrupted"
    assert runner.execute("closed-loop")["status"] == "completed"


def test_guarded_output_registration_sentences() -> None:
    text = " ".join((Path(__file__).parents[1] / "interaction/registration.md").read_text().split())
    for sentence in (
        "Apart from a declared repair, the incomplete-line cut and restoring a guarded output "
        "under Guard, repairs and persistence, this is the only operator decision the run "
        "allows, and it cannot be taken once `held-out` has started.",
        "Do not delete or reset state or transcripts during the run, except for that cut; "
        "restoring a guarded output does not change state or transcripts.",
        "A guarded output that is missing, unreadable or differs from its recorded "
        "digest refuses the step.",
        "The operator may put back a copy of the file, and the step proceeds only if its bytes "
        "match the recorded digest; this is not a decision about any outcome.",
    ):
        assert sentence in text


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_calling_step_declares_both_transcripts(
    runs: Factory, step: str, interrupted: bool
) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    if interrupted:
        runner.steps.chat_client = ChatFault("transport", at=1)
        with pytest.raises(chat.TransportInterruption):
            runner.execute(step)
    else:
        runner.execute(step)
    assert set(state(runner, step)["transcripts"]) == {"decision", "chat"}


def test_save_cannot_omit_previously_declared_family(runs: Factory) -> None:
    runner = runs()
    with runner.guard.lock():
        runner.guard.before("closed-loop")
        with pytest.raises(clients.GuardRefusal, match="families changed"):
            runner.guard.save(
                "closed-loop", state(runner)["payload"], families=("decision",), status="running"
            )


def test_save_declared_family_requires_reader(runs: Factory) -> None:
    runner = runs(prepared="calibrated")
    with runner.guard.lock():
        runner.guard.before("closed-loop")
        with pytest.raises(clients.GuardRefusal, match="transcript reader"):
            runner.guard.save("closed-loop", {}, families=("unknown",), status="running")
