"""Durable arm outcomes, recovery and bounded operator authority."""

import json
import os
import shutil
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tests.interaction_run_support import repository
from zoon.interaction import chat, kernel, prompted, run, score, simulation
from zoon.interaction.guard import digest
from zoon.pilot import clients

Factory = Callable[..., run.Run]


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory) -> Generator[tuple[Path, Path]]:
    directory = tmp_path_factory.mktemp("calibrated")
    root, storage = repository(directory / "repo"), directory / "storage"
    run.Run(root, stub=True, storage=storage).execute()
    yield root, storage
    shutil.rmtree(directory)


@pytest.fixture
def runs(calibrated: tuple[Path, Path], tmp_path: Path) -> Generator[Factory]:
    root, initial = calibrated

    def make(name: str = "run", **kwargs: Any) -> run.Run:
        storage = tmp_path / name
        shutil.copytree(initial, storage)
        kwargs.setdefault("simulation_groups", 1)
        return run.Run(root, stub=True, storage=storage, **kwargs)

    yield make
    shutil.rmtree(tmp_path)


def state(runner: run.Run, step: str = "closed-loop") -> dict[str, Any]:
    return json.loads((runner.guard.directory / f"{step}.json").read_bytes())


def band_logs(runner: run.Run) -> dict[str, bytes]:
    payload = state(runner)["payload"]
    return {
        n: (runner.guard.storage / n).read_bytes()
        for arm in ("zoon", "alone", "prompted")
        for n in payload["simulations"][arm]
    }


class ChatFault(chat.StubClient):
    def __init__(self, fault: str, at: int = 5) -> None:
        super().__init__()
        self.fault, self.at, self.calls, self.preflights = fault, at, 0, 0

    def preflight(self) -> None:
        self.preflights += 1

    def ask(self, text: str) -> chat.Response:
        assert self.preflights
        self.calls += 1
        if self.calls == self.at:
            if self.fault == "withdrawn":
                raise chat.Unusable(chat.Cause.WITHDRAWN)
            if self.fault == "transport":
                raise chat.TransportInterruption("bounded transport failed")
            if self.fault == "storage":
                raise chat.StorageInterruption("storage failed")
        response = super().ask(text)
        if self.calls >= self.at:
            if self.fault == "version":
                response = replace(response, version=chat.MODEL + "-changed")
            if self.fault == "invalid-version":
                response = replace(response, version="invalid")
            if self.fault == "replies":
                response = replace(response, text="not an integer")
            if self.fault == "cap":
                response = replace(response, input_tokens=1_000_000_000)
        return response


@pytest.mark.parametrize(
    "fault,cause",
    [
        ("version", "VERSION"),
        ("invalid-version", "VERSION"),
        ("replies", "REPLIES"),
        ("cap", "CAP"),
        ("withdrawn", "WITHDRAWN"),
    ],
)
def test_chat_outcomes_keep_all_other_arms_and_complete(
    runs: Factory, fault: str, cause: str
) -> None:
    plain = runs("plain")
    plain.execute("closed-loop")
    client = ChatFault(fault, at=1 if fault == "invalid-version" else 5)
    faulty = runs("faulty", chat_client=client)
    completed = faulty.execute("closed-loop")
    payload = completed["payload"]
    assert completed["status"] == "completed"
    assert payload["chat"]["cause"] == cause
    assert payload["chat"]["status"] == "unusable"
    assert "chat_logs" not in payload
    assert "outcome" not in payload and "chat" not in payload["simulations"]
    assert not any("/chat/" in n for n in payload["digests"])
    assert band_logs(faulty) == band_logs(plain)
    assert {n: d for n, d in payload["digests"].items()} == {
        n: d for n, d in state(plain)["payload"]["digests"].items() if "/chat/" not in n
    }
    assert client.calls == (7 if fault == "replies" else 1 if fault == "invalid-version" else 5)
    assert client.preflights == 1
    raw = (faulty.guard.transcript_directory / "chat.jsonl").read_bytes()
    assert payload["chat"]["transcript"]["digest"] == digest(raw)
    assert payload["chat"]["transcript"]["lines"] == client.calls + (fault != "withdrawn")
    answered = 4 if fault in ("withdrawn", "replies") else client.calls
    assert payload["chat"]["answered_decisions"] == answered
    recognition = faulty.execute("recall")
    records = recognition["payload"]["recall"]
    if fault == "replies":
        assert all(r["status"] == "judged" for r in records)
        assert recognition["payload"]["stopped_by"] is None
    else:
        assert all(r["status"] == "not asked" for r in records)
        assert recognition["payload"]["stopped_by"] == cause
        assert client.preflights == 1


def test_chat_transport_resume_matches_uninterrupted_and_skips_finished_arms(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = runs("plain")
    expected = plain.execute("closed-loop")
    client = ChatFault("transport", at=45)
    resumed = runs("resumed", chat_client=client)
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
    assert completed == expected
    assert len(requested) == 440 - 44
    assert (fresh.guard.transcript_directory / "chat.jsonl").read_bytes().startswith(before)
    assert band_logs(fresh) == band_logs(plain)


@pytest.mark.parametrize("fault", ["VERSION", "CAP", "WITHDRAWN", "transport", "other-stop"])
def test_decision_faults_end_or_interrupt_the_whole_run(runs: Factory, fault: str) -> None:
    class Fault(clients.StubClient):
        calls = 0

        def ask(self, request: clients.Request) -> clients.Response:
            self.calls += 1
            if self.calls == 5:
                if fault == "WITHDRAWN":
                    raise clients.TerminalStop(clients.StopReason.WITHDRAWN)
                if fault == "transport":
                    raise clients.StepFailure("bounded transport failed")
                if fault == "other-stop":
                    raise clients.TerminalStop(clients.StopReason.CALIBRATION)
            response = super().ask(request)
            return replace(response, version="another version") if fault == "VERSION" else response

    client = Fault("decision")
    runner = runs(client=client, cap=Decimal(0) if fault == "CAP" else None)
    if fault in ("transport", "other-stop"):
        with pytest.raises(clients.TerminalStop if fault == "other-stop" else clients.StepFailure):
            runner.execute("closed-loop")
        assert state(runner)["status"] == "interrupted"
        completed = runner.execute("closed-loop")
        plain = runs("plain")
        assert completed == plain.execute("closed-loop")
    else:
        completed = runner.execute("closed-loop")
        assert completed["status"] == "terminal"
        assert completed["payload"]["outcome"] == {
            "H2": "no verdict",
            "H3": "no verdict",
            "reason": fault,
        }
        assert completed["payload"]["chat"]["calls"] == 0
        assert client.calls == (0 if fault == "CAP" else 1 if fault == "VERSION" else 5)
        for step in ("calibrate", "closed-loop", "recall"):
            with pytest.raises(clients.GuardRefusal, match="terminal"):
                runner.execute(step)


def test_canonical_requests_are_paid_once(runs: Factory) -> None:
    class Count(clients.StubClient):
        def __init__(self, family: clients.Family) -> None:
            super().__init__(family)
            self.seen: list[bytes] = []

        def ask(self, request: clients.Request) -> clients.Response:
            self.seen.append(clients.canonical(request))
            return super().ask(request)

    client = Count("decision")
    runner = runs(client=client, simulation_groups=2)
    runner.execute("closed-loop")
    assert len(client.seen) == len(set(client.seen)) < 880
    assert runner.steps.decision is not None
    transcript = runner.steps.decision.transcript
    simulation_entries = list(transcript.entries.values())[1024:]
    assert len(simulation_entries) == len(client.seen)
    for entry in simulation_entries:
        document = clients.obj(entry.request)
        assert document["decision"] == digest(
            json.dumps(
                document["request"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        )


@pytest.mark.parametrize("threads", [1, 4])
def test_worker_count_preserves_every_byte_and_state(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, threads: int
) -> None:
    monkeypatch.setattr(simulation, "CHAT_THREADS", 1)
    plain = runs("plain", simulation_groups=2)
    expected = plain.execute("closed-loop")
    monkeypatch.setattr(simulation, "CHAT_THREADS", threads)
    if threads > 1:
        original_pool = simulation.ThreadPoolExecutor

        class ReversedPool:
            def __init__(self, max_workers: int) -> None:
                self.pool = original_pool(max_workers=max_workers)

            def __enter__(self) -> "ReversedPool":
                return self

            def __exit__(self, *args: Any) -> None:
                self.pool.shutdown()

            def map(
                self, function: Callable[[int], kernel.Run], numbers: Iterable[int]
            ) -> Iterator[kernel.Run]:
                order = list(numbers)
                futures = {n: self.pool.submit(function, n) for n in reversed(order)}
                return (futures[n].result() for n in order)

        monkeypatch.setattr(simulation, "ThreadPoolExecutor", ReversedPool)
    other = runs("other", simulation_groups=2)
    assert other.execute("closed-loop") == expected
    for name in expected["payload"]["digests"]:
        assert (other.guard.storage / name).read_bytes() == (
            plain.guard.storage / name
        ).read_bytes()
    assert (other.guard.transcript_directory / "chat.jsonl").read_bytes() == (
        plain.guard.transcript_directory / "chat.jsonl"
    ).read_bytes()


@pytest.mark.parametrize("fault", ["probability", "summary", "saved-log", "replay"])
def test_nonfinite_and_replay_faults_are_not_silently_scored(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    runner = runs(chat_client=ChatFault("transport"))

    def nonfinite(*args: Any) -> score.Floats:
        return np.full(5, np.nan if fault == "probability" else np.inf)

    if fault == "probability":
        monkeypatch.setattr(prompted.Recalibration, "apply", nonfinite)
    elif fault == "summary":
        monkeypatch.setattr(score.compare, "summaries", nonfinite)
    elif fault == "replay":

        def failed(*args: Any, **kwargs: Any) -> Any:
            raise chat.Unusable(chat.Cause.REPLAY)

        monkeypatch.setattr(chat, "replay", failed)
        runner = runs("replay")
        completed = runner.execute("closed-loop")
        assert completed["status"] == "completed"
        assert completed["payload"]["chat"]["cause"] == "REPLAY"
        assert "chat" not in completed["payload"]["simulations"]
        return
    else:
        with pytest.raises(chat.TransportInterruption):
            runner.execute("closed-loop")
        name = next(iter(state(runner)["payload"]["digests"]))
        (runner.guard.storage / name).write_bytes(b"damaged")
    with pytest.raises((ValueError, clients.GuardRefusal)):
        runner.execute("closed-loop")
    interrupted = state(runner)
    assert interrupted["status"] == "interrupted"
    assert "outcome" not in interrupted["payload"]


@pytest.mark.parametrize("fault", ["version", "invalid-version", "cap", "withdrawn", "transport"])
def test_recall_conditions_labels_full_replies_and_resume(runs: Factory, fault: str) -> None:
    # Use a completed, unusable chat arm so the recall-only cases stay small.
    runner = runs(chat_client=ChatFault("replies", at=1))
    runner.execute("closed-loop")

    class RecallFault(ChatFault):
        def ask(self, text: str) -> chat.Response:
            response = super().ask(text)
            return replace(response, text="WANG asymmetric public goods\nFull retained reply")

    client = RecallFault(fault, at=3 if fault != "invalid-version" else 1)
    recognizer = run.Run(
        runner.guard.root,
        stub=True,
        storage=runner.guard.storage,
        chat_client=client,
        simulation_groups=1,
    )
    if fault == "transport":
        with pytest.raises(chat.TransportInterruption):
            recognizer.execute("recall")
        interrupted = state(recognizer, "recall")
        assert interrupted["status"] == "interrupted"
        assert [r["status"] for r in interrupted["payload"]["recall"]] == ["judged"] * 2 + [
            "not asked"
        ] * 6
        completed = recognizer.execute("recall")
        assert client.calls == 9
        assert completed["payload"]["stopped_by"] is None
        assert all(r["status"] == "judged" for r in completed["payload"]["recall"])
    else:
        completed = recognizer.execute("recall")
        cause = {"withdrawn": "WITHDRAWN", "cap": "CAP"}.get(fault, "VERSION")
        assert completed["payload"]["stopped_by"] == cause
        assert completed["payload"]["chat"]["cause"] == ("REPLIES")
        assert client.calls == (1 if fault == "invalid-version" else 3)
        labels = [r["status"] for r in completed["payload"]["recall"]]
        expected = (
            ["not judged: version"] + ["not asked"] * 7
            if fault == "invalid-version"
            else ["judged"] * 2 + ["not asked"] * 6
            if fault == "withdrawn"
            else ["judged"] * 2 + ["not judged: version"] + ["not asked"] * 5
            if fault == "version"
            else ["judged"] * 3 + ["not asked"] * 5
        )
        assert labels == expected
    assert completed["status"] == "completed"
    assert "outcome" not in completed["payload"]
    transcript = chat.Transcript(
        recognizer.guard.transcript_directory,
        chat.RunIdentity(completed["registered_commit"], "stub"),
    )
    assert completed["payload"]["recall"] == [asdict(r) for r in chat.read_recall(transcript)]
    for record in completed["payload"]["recall"]:
        assert record["flagged"] == (record["status"] == "judged")
        if record["response"]:
            assert record["response"]["text"] == "WANG asymmetric public goods\nFull retained reply"
            assert record["line"] is not None and len(record["entry_digest"]) == 64


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
    assert final == expected


def test_completed_chat_logs_have_separate_digests(runs: Factory) -> None:
    # Inspect the required data contract without obtaining a data capability.
    runner = runs()
    completed = runner.execute("closed-loop")
    assert "chat_logs" in completed["payload"]
    outputs = cast(
        dict[str, str], completed["payload"]["digests"] | completed["payload"]["chat_logs"]
    )
    assert isinstance(outputs, dict) and outputs
    for name, expected in outputs.items():
        path = (runner.guard.storage / name).resolve()
        assert path.is_relative_to(runner.guard.storage)
        assert isinstance(expected, str) and len(expected) == 64
        assert path.is_file() and digest(path.read_bytes()) == expected
    chat_record = completed["payload"]["chat"]["transcript"]
    transcript = chat.Transcript(
        runner.guard.transcript_directory, chat.RunIdentity(completed["registered_commit"], "stub")
    )
    verified = chat.replay_transcript(transcript, chat_record["digest"])
    assert chat_record["lines"] == len(transcript.entries) + 1
    for name in completed["payload"]["chat_logs"]:
        raw = (runner.guard.storage / name).read_bytes()
        assert chat.replay(raw, verified).bytes == raw


def test_arm_and_chat_game_checkpoints_precede_next_work(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs(chat_client=ChatFault("transport", at=45))
    original = score.simulate
    checked: list[str] = []

    def simulate(agent: kernel.Agent, *args: Any, **kwargs: Any) -> score.Simulation:
        if isinstance(agent, kernel.Coded) and agent.law.candidate.twin:
            assert "zoon" in state(runner)["payload"]["simulations"]
            checked.append("zoon")
        if isinstance(agent, simulation.Prompted):
            assert "alone" in state(runner)["payload"]["simulations"]
            checked.append("alone")
        return original(agent, *args, **kwargs)

    monkeypatch.setattr(score, "simulate", simulate)
    with pytest.raises(chat.TransportInterruption):
        runner.execute("closed-loop")
    assert checked == ["zoon", "alone"]
    assert "chat_logs" not in state(runner)["payload"]
    original_play = chat.play
    played: list[str] = []

    def play(game: chat.Game, session: chat.Session, group: str) -> kernel.Run:
        played.append(group)
        return original_play(game, session, group)

    monkeypatch.setattr(chat, "play", play)
    fresh = run.Run(runner.guard.root, stub=True, storage=runner.guard.storage, simulation_groups=1)
    assert fresh.execute("closed-loop")["status"] == "completed"
    assert len(played) == 8


def test_operator_cli_records_reason_and_completes_the_step(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = runs(chat_client=ChatFault("transport"))

    def local(**kwargs: Any) -> run.Run:
        assert kwargs["stub"]
        return runner

    monkeypatch.setattr(run, "Run", local)
    assert run.main(["closed-loop", "--end-chat", "Requests remain unavailable", "--stub"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "[interaction stub] closed-loop = completed" in output.out
    assert "Requests remain unavailable" in output.out
    assert state(runner)["status"] == "completed"
    assert state(runner)["payload"]["chat_endings"][0]["reason"] == "Requests remain unavailable"


def test_log_storage_failure_interrupts_and_recovers(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs()

    def refuse(*args: Any) -> None:
        raise OSError("log storage unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(simulation, "write", refuse)
        with pytest.raises(OSError, match="log storage"):
            runner.execute("closed-loop")
    assert state(runner)["status"] == "interrupted"
    plain = runs("plain")
    assert runner.execute("closed-loop") == plain.execute("closed-loop")


def ended_closed_loop(runs: Factory) -> run.Run:
    runner = runs(chat_client=ChatFault("transport", at=45))
    with pytest.raises(chat.TransportInterruption):
        runner.execute("closed-loop")
    runner.steps.chat_client = ChatFault("transport", at=1)
    runner.execute("closed-loop", end_chat="Requests unavailable")
    return runner


def test_each_step_can_end_chat_without_losing_the_first_event(runs: Factory) -> None:
    runner = ended_closed_loop(runs)
    before = state(runner)
    first = before["payload"]["chat_endings"][0]
    runner.steps.chat_client = ChatFault("transport", at=3)
    with pytest.raises(chat.TransportInterruption):
        runner.execute("recall")
    runner.steps.chat_client = ChatFault("transport", at=1)
    completed = runner.execute("recall", end_chat="Recall unavailable")
    events = completed["payload"]["chat_endings"]
    assert events == [
        first,
        {
            "cause": "OPERATOR",
            "step": "recall",
            "reason": "Recall unavailable",
            "answered_decisions": 44,
        },
    ]
    assert first == {
        "cause": "OPERATOR",
        "step": "closed-loop",
        "reason": "Requests unavailable",
        "answered_decisions": 44,
    }
    assert completed["payload"]["chat"]["endings"] == events
    assert completed["payload"]["chat"]["cause"] == "OPERATOR"
    assert state(runner) == before
    assert [r["status"] for r in completed["payload"]["recall"]] == ["judged"] * 2 + [
        "not asked"
    ] * 6


def test_recall_keeps_the_first_cause_and_operator_details(runs: Factory) -> None:
    runner = ended_closed_loop(runs)
    first = state(runner)["payload"]["chat_endings"][0]
    runner.steps.chat_client = ChatFault("withdrawn", at=1)
    completed = runner.execute("recall")
    assert completed["payload"]["chat_endings"] == [first, {"cause": "WITHDRAWN", "step": "recall"}]
    assert completed["payload"]["chat"]["cause"] == "OPERATOR"


def test_ending_leaves_chat_games_unscored(runs: Factory) -> None:
    runner = ended_closed_loop(runs)
    assert "chat_logs" not in state(runner)["payload"]
    assert "chat" not in state(runner)["payload"]["simulations"]
    assert state(runner)["payload"]["chat"]["calls"] == 44


@pytest.mark.parametrize("damage", ["none", "unreadable", "partial"])
def test_actual_append_failure_saves_interrupted(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    runner = runs()
    original_open, original_read = Path.open, Path.read_bytes
    failed = False
    saved: dict[str, Any] = {}

    def opening(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        nonlocal failed, saved
        if failed and damage == "unreadable" and path.name == "chat.jsonl":
            raise OSError("read unavailable")
        if path.name == "chat.jsonl" and mode == "ab":
            saved = state(runner)
            failed = True
            if damage == "unreadable":
                with original_open(path, "wb"):
                    pass
            if damage == "partial":
                with original_open(path, "ab") as handle:
                    handle.write(b'{"incomplete":')
            raise OSError("append unavailable")
        return original_open(path, mode, *args, **kwargs)

    def reading(path: Path) -> bytes:
        if failed and damage == "unreadable" and path.name == "chat.jsonl":
            raise OSError("read unavailable")
        return original_read(path)

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(Path, "read_bytes", reading)
    with pytest.raises((chat.StorageInterruption, clients.GuardRefusal, chat.GuardRefusal)):
        runner.execute("closed-loop")
    assert failed
    interrupted = state(runner)
    assert interrupted["status"] == ("interrupted" if damage == "none" else "running")
    assert interrupted["payload"]["chat"] == saved["payload"]["chat"]
    assert interrupted["transcripts"]["chat"] == saved["transcripts"]["chat"]


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_resume_with_every_answer_skips_preflight(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    runner = runs()
    if step == "recall":
        runner.execute("closed-loop")
    original = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "completed":
            raise OSError("completion unavailable")
        return original(status, reason)

    with monkeypatch.context() as patch:
        patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError, match="completion unavailable"):
            runner.execute(step)

    class Offline(chat.StubClient):
        def preflight(self) -> None:
            pytest.fail("no unanswered request remains")

    runner.steps.chat_client = Offline()
    assert runner.execute(step)["status"] == "completed"


@pytest.mark.parametrize(
    "initial", ["usable", "version", "replies", "cap", "withdrawn", "replay", "operator"]
)
@pytest.mark.parametrize("later", ["none", "version", "cap", "withdrawn", "operator"])
def test_chat_cause_and_events_across_steps(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, initial: str, later: str
) -> None:
    # VERSION, CAP and WITHDRAWN prevent recall requests, so only 'none' is reachable.
    if initial in ("version", "cap", "withdrawn") and later != "none":
        pytest.skip("the earlier version, cap or withdrawal prevents recall requests")
    runner = runs(chat_client=ChatFault("transport" if initial == "operator" else initial, at=5))
    with monkeypatch.context() as patch:
        if initial == "replay":

            def failed(*args: Any, **kwargs: Any) -> Any:
                raise chat.Unusable(chat.Cause.REPLAY)

            patch.setattr(chat, "replay", failed)
        closed = runner.execute(
            "closed-loop", end_chat="Requests unavailable" if initial == "operator" else None
        )
    path = runner.guard.directory / "closed-loop.json"
    before = path.read_bytes()
    runner.steps.chat_client = ChatFault("transport" if later == "operator" else later, at=1)
    recalled = runner.execute(
        "recall", end_chat="Recall unavailable" if later == "operator" else None
    )
    first, last = closed["payload"]["chat"], recalled["payload"]["chat"]
    assert path.read_bytes() == before
    assert first["cause"] == (None if initial == "usable" else initial.upper())
    assert (first["status"] == "unusable") == (first["cause"] is not None)
    if first["cause"]:
        assert last["cause"] == first["cause"]
    else:
        assert last["cause"] == (None if later == "none" else later.upper())
    assert last["endings"][: len(first["endings"])] == first["endings"]
    assert (last["status"] == "unusable") == (last["cause"] is not None)
    assert recalled["status"] == "completed"


@pytest.mark.parametrize("stage", ["decision", "chat", "save"])
@pytest.mark.parametrize("started", [False, True])
def test_initialization_failure_records_current_failure(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, stage: str, started: bool
) -> None:
    runner = runs(chat_client=ChatFault("transport"))
    if started:
        with pytest.raises(chat.TransportInterruption):
            runner.execute("closed-loop")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("storage unavailable")

    with monkeypatch.context() as patch:
        if stage == "decision":
            patch.setattr(run.decisions, "Session", fail)
        elif stage == "chat":
            patch.setattr(runner.steps, "open_chat", fail)
        else:
            original = runner.steps.save
            calls = 0

            def save(*args: Any, **kwargs: Any) -> Any:
                nonlocal calls
                calls += 1
                return fail() if calls == 1 else original(*args, **kwargs)

            patch.setattr(runner.steps, "save", save)
        with pytest.raises(OSError, match="storage unavailable"):
            runner.execute("closed-loop")
    assert state(runner)["status"] == "interrupted"
    assert state(runner)["reason"] == "OSError"


def test_later_append_failure_keeps_checkpoint_line_count(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs()
    original = os.fsync
    count = 0
    path = runner.guard.transcript_directory / "chat.jsonl"

    def fsync(fd: int) -> None:
        nonlocal count
        if path.exists() and os.fstat(fd).st_ino == path.stat().st_ino:
            count += 1
            if count == 150:
                raise OSError("sync unavailable")
        original(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(chat.StorageInterruption):
        runner.execute("closed-loop")
    saved = state(runner)
    assert saved["payload"]["chat"]["transcript"]["lines"] == 151
    assert len(saved["transcripts"]["chat"]["keys"]) == 150


@pytest.mark.parametrize("fault", ["decision", "incomplete"])
def test_cli_preserves_storage_and_incomplete_line_messages(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fault: str
) -> None:
    runner = runs()
    if fault == "incomplete":
        runner.steps.chat_client = ChatFault("transport")
        with pytest.raises(chat.TransportInterruption):
            runner.execute("closed-loop")
        with (runner.guard.transcript_directory / "chat.jsonl").open("ab") as handle:
            handle.write(b'{"partial":')
    else:
        original = Path.open

        def opening(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if path.name == "decision.jsonl" and mode == "a":
                raise OSError("storage unavailable")
            return original(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", opening)

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    assert run.main(["closed-loop", "--stub"]) == 1
    output = capsys.readouterr()
    assert "Traceback" not in output.err
    assert (
        "StepFailure: could not persist decision transcript"
        if fault == "decision"
        else "incomplete transcript line; cut the file back to"
    ) in output.err


def test_simulation_store_contains_paths_and_validates_chat_summaries(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs()
    store = simulation.Store(runner.guard, {}, lambda: None)
    with pytest.raises(clients.GuardRefusal, match="outside run storage"):
        store.read("../outside.jsonl", "unused")

    def summaries(*args: Any, **kwargs: Any) -> Any:
        return np.full(5, np.nan)

    monkeypatch.setattr(simulation.compare, "summaries", summaries)
    with pytest.raises(ValueError):
        store.record("chat", "2-FE", (kernel.Run((), ()),))


@pytest.mark.parametrize("fault", ["version", "replies", "cap", "withdrawn"])
def test_unusable_leaves_chat_games_unscored(runs: Factory, fault: str) -> None:
    runner = runs(chat_client=ChatFault(fault, at=150))
    result = runner.execute("closed-loop")
    assert result["status"] == "completed"
    assert "chat_logs" not in result["payload"]
    assert "chat" not in result["payload"]["simulations"]
