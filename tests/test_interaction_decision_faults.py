"""Paid receipts, terminal replay and storage faults stop further dispatch."""

import json
import threading
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.interaction_run_support import repository
from zoon.interaction import decisions, prompted, run
from zoon.interaction.guard import checkpoint
from zoon.pilot import clients, prices


class SDK:
    def __init__(self, reply: Callable[[int], Any]) -> None:
        self.reply = reply
        self.calls = 0
        self.threads: list[int] = []

    def Choice(self, **kwargs: Any) -> None:
        pass

    def TypeSafeClient(self, **kwargs: Any) -> Any:
        return self

    def system_one(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.threads.append(threading.get_ident())
        return self.reply(self.calls)


def reply(*, version: str = prices.DECISION.model, tokens: int = 12) -> Any:
    return SimpleNamespace(
        model=version,
        usage=SimpleNamespace(input_tokens=tokens, output_tokens=1),
        answers={"q": SimpleNamespace(probabilities=dict.fromkeys(prompted.ANSWERS, 0.2))},
    )


def session(
    directory: Path, client: clients.Client, *, cap: Decimal = Decimal(17)
) -> decisions.Session:
    return decisions.Session(
        directory, client, cap, {"mode": "stub", "registered_commit": "a" * 40}
    )


def requests() -> list[decisions.Request]:
    request = prompted.fixtures()["2-AI-type-0-round-1.json"]
    return [("one", request), ("two", request), ("three", request)]


@pytest.mark.parametrize("interrupt", [False, True])
def test_live_version_receipt_is_durable_before_any_later_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    root = repository(tmp_path / "repo")

    def answer(n: int) -> Any:
        if n > 1:
            raise KeyboardInterrupt
        return reply(version="another-version", tokens=1_000_000_000)

    sdk = SDK(answer)
    original = decisions.append

    def received(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        if interrupt:
            raise KeyboardInterrupt

    monkeypatch.setattr(decisions, "append", received)
    runner = run.Run(root, client=decisions.LiveClient(sdk))
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            runner.execute()
        state = json.loads((runner.guard.directory / "calibrate.json").read_bytes())
        assert state["status"] == "interrupted"
    else:
        state = runner.execute()
        assert state["status"] == "terminal"
        assert state["payload"]["outcome"]["reason"] == "VERSION"
    assert sdk.calls == 1
    assert sdk.threads == [threading.get_ident()]
    assert runner.session is not None
    assert len(runner.session.transcript.entries) == 1
    ledger = clients.Transcript(
        runner.guard.transcript_directory, "decision", runner.guard.identity
    )
    assert checkpoint(ledger) == checkpoint(runner.session.transcript)
    assert next(iter(ledger.entries.values())).response.version == "another-version"
    assert ledger.spent > runner.session.cap
    if interrupt:
        monkeypatch.setattr(decisions, "append", original)
        resumed = run.Run(root, client=decisions.LiveClient(sdk)).execute()
        assert resumed["status"] == "terminal"
        assert resumed["payload"]["outcome"]["reason"] == "VERSION"
        assert sdk.calls == 1


@pytest.mark.parametrize(
    "reason", [clients.StopReason.VERSION, clients.StopReason.WITHDRAWN, clients.StopReason.CAP]
)
def test_dispatch_sends_nothing_after_terminal_result(
    tmp_path: Path, reason: clients.StopReason
) -> None:
    class Stop(clients.StubClient):
        calls = 0

        def ask(self, request: clients.Request) -> clients.Response:
            self.calls += 1
            if reason == clients.StopReason.VERSION:
                return replace(super().ask(request), version="another-version")
            if reason == clients.StopReason.CAP:
                return replace(super().ask(request), input_tokens=1_000_000_000)
            raise clients.TerminalStop(reason)

    client = Stop("decision")
    with pytest.raises(clients.TerminalStop) as stopped:
        session(tmp_path, client).ask_many(requests())
    assert stopped.value.reason == reason
    assert client.calls == 1


@pytest.mark.parametrize("headroom", [False, True])
def test_injected_client_admission_uses_recorded_cost(tmp_path: Path, headroom: bool) -> None:
    class Costly(clients.StubClient):
        calls = 0

        def ask(self, request: clients.Request) -> clients.Response:
            self.calls += 1
            return replace(super().ask(request), input_tokens=1_000_000)

    client = Costly("decision")
    cap = prices.DECISION.dollars(1_000_000, 1)
    if headroom:
        cap += clients.estimate("decision", requests()[0][1]) / 2
    ledger = session(tmp_path, client, cap=cap)
    with pytest.raises(clients.TerminalStop) as stopped:
        ledger.ask_many(requests()[:2])
    assert stopped.value.reason == clients.StopReason.CAP
    assert client.calls == len(ledger.transcript.entries) == 1
    assert ledger.transcript.spent == prices.DECISION.dollars(1_000_000, 1)


def test_authentication_failure_stops_before_the_next_request(tmp_path: Path) -> None:
    class AuthenticationError(Exception):
        status_code = 401

    def denied(n: int) -> Any:
        raise AuthenticationError

    sdk = SDK(denied)
    with pytest.raises(clients.StepFailure, match="authentication"):
        session(tmp_path, decisions.LiveClient(sdk)).ask_many(requests())
    assert sdk.calls == 1


def test_first_request_failure_is_raised_before_another_failure(tmp_path: Path) -> None:
    failures = [clients.StepFailure("first request failed"), ValueError("second request failed")]

    class Failing(clients.StubClient):
        calls = 0

        def ask(self, request: clients.Request) -> clients.Response:
            error = failures[self.calls]
            self.calls += 1
            raise error

    client = Failing("decision")
    with pytest.raises(clients.StepFailure) as failed:
        session(tmp_path, client).ask_many(requests()[:2])
    assert failed.value is failures[0]
    assert client.calls == 1


def test_paid_replies_without_answers_are_all_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-connection-value")
    pauses: list[float] = []
    bounded = clients.retry

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return bounded(call, pause=pauses.append)

    monkeypatch.setattr(clients, "retry", retry)
    result = SimpleNamespace(
        model=prices.DECISION.model,
        usage=SimpleNamespace(input_tokens=12, output_tokens=1),
    )
    sdk = SDK(lambda n: result)
    ledger = session(tmp_path, decisions.LiveClient(sdk))
    with pytest.raises(clients.StepFailure, match="bounded retries"):
        ledger.ask_many(requests())
    assert sdk.calls == len(ledger.transcript.entries) == 4
    assert pauses == [1, 2, 4]
    assert ledger.transcript.spent == 4 * prices.DECISION.dollars(12, 1)
    assert all(not entry.response.probabilities for entry in ledger.transcript.entries.values())
    reloaded = clients.Transcript(tmp_path, "decision", ledger.transcript.identity)
    assert checkpoint(reloaded) == checkpoint(ledger.transcript)


def test_live_withdrawal_stops_at_the_first_established_withdrawal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Missing(Exception):
        status_code = 404

    def missing(n: int) -> Any:
        raise Missing

    sdk = SDK(missing)
    bounded = clients.retry
    pauses: list[float] = []

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return bounded(call, pause=pauses.append)

    monkeypatch.setattr(clients, "retry", retry)
    with pytest.raises(clients.TerminalStop) as stopped:
        session(tmp_path, decisions.LiveClient(sdk)).ask_many(requests())
    assert stopped.value.reason == clients.StopReason.WITHDRAWN
    assert sdk.calls == 4 and pauses == [1, 2, 4]


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_last_receipt_above_cap_is_terminal_and_replays_without_calls(
    tmp_path: Path, live: bool, count: int
) -> None:
    class Costly(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            return replace(super().ask(request), input_tokens=1_000_000_000)

    sdk = SDK(lambda n: reply(tokens=1_000_000_000))
    client = decisions.LiveClient(sdk) if live else Costly("decision")
    ledger = session(tmp_path, client)
    with pytest.raises(clients.TerminalStop) as stopped:
        ledger.ask_many(requests()[:count])
    assert stopped.value.reason == clients.StopReason.CAP
    assert len(ledger.transcript.entries) == 1
    # Existing terminal evidence must be checked even with no pending request.
    with pytest.raises(clients.TerminalStop) as resumed:
        session(tmp_path, client).ask_many([])
    assert resumed.value.reason == clients.StopReason.CAP
    if live:
        assert sdk.calls == 1


def test_session_requires_decision_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decision client"):
        session(tmp_path, clients.StubClient("second"))


def test_injected_client_responses_are_validated(tmp_path: Path) -> None:
    class Invalid(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            return replace(super().ask(request), probabilities={"unknown": 1.0})

    with pytest.raises(clients.StepFailure, match="probabilities"):
        session(tmp_path, Invalid("decision")).ask_many(requests())


def test_storage_fault_is_not_retried_and_resumes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path / "repo")
    sdk = SDK(lambda n: reply())
    bounded = clients.retry
    pauses: list[float] = []

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return bounded(call, pause=pauses.append)

    monkeypatch.setattr(clients, "retry", retry)
    runner = run.Run(root, client=decisions.LiveClient(sdk))
    original: Any = Path.open

    def unwritable(path: Path, *args: Any, **kwargs: Any) -> Any:
        if (
            path == runner.guard.transcript_directory / "decision.jsonl"
            and args
            and args[0] == "a"
            and sdk.calls > 3
        ):
            raise OSError("storage unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unwritable)
    with pytest.raises(clients.StepFailure, match=r"persist decision transcript.*writable storage"):
        runner.execute()
    assert sdk.calls == 4 and pauses == []
    state = json.loads((runner.guard.directory / "calibrate.json").read_bytes())
    assert state["status"] == "interrupted" and state["reason"] == "StorageFault"
    assert state["payload"]["calls"] == 3
    monkeypatch.setattr(Path, "open", original)
    final = run.Run(root, client=decisions.LiveClient(sdk)).execute()
    assert final["status"] == "completed"
    assert final["payload"]["calls"] == 1024
    assert sdk.calls == 1025


def test_connection_metadata_check_precedes_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "synthetic-connection-value"
    monkeypatch.setenv("TYPESAFE_API_KEY", marker)
    for shape in ("answers", "version", "no version"):
        response = reply()
        response.answers["private"] = marker
        if shape == "version":
            response.model = marker
        elif shape == "no version":
            response.model = None
        sdk = SDK(lambda n, response=response: response)
        root = repository(tmp_path / shape.replace(" ", "-"))
        runner = run.Run(root, client=decisions.LiveClient(sdk))
        with pytest.raises(clients.StepFailure, match="no transcript was written"):
            runner.execute()
        state = json.loads((runner.guard.directory / "calibrate.json").read_bytes())
        assert state["status"] == "interrupted" and "outcome" not in state
        assert sdk.calls == 1
        ledger = runner.guard.transcript_directory / "decision.jsonl"
        assert not ledger.exists() or ledger.read_bytes().count(b"\n") <= 1


def test_each_live_receipt_is_synced_before_the_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    original = decisions.os.fsync

    def sync(descriptor: int) -> None:
        synced.append(descriptor)
        original(descriptor)

    monkeypatch.setattr(decisions.os, "fsync", sync)

    def answer(n: int) -> Any:
        assert len(synced) == n - 1
        return reply()

    sdk = SDK(answer)
    ledger = session(tmp_path, decisions.LiveClient(sdk))
    ledger.ask_many(requests())
    assert len(synced) == sdk.calls == len(ledger.transcript.entries) == 3


def test_version_is_recorded_before_returning_the_sdk_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-connection-value")

    class Result:
        def __init__(self) -> None:
            self.reads = 0
            self.usage = SimpleNamespace(input_tokens=12, output_tokens=1)
            self.answers: dict[str, Any] = {}

        @property
        def model(self) -> str:
            self.reads += 1
            if self.reads > 1:
                raise KeyboardInterrupt
            return "another-version"

    result = Result()
    sdk = SDK(lambda n: result)
    ledger = session(tmp_path, decisions.LiveClient(sdk))
    try:
        with pytest.raises(clients.TerminalStop) as stopped:
            ledger.ask_many(requests())
    except KeyboardInterrupt:
        pytest.fail("paid version reply was not recorded before returning from the SDK")
    assert stopped.value.reason == clients.StopReason.VERSION
    assert len(ledger.transcript.entries) == sdk.calls == result.reads == 1


@pytest.mark.parametrize(
    "usage,incoming,outgoing",
    [
        (None, 0, 0),
        (SimpleNamespace(input_tokens=12, output_tokens="bad"), 12, 0),
        (SimpleNamespace(input_tokens=True, output_tokens=3), 0, 3),
        (SimpleNamespace(input_tokens=-1, output_tokens=-1), 0, 0),
    ],
)
def test_version_change_is_terminal_even_without_valid_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, usage: Any, incoming: int, outgoing: int
) -> None:
    bounded = clients.retry

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return bounded(call, pause=lambda seconds: None)

    monkeypatch.setattr(clients, "retry", retry)
    root = repository(tmp_path / "repo")
    sdk = SDK(lambda n: SimpleNamespace(model="another-version", usage=usage))
    runner = run.Run(root, client=decisions.LiveClient(sdk))
    state = runner.execute()
    assert state["status"] == "terminal"
    assert state["payload"]["outcome"]["reason"] == "VERSION"
    assert state["payload"]["calls"] == sdk.calls == 1
    assert state["payload"]["input_tokens"] == incoming
    assert state["payload"]["output_tokens"] == outgoing
    assert runner.session is not None
    assert len(runner.session.transcript.entries) == 1


def test_normalised_label_collision_is_unusable_before_a_good_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zoon.pilot import prompts

    request = prompts.Decision(
        "state", "question", (prompts.Answer("1", "first"), prompts.Answer("2", "second"))
    )
    bad = {1: 0.8, "1": 0.5, "2": 0.5}
    good = {"1": 0.25, "2": 0.75}
    sdk = SDK(
        lambda n: SimpleNamespace(
            model=prices.DECISION.model,
            usage=SimpleNamespace(input_tokens=12, output_tokens=1),
            answers={"q": SimpleNamespace(probabilities=bad if n == 1 else good)},
        )
    )
    bounded = clients.retry

    def retry(call: Callable[[], clients.Response]) -> clients.Response:
        return bounded(call, pause=lambda seconds: None)

    monkeypatch.setattr(clients, "retry", retry)
    ledger = session(tmp_path, decisions.LiveClient(sdk))
    responses = ledger.ask_many([("decision", request)])
    assert responses[0].probabilities == good
    assert sdk.calls == len(ledger.transcript.entries) == 2
    first = next(iter(ledger.transcript.entries.values()))
    assert ":failed:1" in str(first.request)
    assert not first.response.probabilities
