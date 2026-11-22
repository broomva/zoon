"""The chat arm with made-up histories and injected, offline transports only."""

import hashlib
import json
import os
import socket
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from zoon.interaction import chat, chat_prices, kernel, prompted
from zoon.interaction.games import GAMES, ROUNDS

BINDING = chat.RunIdentity("a" * 40, "stub")
GROUP = "registered|2-AI|0"


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("external access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(chat.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(chat.subprocess, "run", forbidden)
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "connection-location-sentinel")
    raised: set[type[chat.Interruption]] = set()
    previous_trace = sys.gettrace()

    def trace(frame: Any, event: str, arg: Any) -> Any:
        if frame.f_globals.get("__name__") != chat.__name__:
            return None
        if event == "exception":
            error = cast(BaseException, arg[1])
            if isinstance(error, chat.Interruption):
                raised.add(type(error))
        return trace

    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(previous_trace)
    assert chat.Interruption not in raised


def response(text: str = "12", **kwargs: Any) -> chat.Response:
    return replace(chat.Response(text, chat.MODEL + "-001", 12, 2, 3), **kwargs)


class Replies:
    def __init__(self, replies: Sequence[chat.Response | Exception]) -> None:
        self.replies = iter(replies)
        self.requests: list[str] = []

    def ask(self, text: str) -> chat.Response:
        self.requests.append(text)
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return value


def session(tmp_path: Path, client: chat.Client, cap: Decimal = Decimal("1000")) -> chat.Session:
    return chat.Session(chat.Transcript(tmp_path, BINDING, read_only=False), client, cap)


def view(seat: int = 0) -> kernel.View:
    return kernel.View(GAMES["2-AI"], seat, ())


def key(seat: int = 0, attempt: int = 1) -> chat.Identity:
    return chat.Identity("2-AI", GROUP, 0, seat, attempt)


def test_exact_request_and_same_made_up_histories_as_prompted(tmp_path: Path) -> None:
    generated = chat.fixtures()
    source = prompted.fixtures()
    assert len(generated) == len(source) == 28
    directory = prompted.DIRECTORY / "chat-requests"
    digests = json.loads((directory / "sha256.json").read_text())
    assert set(digests) == set(generated)
    assert {p.name for p in directory.iterdir()} == {*generated, "sha256.json"}
    assert (directory / "sha256.json").read_bytes() == (
        json.dumps({name: chat.digest(text) for name, text in generated.items()}, indent=2) + "\n"
    ).encode()
    for name, text in generated.items():
        original = source[name.replace(".txt", ".json")]
        cell, kind, r = (
            name.split("-type-")[0],
            int(name.split("-type-")[1][0]),
            int(name.rsplit("-", 1)[1].removesuffix(".txt")),
        )
        game = GAMES[cell]
        seat = 0 if kind == 0 else 1 if game.size == 2 else 2
        history = original.state.split(f"The player you are asked about is Player {seat + 1}.\n")[1]
        expected = (
            prompted.FIXTURE_LABEL
            + prompted.instructions(game)
            + f"\nYou are Player {seat + 1}.\n\n"
            + history
            + f"\n\nHow many of your {game.endowments[seat]} points do you contribute "
            + f"to the common pool in round {r}? "
            + "Answer with one whole number of points and nothing else."
        )
        assert text == expected
        assert not text.endswith("\n")
        assert (directory / name).read_bytes() == text.encode()
        assert digests[name] == hashlib.sha256(text.encode()).hexdigest()
    assert chat.EMPTY_HISTORY in generated["2-AI-type-0-round-1.txt"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        (" \n\t", None),
        ("+1", None),
        ("-1", None),
        ("1.0", None),
        ("one", None),
        ("1 point", None),
        ("37", None),
        ("\uff11\uff12", None),
        ("١٢", None),
        ("12.", None),
        ("1 2", None),
        ("1\n2", None),
        ("0", 0),
        ("36", 36),
        ("00012", 12),
        (" \n12\t", 12),
        ("0" * 10000, 0),
        ("9" * 10000, None),
        ("2", 2),  # A digit-only reply cut off at the limit still passes.
        ("2 poi", None),
    ],
)
def test_reply_rule(text: str, expected: int | None) -> None:
    assert chat.points(text, 36) == expected


def test_reply_rule_uses_that_seats_endowment() -> None:
    assert chat.points("13", 12) is None
    assert chat.points("13", 36) == 13


def payload(**updates: Any) -> dict[str, Any]:
    return {
        "modelVersion": chat.MODEL + "-001",
        "usageMetadata": {
            "promptTokenCount": 19,
            "candidatesTokenCount": 2,
            "thoughtsTokenCount": 7,
        },
        "candidates": [
            {"finishReason": "STOP", "content": {"parts": [{"text": "1"}, {"text": "2"}]}},
            {"content": {"parts": [{"text": "do not use"}]}},
        ],
        **updates,
    }


def preflight_ok(*args: Any) -> tuple[int, dict[str, Any]]:
    return 200, {}


def fake_client(
    replies: Sequence[tuple[int, dict[str, Any]] | Exception],
) -> tuple[chat.VertexClient, list[Any], list[float]]:
    pending = iter(replies)
    calls: list[Any] = []
    pauses: list[float] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        calls.append((url, headers, json.loads(body)))
        result = next(pending)
        if isinstance(result, Exception):
            raise result
        return result

    client = chat.VertexClient(
        transport,
        token=lambda: "fake-bearer",
        pause=pauses.append,
        preflight_transport=preflight_ok,
    )
    client.preflight()
    return client, calls, pauses


def test_generate_content_wire_settings_and_thinking_billed_as_output() -> None:
    client, calls, pauses = fake_client([(200, payload())])
    result = client.ask(chat.request(view()))
    assert result == response("12", input_tokens=19, output_tokens=2, thinking_tokens=7)
    assert not pauses
    url, headers, body = calls[0]
    assert url.endswith(
        "/locations/global/publishers/google/models/gemini-4-carbon:generateContent"
    )
    assert headers["x-goog-user-project"] == os.environ["ZOON_VERTEX_PROJECT"]
    assert body == {
        "contents": [{"role": "user", "parts": [{"text": chat.request(view())}]}],
        "generationConfig": {
            "temperature": 1.0,
            "thinkingConfig": {"thinkingLevel": "low"},
            "maxOutputTokens": 2048,
        },
        "safetySettings": chat.SAFETY_SETTINGS,
    }
    assert [
        {"category": "HARM_CATEGORY_" + category, "threshold": "OFF"}
        for category in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")
    ] == chat.SAFETY_SETTINGS
    assert chat.cost(result) == Decimal("0.000256")
    assert chat.cost(result, chat_prices.LOW) == Decimal("0.000128")


@pytest.mark.parametrize(
    "change",
    cast(
        list[dict[str, Any]],
        [
            {"candidates": []},
            {"candidates": [{}]},
            {"promptFeedback": {"blockReason": "SAFETY"}},
            {"promptFeedback": {"safetyRatings": [{"blocked": True}]}},
            {"candidates": [{"blockReason": "SAFETY", "content": {"parts": [{"text": "12"}]}}]},
            {"candidates": [{"finishReason": "SAFETY", "content": {"parts": [{"text": "12"}]}}]},
            {
                "candidates": [
                    {"safetyRatings": [{"blocked": True}], "content": {"parts": [{"text": "12"}]}}
                ]
            },
        ],
    ),
)
def test_empty_or_blocked_reply_keeps_usage(change: dict[str, Any]) -> None:
    client, _, _ = fake_client([(200, payload(**change))])
    assert client.ask("question") == response(
        "", input_tokens=19, output_tokens=2, thinking_tokens=7
    )


def test_cut_off_reply_obeys_the_same_rule() -> None:
    client, _, _ = fake_client(
        [
            (
                200,
                payload(
                    candidates=[
                        {"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "2"}]}}
                    ]
                ),
            )
        ]
    )
    assert chat.points(client.ask("question").text, 36) == 2


@pytest.mark.parametrize("failure", [TimeoutError, ConnectionError, chat.TransientError, OSError])
def test_bounded_transport_retries(failure: type[Exception]) -> None:
    client, calls, pauses = fake_client([failure("private detail")] * 4)
    with pytest.raises(chat.Interruption, match="transport failure"):
        client.ask("question")
    assert len(calls) == 4
    assert pauses == [1, 2, 4]


def test_withdrawal_requires_missing_on_every_bounded_try() -> None:
    for results, expected in cast(
        tuple[tuple[Sequence[tuple[int, dict[str, Any]]], type[Exception]], ...],
        (
            ([(404, {})] * 4, chat.Unusable),
            ([(410, {})] * 4, chat.Unusable),
            ([(404, {}), (500, {}), (404, {}), (404, {})], chat.Interruption),
        ),
    ):
        client, calls, pauses = fake_client(results)
        with pytest.raises(expected) as caught:
            client.ask("question")
        if isinstance(caught.value, chat.Unusable):
            assert caught.value.cause == chat.Cause.WITHDRAWN
        assert len(calls) == 4 and pauses == [1, 2, 4]


@pytest.mark.parametrize("status", [400, 429, 500, 503, 401])
def test_retryable_statuses_and_success(status: int) -> None:
    client, calls, pauses = fake_client([(status, {}), (200, payload())])
    assert client.ask("question").text == "12"
    assert len(calls) == 2 and pauses == [1]


@pytest.mark.parametrize("status", [403, 302])
def test_nonretryable_status_is_sanitised(status: int) -> None:
    client, calls, _ = fake_client([(status, {"error": os.environ["ZOON_VERTEX_PROJECT"]})])
    with pytest.raises(chat.Interruption, match="request refused"):
        client.ask("question")
    assert len(calls) == 1


def test_project_is_read_at_call_time_and_missing_project_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls, _ = fake_client([(200, payload()), (200, payload())])
    client.ask("first")
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "changed-connection-sentinel")
    client.ask("second")
    assert "/changed-connection-sentinel/" in calls[-1][0]
    monkeypatch.delenv("ZOON_VERTEX_PROJECT")
    with pytest.raises(chat.Interruption, match="required"):
        client.ask("third")
    assert len(calls) == 2


def test_bad_credentials_and_malformed_response_are_sanitised() -> None:
    def fail() -> str:
        raise ValueError(os.environ["ZOON_VERTEX_PROJECT"])

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, payload()

    client = chat.VertexClient(transport, token=fail, preflight_transport=preflight_ok)
    with pytest.raises(chat.ConfigurationInterruption, match="credentials unavailable"):
        client.preflight()
    for change in (
        {"usageMetadata": {"promptTokenCount": -1}},
        {"usageMetadata": {"thoughtsTokenCount": 1.5}},
        {"usageMetadata": {"candidatesTokenCount": True}},
        {"modelVersion": 3},
        {"candidates": "broken"},
    ):
        client, _, _ = fake_client([(200, payload(**change))])
        with pytest.raises(chat.Interruption):
            client.ask("question")


def test_every_seat_round_group_is_a_new_call_even_for_identical_requests(tmp_path: Path) -> None:
    client = Replies([response("0")] * (2 * 2 * ROUNDS))
    run = session(tmp_path, client)
    for number in range(2):
        result = chat.play(GAMES["2-AI"], run, f"registered|2-AI|{number}")
        assert result.rounds == ((0, 0),) * ROUNDS
        assert (
            chat.replay(
                result.bytes, chat.replay_transcript(run.transcript, run.transcript.digest())
            ).bytes
            == result.bytes
        )
    assert len(client.requests) == len(run.transcript.entries) == 80
    assert client.requests[0] == client.requests[40]
    assert len(set(client.requests)) == 40
    assert run.transcript.lines[key()] == 2
    assert run.reserved == 0 and not run.pending


def test_resume_only_requests_missing_decisions(tmp_path: Path) -> None:
    first = Replies([response("0")] * 5 + [chat.TransportInterruption("transport failure")])
    run = session(tmp_path, first)
    with pytest.raises(chat.Interruption):
        chat.play(GAMES["2-AI"], run, GROUP)
    assert len(run.transcript.entries) == 5
    assert run.reserved == 0
    next_client = Replies([response("0")] * 35)
    resumed = session(tmp_path, next_client)
    result = chat.play(GAMES["2-AI"], resumed, GROUP)
    assert len(next_client.requests) == 35
    assert (
        chat.replay(
            result.bytes, chat.replay_transcript(resumed.transcript, resumed.transcript.digest())
        ).bytes
        == result.bytes
    )
    assert chat.play(GAMES["2-AI"], resumed, GROUP).bytes == result.bytes
    assert len(next_client.requests) == 35


def test_three_unusable_replies_are_recorded_counted_and_stop_only_chat(tmp_path: Path) -> None:
    client = Replies([response("no")] * 4)
    run = session(tmp_path, client)
    with pytest.raises(chat.Unusable) as caught:
        run.decision(view(), GROUP)
    assert caught.value.cause == chat.Cause.REPLIES
    assert len(client.requests) == len(run.transcript.entries) == 3
    assert run.transcript.spent == 3 * chat.cost(response("no"))
    with pytest.raises(chat.Unusable):
        session(tmp_path, client).decision(view(1), GROUP)
    assert len(client.requests) == 3


def test_transport_interruption_does_not_consume_a_reply_attempt(tmp_path: Path) -> None:
    broken, calls, pauses = fake_client(cast(list[tuple[int, dict[str, Any]]], [(503, {})] * 4))
    run = session(tmp_path, broken)
    with pytest.raises(chat.Interruption):
        run.decision(view(), GROUP)
    assert len(calls) == 4 and pauses == [1, 2, 4]
    assert not run.transcript.entries and run.reserved == 0
    resumed = session(tmp_path, Replies([response("no"), response("12")]))
    assert resumed.decision(view(), GROUP).points == 12
    assert [k.attempt for k in resumed.transcript.entries] == [1, 2]


@pytest.mark.parametrize("versions", [["other"], [""], [chat.MODEL + "-001", chat.MODEL + "-002"]])
def test_version_rule_records_the_response_before_making_chat_unusable(
    tmp_path: Path, versions: list[str]
) -> None:
    client = Replies([response(version=v) for v in versions])
    run = session(tmp_path, client)
    with pytest.raises(chat.Unusable) as caught:
        for s in range(len(versions)):
            run.decision(view(s), GROUP)
    assert caught.value.cause == chat.Cause.VERSION
    assert len(run.transcript.entries) == len(versions)
    with pytest.raises(chat.Unusable):
        session(tmp_path, client).check(recall=True)


def test_transcript_is_append_only_fsynced_and_resume_bound_to_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    fsync = os.fsync

    def recorded(fd: int) -> None:
        synced.append(fd)
        fsync(fd)

    monkeypatch.setattr(chat.os, "fsync", recorded)
    run = session(tmp_path, Replies([response(), response()]))
    run.decision(view(), GROUP)
    before = run.transcript.path.read_bytes()
    run.decision(view(1), GROUP)
    assert run.transcript.path.read_bytes().startswith(before)
    assert len(synced) == 2
    with pytest.raises(chat.GuardRefusal, match="changed on resume"):
        run.ask(key(), "changed")
    with pytest.raises(chat.GuardRefusal, match="duplicate"):
        run.transcript.append(key(), chat.request(view()), response())
    assert run.transcript.spent == 2 * chat.cost(response())
    assert run.transcript.digest() == hashlib.sha256(run.transcript.path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    ["commit", "mode", "duplicate", "digest", "tokens", "json", "noncanonical"],
)
def test_transcript_corruption_or_wrong_run_is_refused(tmp_path: Path, mutation: str) -> None:
    run = session(tmp_path, Replies([response()]))
    run.decision(view(), GROUP)
    lines = run.transcript.path.read_bytes().splitlines(keepends=True)
    identity = BINDING
    if mutation == "commit":
        identity = replace(identity, registered_commit="b" * 40)
    elif mutation == "mode":
        identity = replace(identity, mode="live")
    elif mutation == "partial":
        lines[-1] = lines[-1].rstrip(b"\n")
    elif mutation == "duplicate":
        lines += lines[1:]
    elif mutation == "digest":
        lines[-1] = lines[-1].replace(b'"request_digest":"', b'"request_digest":"x', 1)
    elif mutation == "noncanonical":
        lines[-1] = b" " + lines[-1]
    elif mutation == "tokens":
        lines[-1] = lines[-1].replace(b'"input_tokens":12', b'"input_tokens":-1')
    else:
        lines[-1] = b"not json\n"
    run.transcript.path.write_bytes(b"".join(lines))
    with pytest.raises((chat.GuardRefusal, chat.Interruption)):
        chat.Transcript(tmp_path, identity, read_only=False)


@pytest.mark.parametrize(
    "changes",
    [
        {"game": "unknown"},
        {"group": "registered|2-AI|100"},
        {"group": "registered|2-FE|0"},
        {"group": "registered|2-AI|01"},
        {"round": -1},
        {"round": 20},
        {"seat": 2},
        {"attempt": 0},
        {"attempt": 4},
        {"round": True},
        {"kind": "recall"},
    ],
)
def test_invalid_decision_identity_is_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(chat.GuardRefusal):
        chat.Identity(**{**asdict(key()), **changes})


@pytest.mark.parametrize(
    "changes", [{"registered_commit": "short"}, {"mode": "other"}, {"arm": "other"}]
)
def test_invalid_run_identity_is_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(chat.GuardRefusal):
        chat.RunIdentity(**{**asdict(BINDING), **changes})


@pytest.mark.parametrize("mutation", ["points", "line", "ledger", "reply", "header", "missing"])
def test_replay_rejects_tampered_log_or_transcript(tmp_path: Path, mutation: str) -> None:
    run = session(tmp_path, chat.StubClient())
    original = chat.play(GAMES["2-AI"], run, GROUP)
    log = original.bytes
    assert (
        chat.replay(log, chat.replay_transcript(run.transcript, run.transcript.digest())).bytes
        == log
    )
    if mutation in ("points", "line", "ledger", "header"):
        records = [json.loads(line) for line in log.splitlines()]
        if mutation == "points":
            records[1]["points"] += 1
        elif mutation == "line":
            records[1]["line"] += 1
        elif mutation == "ledger":
            records[3]["payoff"] = "wrong"
        else:
            records[0]["registered_commit"] = "b" * 40
        log = "".join(kernel.encoded(row) for row in records).encode()
    elif mutation == "missing":
        lines = run.transcript.path.read_bytes().splitlines(keepends=True)
        run.transcript.path.write_bytes(b"".join(lines[:1] + lines[2:]))
    else:
        entry = run.transcript.entries[key()]
        # Even an equivalent point reply must match the linked transcript bytes.
        changed = replace(entry, response=replace(entry.response, text="0" + entry.response.text))
        lines = run.transcript.path.read_bytes().splitlines(keepends=True)
        lines[1] = changed.bytes
        run.transcript.path.write_bytes(b"".join(lines))
    with pytest.raises(chat.Unusable) as caught:
        chat.replay(log, chat.replay_transcript(run.transcript, run.transcript.digest()))
    assert caught.value.cause == chat.Cause.REPLAY


def test_reaching_cap_from_reported_usage_makes_chat_unusable(tmp_path: Path) -> None:
    expensive = response(input_tokens=100000, output_tokens=0, thinking_tokens=0)
    cap = chat.cost(expensive)
    run = session(tmp_path, Replies([expensive]), cap - Decimal("0.000001"))
    with pytest.raises(chat.Unusable) as caught:
        run.decision(view(), GROUP)
    assert caught.value.cause == chat.Cause.CAP
    assert len(run.transcript.entries) == 1 and run.transcript.spent == cap


@pytest.mark.parametrize("cap", [Decimal("-1"), Decimal("NaN"), Decimal("Infinity")])
def test_invalid_cap_is_refused(tmp_path: Path, cap: Decimal) -> None:
    with pytest.raises(ValueError, match="finite"):
        session(tmp_path, chat.StubClient(), cap)


def test_project_name_cannot_enter_any_written_artifact_or_exception(tmp_path: Path) -> None:
    secret = os.environ["ZOON_VERTEX_PROJECT"]
    clean = session(tmp_path, chat.StubClient())
    result = chat.play(GAMES["2-AI"], clean, GROUP)
    (tmp_path / "game.jsonl").write_bytes(result.bytes)
    cases: list[Callable[[], object]] = [
        lambda: clean.transcript.append(key(), secret, response()),
        lambda: clean.transcript.append(key(), "request", response(secret)),
        lambda: clean.transcript.append(key(), "request", response(version=secret)),
        lambda: chat.Transcript(
            tmp_path / "identity", chat.RunIdentity(secret, "stub"), read_only=False
        ),
    ]
    for field in ("modelVersion", "unused"):
        client, _, _ = fake_client([(200, payload(**{field: secret}))])
        cases.append(lambda client=client: client.ask("question"))
    client, _, _ = fake_client([TimeoutError(secret)] * 4)
    cases.append(lambda: client.ask("question"))
    for call in cases:
        with pytest.raises((chat.GuardRefusal, chat.Interruption)) as caught:
            call()
        rendered = "".join(traceback.format_exception(caught.value))
        assert secret not in rendered
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_recall_uses_same_client_and_keeps_full_replies_after_unusable_decision(
    tmp_path: Path,
) -> None:
    texts = [
        "Wang's study",
        "Hilbe",
        "ZHANG",
        "asymmetric-public_goods",
        "wang2",
        "unknown",
        "",
        "x" * 2000,
    ]
    client = Replies([response("no")] * 3 + [response(text) for text in texts])
    run = session(tmp_path, client)
    with pytest.raises(chat.Unusable):
        run.decision(view(), GROUP)
    replies = chat.recall(run)
    assert len(replies) == 8
    assert [reply.response.text for reply in replies if reply.response] == texts
    assert [reply.flagged for reply in replies] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert client.requests[3:] == [prompted.recall(game) for _, game in sorted(GAMES.items())]
    assert len({key.game for key in run.transcript.entries if key.kind == "recall"}) == 8
    assert chat.recall(run) == replies and len(client.requests) == 11


def test_recall_interruption_resumes_only_missing_questions(tmp_path: Path) -> None:
    run = session(
        tmp_path, Replies([response("unknown"), chat.TransportInterruption("transport failure")])
    )
    with pytest.raises(chat.Interruption):
        chat.recall(run)
    client = Replies([response("unknown")] * 7)
    resumed = session(tmp_path, client)
    assert len(chat.recall(resumed)) == 8 and len(client.requests) == 7


def test_stub_smoke_strips_label_and_prints_all_reported_fields() -> None:
    class Recording(chat.StubClient):
        def ask(self, text: str) -> chat.Response:
            assert not text.startswith(prompted.FIXTURE_LABEL)
            return super().ask(text)

    lines = chat.smoke(Recording())
    assert len(lines) == 12
    assert all("usable = True" in line and "thinking tokens = 0" in line for line in lines)
    assert all("USD higher = " in line and "USD lower = " in line for line in lines)
    assert chat.smoke(Recording()) == lines


def test_smoke_defaults_to_stub_and_live_selects_injected_class_without_constructing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chat.main(["smoke"])
    assert capsys.readouterr().out.count("[smoke ") == 12
    selected: list[object] = []

    class StandIn:
        def preflight(self) -> None:
            selected.append("preflight")

    stand_in = StandIn()

    def constructor(transport: chat.HTTP) -> object:
        assert transport is chat.post
        selected.append("live")
        return stand_in

    def smoke(client: object) -> list[str]:
        assert selected == ["live", "preflight"]
        assert client is stand_in
        return ["selected"]

    monkeypatch.setattr(chat, "VertexClient", constructor)
    monkeypatch.setattr(chat, "smoke", smoke)
    chat.main(["smoke", "--live"])
    assert selected == ["live", "preflight"] and capsys.readouterr().out == "selected\n"
    with pytest.raises(SystemExit):
        chat.main(["smoke", "--live", "--stub"])


def test_fixtures_command_regenerates_identical_bytes_in_a_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prompted, "DIRECTORY", tmp_path)
    chat.main(["fixtures"])
    expected = chat.fixtures()
    for name, text in expected.items():
        assert (tmp_path / "chat-requests" / name).read_bytes() == text.encode()
    digests = json.loads((tmp_path / "chat-requests" / "sha256.json").read_bytes())
    assert digests == {name: chat.digest(text) for name, text in expected.items()}
    assert (tmp_path / "chat-requests" / "sha256.json").read_bytes() == (
        json.dumps(digests, indent=2) + "\n"
    ).encode()


def test_replay_failure_does_not_suppress_recall(tmp_path: Path) -> None:
    run = session(tmp_path, chat.StubClient())
    chat.play(GAMES["2-AI"], run, GROUP)
    with pytest.raises(chat.Unusable):
        chat.replay(b"invalid", chat.replay_transcript(run.transcript, run.transcript.digest()))
    assert all(reply.response is not None for reply in chat.recall(run))


def test_invalid_transport_attempt_count_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "TRANSPORT_ATTEMPTS", 0)
    with pytest.raises(chat.Interruption, match="positive"):
        chat.retry(lambda: response())


def test_unexpected_client_failure_is_sanitised_and_resumable(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([RuntimeError(os.environ["ZOON_VERTEX_PROJECT"])]))
    with pytest.raises(chat.ConsistencyInterruption, match="client failed"):
        run.decision(view(), GROUP)
    assert not run.transcript.entries and not run.pending and run.reserved == 0


@pytest.mark.parametrize("cell", sorted(GAMES))
def test_every_game_replays_with_exact_transfers(tmp_path: Path, cell: str) -> None:
    game = GAMES[cell]
    run = session(tmp_path, chat.StubClient())
    result = chat.play(game, run, f"registered|{cell}|0")
    restored = chat.Transcript(tmp_path, BINDING, read_only=False)
    assert (
        chat.replay(result.bytes, chat.replay_transcript(restored, restored.digest())).bytes
        == result.bytes
    )
    records = [json.loads(line) for line in result.bytes.splitlines()]
    transfers = [record for record in records if record["kind"] == "transfer"]
    assert len(restored.entries) == game.size * ROUNDS
    for r, row in enumerate(result.rounds):
        assert [record["payoff"] for record in transfers if record["round"] == r] == [
            str(value) for value in game.payoffs(row)
        ]


def test_all_seats_are_asked_before_the_round_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = GAMES["4-AI"]
    client = Replies([response("1")] * (game.size * ROUNDS))
    run = session(tmp_path, client)
    balance = kernel.balance
    applied: list[int] = []

    def checked(game: Any, row: tuple[int, ...]) -> Any:
        r = len(applied)
        assert len(client.requests) == (r + 1) * game.size
        for text in client.requests[r * game.size :]:
            assert text.count("\nRound ") == max(r, 1)
            assert f"Round {r + 1}\n" not in text
        applied.append(r)
        return balance(game, row)

    monkeypatch.setattr(kernel, "balance", checked)
    result = chat.play(game, run, "registered|4-AI|0")
    assert result.rounds == ((1, 1, 1, 1),) * ROUNDS
    assert applied == list(range(ROUNDS))


def test_default_cap_is_computed_by_the_budget_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zoon.interaction import budget

    monkeypatch.setattr(budget, "chat_cap", lambda: Decimal("9"))
    run = chat.Session(chat.Transcript(tmp_path, BINDING, read_only=False), chat.StubClient())
    assert run.cap == 9


def test_incomplete_transport_read_is_retried_with_the_same_attempt_count() -> None:
    error = chat.http.client.IncompleteRead(b"partial", 10)
    client, calls, pauses = fake_client([error] * 4)
    with pytest.raises(chat.Interruption, match="transport failure"):
        client.ask("question")
    assert len(calls) == 4 and pauses == [1, 2, 4]


def test_request_refuses_closed_or_incomplete_history() -> None:
    with pytest.raises(ValueError, match="open channel"):
        chat.request(replace(view(), open_channel=False))
    with pytest.raises(ValueError, match="seat or round"):
        chat.request(replace(view(), seat=2))
    observation = kernel.Observation(((0, 1),), None)
    with pytest.raises(ValueError, match="every contribution"):
        chat.request(replace(view(), history=(observation,)))
    wrong = kernel.Observation(((0, 1), (1, 1)), ())
    with pytest.raises(ValueError, match="payoffs differ"):
        chat.request(replace(view(), history=(wrong,)))
