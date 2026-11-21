"""Offline chat failure-path checks with injected transports."""

import http.client
import io
import json
import os
import subprocess
import urllib.error
from dataclasses import replace
from decimal import Decimal
from email.message import Message
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest

from tests.test_interaction_chat import (
    BINDING,
    GROUP,
    Replies,
    fake_client,
    key,
    payload,
    preflight_ok,
    response,
    session,
    view,
)
from tests.test_interaction_chat import (
    offline as offline,
)
from zoon.interaction import chat, kernel
from zoon.interaction.games import GAMES


def generation_ok(*args: Any) -> tuple[int, dict[str, Any]]:
    return 200, payload()


def assert_private(error: BaseException, *secrets: str) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert all(secret not in repr(current) for secret in secrets)
        pending.extend(e for e in (current.__context__, current.__cause__) if e is not None)
    assert error.__context__ is None and error.__cause__ is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.", None),
        ("36\r\n", 36),
        ("\x1f36", None),
        ("\x0b36", None),
        ("\x0c36", None),
        ("36\xa0", None),
    ],
)
def test_precise_trimming_and_fullmatch(text: str, expected: int | None) -> None:
    assert chat.points(text, 36) == expected


def test_admission_estimate_independent_high_price_and_maximum_output() -> None:
    text = "a\u00d7b\n"
    assert (
        chat.estimate(text)
        == (Decimal(len(text.encode("utf-8")) + 400) * 4 + Decimal(256) * 20) / 1_000_000
    )


def test_admission_cap_is_latched_during_play_and_across_processes(tmp_path: Path) -> None:
    # Round 1 fits; the larger round 2 request is refused at admission.
    client = Replies([response("0"), response("0")])
    run = session(tmp_path, client, Decimal("0.0145"))
    with pytest.raises(chat.Unusable) as caught:
        chat.play(GAMES["2-AI"], run, GROUP)
    assert caught.value.cause == run.terminal == chat.Cause.CAP
    assert len(client.requests) == 2
    assert chat.recall(run).terminal == chat.Cause.CAP
    resumed = chat.Session(
        chat.Transcript(tmp_path, BINDING, read_only=False), client, terminal=run.terminal
    )
    result = chat.recall(resumed)
    assert result.terminal == chat.Cause.CAP
    assert all(r.status == "not asked" and r.response is None for r in result)
    assert len(client.requests) == 2


def test_withdrawal_is_latched_during_play_and_recall_in_new_process(tmp_path: Path) -> None:
    client, calls, _ = fake_client([(404, cast(dict[str, Any], {}))] * 4)
    run = session(tmp_path, client)
    with pytest.raises(chat.Unusable) as caught:
        chat.play(GAMES["2-AI"], run, GROUP)
    assert run.terminal == caught.value.cause == chat.Cause.WITHDRAWN
    resumed = chat.Session(
        chat.Transcript(tmp_path, BINDING, read_only=False), client, terminal=run.terminal
    )
    assert chat.recall(resumed).terminal == chat.Cause.WITHDRAWN
    assert len(calls) == 4


def test_version_change_in_recall_is_paid_but_not_judged(tmp_path: Path) -> None:
    client = Replies([response("unknown"), response("Wang", version=chat.MODEL + "-002")])
    run = session(tmp_path, client)
    result = chat.recall(run)
    assert result.terminal == run.terminal == chat.Cause.VERSION
    assert result[0].status == "judged"
    assert result[1].response == response("Wang", version=chat.MODEL + "-002")
    assert result[1].status == "not judged: version" and not result[1].flagged
    assert all(r.status == "not asked" for r in result[2:])
    assert len(run.transcript.entries) == 2
    assert run.transcript.spent == 2 * chat.cost(response())
    assert chat.recall(run) == result


def test_equal_cap_is_allowed_and_recorded_answer_precedes_cap_check(tmp_path: Path) -> None:
    paid = response(input_tokens=100000, output_tokens=0, thinking_tokens=0)
    run = session(tmp_path, Replies([paid]), chat.cost(paid))
    assert run.decision(view(), GROUP).points == 12
    run.cap = Decimal("0")
    assert run.decision(view(), GROUP).points == 12
    with pytest.raises(chat.Unusable):
        run.decision(view(1), GROUP)
    assert run.terminal == chat.Cause.CAP


def test_usable_third_reply_is_accepted_and_later_decisions_continue(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([response("no"), response("no"), response("1"), response("2")]))
    assert run.decision(view(), GROUP).points == 1
    assert run.decision(view(1), GROUP).points == 2
    assert run.terminal is None


@pytest.mark.parametrize("failure", ["write", "short", "interrupt", "exit", "sync"])
def test_partial_write_rolls_back_and_next_append_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    before = transcript.path.read_bytes()
    original: Any = Path.open
    interrupt = KeyboardInterrupt("interrupted") if failure == "interrupt" else SystemExit(3)

    class HalfWrite:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.handle.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

        def write(self, data: bytes) -> int:
            if failure == "sync":
                return self.handle.write(data)
            written = self.handle.write(data[: len(data) // 2])
            if failure == "short":
                return written
            if failure in ("interrupt", "exit"):
                raise interrupt
            raise OSError("write cut short")

    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original(path, *args, **kwargs)
        return HalfWrite(handle) if path == transcript.path and args == ("ab",) else handle

    sync_calls = 0
    original_sync = os.fsync

    def sync(fd: int) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if failure == "sync" and sync_calls == 1:
            raise OSError("sync interrupted")
        original_sync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", opening)
        patch.setattr(chat.os, "fsync", sync)
        expected = (
            KeyboardInterrupt
            if failure == "interrupt"
            else SystemExit
            if failure == "exit"
            else chat.StorageInterruption
        )
        with pytest.raises(expected) as caught:
            transcript.append(key(1), chat.request(view(1)), response())
        if failure in ("interrupt", "exit"):
            assert caught.value is interrupt
        assert sync_calls == (2 if failure == "sync" else 1)
    assert transcript.path.read_bytes() == before
    assert key(1) not in transcript.entries
    transcript.append(key(1), chat.request(view(1)), response())
    assert len(chat.Transcript(tmp_path, BINDING, read_only=False).entries) == 2


def test_torn_tail_removed_once_and_decision_asked_again_once(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([response()]))
    run.decision(view(), GROUP)
    raw = run.transcript.path.read_bytes()
    header = raw.split(b"\n", 1)[0] + b"\n"
    run.transcript.path.write_bytes(raw[:-8])
    client = Replies([response()])
    resumed = session(tmp_path, client)
    assert resumed.tail_removals == 1
    assert resumed.transcript.path.read_bytes() == header
    resumed.decision(view(), GROUP)
    resumed.decision(view(), GROUP)
    assert len(client.requests) == 1 and resumed.transcript.path.read_bytes() == raw
    assert session(tmp_path, Replies([])).tail_removals == 1
    assert session(tmp_path, Replies([])).tail_removals == 1


def test_append_refuses_torn_tail_without_modifying_it(tmp_path: Path) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    raw = transcript.path.read_bytes() + b"fragment"
    transcript.path.write_bytes(raw)
    with pytest.raises(chat.GuardRefusal, match="incomplete"):
        transcript.append(key(1), chat.request(view(1)), response())
    assert transcript.path.read_bytes() == raw


@pytest.mark.parametrize("damage", ["blank", "noncanonical header", "foreign line"])
def test_complete_damage_is_never_repaired(tmp_path: Path, damage: str) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    raw = transcript.path.read_bytes()
    if damage == "blank":
        raw += b"\n"
    elif damage == "noncanonical header":
        raw = b" " + raw
    else:
        other = chat.Transcript(tmp_path / "other", replace(BINDING, mode="live"), read_only=False)
        other.append(key(1), chat.request(view(1)), response())
        raw += other.path.read_bytes().split(b"\n")[1] + b"\n"
    raw += b"torn"
    transcript.path.write_bytes(raw)
    with pytest.raises(chat.GuardRefusal):
        chat.Transcript(tmp_path, BINDING, read_only=False)
    assert transcript.path.read_bytes() == raw


@pytest.mark.parametrize(
    "field", ["modelVersion", "promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"]
)
@pytest.mark.parametrize("invalid", [None, "", True, 1.5])
def test_malformed_success_retried_without_record(
    tmp_path: Path, field: str, invalid: object
) -> None:
    data = payload()
    target = data if field == "modelVersion" else data["usageMetadata"]
    if invalid is None:
        if field in ("candidatesTokenCount", "thoughtsTokenCount"):
            target[field] = None
        else:
            target.pop(field)
    else:
        target[field] = invalid
    client, calls, pauses = fake_client([(200, data)] * 4)
    run = session(tmp_path, client)
    with pytest.raises(chat.TransportInterruption):
        run.decision(view(), GROUP)
    assert len(calls) == 4 and pauses == [1, 2, 4]
    assert not run.transcript.entries and not run.transcript.path.exists()


def test_malformed_success_can_recover_within_attempts() -> None:
    client, calls, pauses = fake_client([(200, {}), (200, payload())])
    assert client.ask("question").text == "12"
    assert len(calls) == 2 and pauses == [1]


def test_absent_output_counts_are_zero_without_candidate_text() -> None:
    client, _, _ = fake_client(
        [(200, payload(candidates=[], usageMetadata={"promptTokenCount": 9}))]
    )
    assert client.ask("question") == response(
        "", input_tokens=9, output_tokens=0, thinking_tokens=0
    )


def test_missing_finish_reason_preserves_candidate_text() -> None:
    client, _, _ = fake_client(
        [(200, payload(candidates=[{"content": {"parts": [{"text": "1"}]}}]))]
    )
    assert client.ask("question").text == "1"


@pytest.mark.parametrize("secret_kind", ["project", "bearer"])
@pytest.mark.parametrize("field", ["request", "text", "version"])
def test_transcript_leak_guard_fresh_identity_on_append_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret_kind: str, field: str
) -> None:
    secret = "private-metadata-sentinel"
    # Build canonical tainted bytes while the secret is not configured, then load
    # with the guard configured. Every identity is new, so duplicates cannot mask it.
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "unrelated-project")
    text = secret if field == "request" else chat.request(view())
    reply = response(**{field: secret}) if field != "request" else response()
    tainted = chat.Transcript(tmp_path / "tainted", BINDING, read_only=False)
    tainted.append(key(), text, reply)
    if secret_kind == "project":
        monkeypatch.setenv("ZOON_VERTEX_PROJECT", secret)
    bearer = secret if secret_kind == "bearer" else ""
    clean = chat.Transcript(tmp_path / "clean", BINDING, bearer=bearer, read_only=False)
    with pytest.raises(chat.GuardRefusal, match="connection metadata refused"):
        clean.append(key(), text, reply)
    assert not clean.path.exists()
    raw = tainted.path.read_bytes()
    with pytest.raises(chat.GuardRefusal, match="connection metadata refused"):
        chat.Transcript(tmp_path / "tainted", BINDING, bearer=bearer, read_only=False)
    assert tainted.path.read_bytes() == raw


@pytest.mark.parametrize("text", ["fake-bearer", "connection-location-sentinel"])
def test_outgoing_request_guard_includes_bearer(text: str) -> None:
    client, calls, _ = fake_client([])
    with pytest.raises(chat.GuardRefusal, match="connection metadata refused"):
        client.ask(text)
    assert not calls


def test_live_wire_with_patched_urlopen_and_token_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[Any] = []
    tokens: list[Any] = []

    def token(*args: Any, **kwargs: Any) -> Any:
        tokens.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, f" token-{len(tokens)}\n", "")

    def opening(request: Any, *, timeout: int) -> Any:
        assert timeout == 30
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 401, "private", Message(), None)
        result = io.BytesIO(json.dumps(payload()).encode())
        result.status = 200  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(chat.subprocess, "run", token)
    monkeypatch.setattr(chat.urllib.request, "urlopen", opening)
    pauses: list[float] = []
    client = chat.VertexClient(chat.post, pause=pauses.append, preflight_transport=preflight_ok)
    client.preflight()
    assert client.ask("question").text == "12"
    assert pauses == [1] and len(tokens) == len(requests) == 2
    assert tokens[0] == (
        (["gcloud", "auth", "application-default", "print-access-token"],),
        {"capture_output": True, "text": True, "check": True},
    )
    for index, outgoing in enumerate(requests, 1):
        assert (
            outgoing.full_url
            == "https://aiplatform.googleapis.com/v1/projects/connection-location-sentinel/locations/global/publishers/google/models/gemini-4-carbon:generateContent"
        )
        assert outgoing.method == "POST"
        assert dict((k.lower(), v) for k, v in outgoing.header_items()) == {
            "authorization": f"Bearer token-{index}",
            "content-type": "application/json",
            "x-goog-user-project": "connection-location-sentinel",
        }
        assert json.loads(outgoing.data) == {
            "contents": [{"role": "user", "parts": [{"text": "question"}]}],
            "generationConfig": {
                "temperature": 1.0,
                "thinkingConfig": {"thinkingLevel": "low"},
                "maxOutputTokens": 256,
            },
            "safetySettings": chat.SAFETY_SETTINGS,
        }


@pytest.mark.parametrize("failure", ["json", "credentials", "status line", "processing"])
def test_sanitized_faults_have_no_reachable_secret_chain(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    secret = "token-and-project-sentinel"
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", secret)
    if failure == "credentials":

        def failing_token(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(1, secret, secret, secret)

        monkeypatch.setattr(chat.subprocess, "run", failing_token)

        def call() -> object:
            return chat.access_token()

        expected = chat.ConfigurationInterruption
    else:

        def opening(*args: Any, **kwargs: Any) -> Any:
            if failure == "status line":
                raise http.client.BadStatusLine(secret)
            return io.BytesIO(secret.encode())

        monkeypatch.setattr(chat.urllib.request, "urlopen", opening)
        if failure == "processing":

            def processing(*args: Any) -> Any:
                raise RuntimeError(secret)

            client = chat.VertexClient(
                processing,
                token=lambda: "safe",
                pause=lambda _: None,
                preflight_transport=preflight_ok,
            )
        else:
            client = chat.VertexClient(
                chat.post,
                token=lambda: "safe",
                pause=lambda _: None,
                preflight_transport=preflight_ok,
            )

        client.preflight()

        def call() -> object:
            return client.ask("question")

        expected = chat.TransportInterruption
    with pytest.raises(expected) as caught:
        call()
    assert_private(caught.value, secret)


@pytest.mark.parametrize("status", [404, 410, 403])
def test_preflight_missing_is_configuration_and_never_withdrawal(status: int) -> None:
    calls: list[Any] = []

    def metadata(*args: Any) -> Any:
        calls.append(args)
        return status, {}

    def forbidden(*args: Any) -> Any:
        pytest.fail("generation forbidden")

    client = chat.VertexClient(
        forbidden,
        token=lambda: "token",
        preflight_transport=metadata,
    )
    with pytest.raises(chat.ConfigurationInterruption):
        client.preflight()
    assert len(calls) == 1
    assert (
        calls[0][0]
        == "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-4-carbon"
    )
    assert calls[0][2] == b""


def test_preflight_live_transport_is_one_non_generating_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def opening(request: Any, *, timeout: int) -> Any:
        calls.append(request)
        assert timeout == 30 and request.method == "GET" and request.data is None
        result = io.BytesIO(b"{}")
        result.status = 200  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(chat.urllib.request, "urlopen", opening)
    client = chat.VertexClient(chat.post, token=lambda: "token")
    client.preflight()
    assert len(calls) == 1


def test_negative_stored_usage_is_consistency_fault(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([response()]))
    run.decision(view(), GROUP)
    path = run.transcript.path
    path.write_bytes(path.read_bytes().replace(b'"input_tokens":12', b'"input_tokens":-1'))
    with pytest.raises(chat.ConsistencyInterruption):
        chat.Transcript(tmp_path, BINDING, read_only=False)


def test_ledger_each_field_is_exact_fraction(tmp_path: Path) -> None:
    run = session(tmp_path, chat.StubClient())
    result = chat.play(GAMES["4-MI"], run, "registered|4-MI|0")
    game = GAMES["4-MI"]
    transfers: list[dict[str, Any]] = [
        json.loads(line) for line in result.log if json.loads(line)["kind"] == "transfer"
    ]
    for transfer in transfers:
        row = result.rounds[int(transfer["round"])]
        seat = int(transfer["seat"])
        pool = sum(
            (Fraction(c) * p for c, p in zip(row, game.productivities, strict=True)), Fraction()
        )
        kept = Fraction(game.endowments[seat] - row[seat])
        share = pool / game.size
        assert {k: Fraction(transfer[k]) for k in ("kept", "pool", "share", "payoff")} == {
            "kept": kept,
            "pool": pool,
            "share": share,
            "payoff": kept + share,
        }


def test_replay_checks_request_even_with_matching_links(tmp_path: Path) -> None:
    run = session(tmp_path, chat.StubClient())
    result = chat.play(GAMES["2-AI"], run, GROUP)
    entry = run.transcript.entries[key()]
    changed = replace(entry, request="different", request_digest=chat.digest("different"))
    run.transcript.entries[key()] = changed
    records = [json.loads(line) for line in result.bytes.splitlines()]
    records[1]["entry_digest"] = chat.digest(changed.bytes.decode())
    records[1]["attempts"] = [chat.digest(changed.bytes.decode())]
    log = "".join(kernel.encoded(row) for row in records).encode()
    with pytest.raises(chat.Unusable):
        chat.replay(log, run.transcript)


@pytest.mark.parametrize("kind", ["unusable", "recall"])
@pytest.mark.parametrize("field", ["text", "input_tokens", "output_tokens", "thinking_tokens"])
def test_whole_transcript_replay_binds_all_paid_lines(
    tmp_path: Path, kind: str, field: str
) -> None:
    replies = Replies(
        [response("no"), response("0")] + [response("0")] * 39 + [response("unknown")] * 8
    )
    run = session(tmp_path, replies)
    result = chat.play(GAMES["2-AI"], run, GROUP)
    chat.recall(run)
    expected = run.transcript.digest()
    chat.replay_transcript(run.transcript, expected)
    selected = (
        key()
        if kind == "unusable"
        else next(k for k in run.transcript.entries if k.kind == "recall")
    )
    lines = run.transcript.path.read_bytes().splitlines(keepends=True)
    entry = run.transcript.entries[selected]
    changed = replace(
        entry,
        response=replace(entry.response, **{field: "still unusable" if field == "text" else 999}),
    )
    lines[run.transcript.lines[selected] - 1] = changed.bytes
    run.transcript.path.write_bytes(b"".join(lines))
    restored = chat.Transcript(tmp_path, BINDING, read_only=False)
    with pytest.raises(chat.Unusable):
        chat.replay_transcript(restored, expected)
    if kind == "unusable":
        with pytest.raises(chat.Unusable):
            chat.replay(result.bytes, restored)


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("asymmetric public-goods", True),
        ("asymmetric, public goods", True),
        ("Asymmetric Public Goods Games", True),
        ("public goods", False),
        ("asymmetric private public goods", False),
        ("Hwang", False),
        ("zhang's", True),
    ],
)
def test_recall_adjacent_runs(tmp_path: Path, text: str, flagged: bool) -> None:
    result = chat.recall(session(tmp_path, Replies([response(text)] * 8)))
    assert all(r.flagged == flagged for r in result)


def test_fsync_follows_visible_complete_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    original = os.fsync
    synced: list[int] = []

    def sync(fd: int) -> None:
        raw = transcript.path.read_bytes()
        assert raw.endswith(b"\n") and len(raw.splitlines()) == 2
        synced.append(fd)
        original(fd)

    monkeypatch.setattr(chat.os, "fsync", sync)
    transcript.append(key(), chat.request(view()), response())
    assert len(synced) == 1


def test_short_registered_commit_and_unknown_decision_kind_refused() -> None:
    with pytest.raises(chat.GuardRefusal):
        chat.RunIdentity("a" * 7, "stub")
    with pytest.raises(chat.GuardRefusal):
        chat.Identity("2-AI", GROUP, 0, 0, 1, "other")  # type: ignore[arg-type]


def test_direct_json_and_http_errors_never_escape_or_become_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-url-and-token"

    def bad_json(*args: Any, **kwargs: Any) -> Any:
        return io.BytesIO(secret.encode())

    monkeypatch.setattr(chat.urllib.request, "urlopen", bad_json)
    with pytest.raises(chat.TransientError) as caught:
        chat.post("https://example.invalid/" + secret, {}, b"{}")
    assert_private(caught.value, secret)

    def http_error(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.HTTPError(secret, 503, secret, Message(), None)

    monkeypatch.setattr(chat.urllib.request, "urlopen", http_error)
    assert chat.post("https://example.invalid", {}, b"{}") == (503, {})


def test_client_token_errors_are_configuration_without_secret_chain() -> None:
    secret = "private-token-credential-error"

    def token() -> str:
        raise RuntimeError(secret)

    def transport(*args: Any) -> tuple[int, dict[str, Any]]:
        return 200, payload()

    client = chat.VertexClient(transport, token=token, preflight_transport=preflight_ok)
    with pytest.raises(chat.ConfigurationInterruption) as caught:
        client.preflight()
    assert_private(caught.value, secret)


def test_bad_status_line_retries_all_attempts() -> None:
    client, calls, pauses = fake_client([http.client.BadStatusLine("private")] * 4)
    with pytest.raises(chat.TransportInterruption):
        client.ask("question")
    assert len(calls) == 4 and pauses == [1, 2, 4]


@pytest.mark.parametrize("field", ["modelVersion", "unused"])
def test_client_response_guard_includes_bearer(field: str) -> None:
    client, calls, _ = fake_client([(200, payload(**{field: "fake-bearer"}))])
    with pytest.raises(chat.GuardRefusal, match="connection metadata refused"):
        client.ask("question")
    assert len(calls) == 1


@pytest.mark.parametrize("cause", list(chat.Cause))
def test_every_persisted_cause_stops_play_and_controls_recall(
    tmp_path: Path, cause: chat.Cause
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response("0"))
    client = Replies([response("Wang")] * 8)
    run = chat.Session(transcript, client, Decimal("1000"), terminal=cause)
    with pytest.raises(chat.Unusable) as caught:
        chat.play(GAMES["2-AI"], run, GROUP)
    assert caught.value.cause is run.terminal is cause
    assert client.requests == []
    result = chat.recall(run)
    allowed = cause in (chat.Cause.REPLIES, chat.Cause.REPLAY, chat.Cause.OPERATOR_CLOSED_LOOP)
    assert len(client.requests) == (8 if allowed else 0)
    assert result.terminal is run.terminal is cause
    assert result.stopped_by is (None if allowed else cause)
    for reply in result:
        if allowed:
            identity = chat.Identity(reply.game, "recall", 0, 0, 1, "recall")
            entry = transcript.entries[identity]
            assert reply.response == response("Wang")
            assert reply.status == "judged" and reply.flagged
            assert reply.line == transcript.lines[identity]
            assert reply.entry_digest == chat.digest(entry.bytes.decode())
        else:
            assert reply.response is None and reply.status == "not asked"
            assert reply.line is reply.entry_digest is None and not reply.flagged
    assert chat.recall(run) == result
    assert len(client.requests) == (8 if allowed else 0)


@pytest.mark.parametrize("terminal", ["CAP", chat.Cause.CAP.value, 1, False, object()])
def test_terminal_requires_enum_member(tmp_path: Path, terminal: object) -> None:
    with pytest.raises(TypeError, match="Cause"):
        chat.Session(chat.Transcript(tmp_path, BINDING), Replies([]), terminal=terminal)  # type: ignore[arg-type]


def test_operator_stop_during_recall_keeps_answered_questions(tmp_path: Path) -> None:
    client = Replies([response("Wang"), chat.TransportInterruption("interrupted")])
    run = session(tmp_path, client)
    with pytest.raises(chat.TransportInterruption):
        chat.recall(run)
    resumed_client = Replies([])
    resumed = chat.Session(
        chat.Transcript(tmp_path, BINDING), resumed_client, terminal=chat.Cause.OPERATOR_RECALL
    )
    result = chat.recall(resumed)
    assert result.terminal is result.stopped_by is chat.Cause.OPERATOR_RECALL
    assert result[0].response == response("Wang")
    assert result[0].status == "judged" and result[0].flagged
    assert result[0].line == 2 and result[0].entry_digest is not None
    assert all(r.status == "not asked" and r.response is None for r in result[1:])
    assert resumed_client.requests == []


@pytest.mark.parametrize("later", [chat.Cause.VERSION, chat.Cause.CAP, chat.Cause.WITHDRAWN])
def test_first_arm_cause_stands_when_something_else_stops_recall(
    tmp_path: Path, later: chat.Cause
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response("no"))
    replies = (
        [response("Wang", version=chat.MODEL + "-002")]
        if later is chat.Cause.VERSION
        else [response("Wang", input_tokens=1_000_000)]
        if later is chat.Cause.CAP
        else [chat.Unusable(later)]
    )
    client = Replies(replies)
    run = chat.Session(transcript, client, Decimal("1"), terminal=chat.Cause.REPLIES)
    result = chat.recall(run)
    assert result.terminal is run.terminal is chat.Cause.REPLIES
    assert result.stopped_by is later
    assert len(client.requests) == 1
    if later is chat.Cause.VERSION:
        assert result[0].response == replies[0]
        assert result[0].status == "not judged: version" and not result[0].flagged
    elif later is chat.Cause.CAP:
        assert result[0].response == replies[0]
        assert result[0].status == "judged" and result[0].flagged
        assert run.transcript.spent > run.cap
        assert len(run.transcript.entries) == 2
    else:
        assert result[0].status == "not asked" and result[0].response is None
    assert all(r.status == "not asked" for r in result[1:])


def test_replies_cause_latched_from_loaded_attempts_and_from_decision(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([response("no")] * 3))
    with pytest.raises(chat.Unusable) as caught:
        run.decision(view(), GROUP)
    assert run.terminal is caught.value.cause is chat.Cause.REPLIES
    resumed = session(tmp_path, Replies([]))
    with pytest.raises(chat.Unusable) as caught:
        resumed.check()
    assert resumed.terminal is caught.value.cause is chat.Cause.REPLIES
    # Exercise decision's final stop independently of the post-append check.
    transcript = chat.Transcript(tmp_path, BINDING)
    replayed = chat.Session(transcript, Replies([]), Decimal("1000"))
    with pytest.raises(chat.Unusable):
        replayed.decision(view(), GROUP)
    assert replayed.terminal is chat.Cause.REPLIES


def test_recorded_version_invalid_answer_is_never_returned_on_resume(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([response(version="other")]))
    with pytest.raises(chat.Unusable):
        run.decision(view(), GROUP)
    resumed = session(tmp_path, Replies([]))
    with pytest.raises(chat.Unusable) as caught:
        resumed.decision(view(), GROUP)
    assert caught.value.cause is resumed.terminal is chat.Cause.VERSION


@pytest.mark.parametrize("version", ["foreign", chat.MODEL + "-002"])
def test_recall_checks_first_version_and_prior_decision_versions(
    tmp_path: Path, version: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    if version.startswith(chat.MODEL):
        transcript.append(key(), chat.request(view()), response("0"))
    client = Replies([response("Wang", version=version)])
    run = chat.Session(transcript, client, Decimal("1000"))
    result = chat.recall(run)
    assert result.terminal is result.stopped_by is run.terminal is chat.Cause.VERSION
    assert result[0].response == response("Wang", version=version)
    assert result[0].status == "not judged: version" and not result[0].flagged
    assert result[0].line == (3 if transcript.entries.get(key()) else 2)
    entry = next(e for k, e in transcript.entries.items() if k.kind == "recall")
    assert result[0].entry_digest == chat.digest(entry.bytes.decode())
    assert all(r.response is None and r.status == "not asked" for r in result[1:])
    assert len(client.requests) == 1


def test_recorded_service_reply_shapes_are_recorded_once_and_costed(tmp_path: Path) -> None:
    import re

    record = (Path(__file__).parents[1] / "docs/interaction/reply-shape.txt").read_text()
    shapes = record.split("reply ")[1:]
    assert len(shapes) == 2
    for index, shape in enumerate(shapes):
        counts = {name: int(value) for name, value in re.findall(r"(\w+TokenCount)=(\d+)", shape)}
        usage = {k: v for k, v in counts.items() if k != "totalTokenCount"}
        data = payload(usageMetadata=usage, modelVersion=chat.MODEL)
        client, calls, pauses = fake_client([(200, data)])
        run = session(tmp_path / str(index), client)
        assert run.decision(view(), GROUP).points == 12
        assert run.decision(view(), GROUP).points == 12
        assert len(calls) == len(run.transcript.entries) == 1 and pauses == []
        reply = next(iter(run.transcript.entries.values())).response
        assert reply.input_tokens == counts["promptTokenCount"]
        assert reply.output_tokens == 6 and reply.thinking_tokens == (0 if index == 0 else 103)
        expected = Decimal("0.000372") if index == 0 else Decimal("0.003968")
        assert run.transcript.spent == chat.cost(reply) == expected


@pytest.mark.parametrize("missing", ["candidatesTokenCount", "thoughtsTokenCount"])
def test_absent_optional_counts_with_text_are_zero(missing: str) -> None:
    data = payload()
    data["usageMetadata"].pop(missing)
    client, calls, _ = fake_client([(200, data)])
    reply = client.ask("question")
    assert reply.text == "12"
    assert (reply.output_tokens, reply.thinking_tokens) == (
        (0, 7) if missing == "candidatesTokenCount" else (2, 0)
    )
    assert len(calls) == 1


def test_tail_receipt_is_synced_before_truncation_and_survives_openings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    complete = transcript.path.read_bytes()
    tail = b"unfinished\xffline"
    transcript.path.write_bytes(complete + tail)
    receipt = {
        "offset": len(complete),
        "length": len(tail),
        "sha256": hashlib.sha256(tail).hexdigest(),
    }
    events: list[str] = []
    original_open: Any = Path.open
    original_sync = os.fsync

    class Traced:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.handle.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

        def truncate(self, size: int) -> int:
            assert events == ["receipt synced"]
            events.append("truncated")
            return self.handle.truncate(size)

    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original_open(path, *args, **kwargs)
        return Traced(handle) if path == transcript.path and args == ("r+b",) else handle

    def sync(fd: int) -> None:
        if not events:
            assert transcript.path.read_bytes() == complete + tail
            assert transcript.removals_path.read_bytes() == kernel.encoded(receipt).encode()
            events.append("receipt synced")
        else:
            assert events == ["receipt synced", "truncated"]
            assert transcript.path.read_bytes() == complete
            events.append("truncation synced")
        original_sync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", opening)
        patch.setattr(chat.os, "fsync", sync)
        reopened = chat.Transcript(tmp_path, BINDING, read_only=False)
    assert events == ["receipt synced", "truncated", "truncation synced"]
    assert reopened.tail_removals == 1
    assert chat.Transcript(tmp_path, BINDING).tail_removals == 1
    assert chat.Transcript(tmp_path, BINDING, read_only=False).tail_removals == 1
    assert transcript.removals_path.read_bytes() == kernel.encoded(receipt).encode()
    transcript.path.write_bytes(complete + b"another tail")
    assert chat.Transcript(tmp_path, BINDING, read_only=False).tail_removals == 2
    assert chat.Transcript(tmp_path, BINDING).tail_removals == 2
    assert len(transcript.removals_path.read_bytes().splitlines()) == 2


@pytest.mark.parametrize("header_only", [False, True])
def test_read_only_opening_and_replay_refuse_torn_tail_without_writes(
    tmp_path: Path, header_only: bool
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    raw = transcript.path.read_bytes()
    raw = raw.split(b"\n")[0][:-4] if header_only else raw[:-4]
    transcript.path.write_bytes(raw)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for operation in (
        lambda: chat.Transcript(tmp_path, BINDING),
        transcript.digest,
        lambda: chat.replay_transcript(transcript, "unused"),
    ):
        with pytest.raises(chat.GuardRefusal, match="incomplete"):
            operation()
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    with pytest.raises(chat.Unusable):
        chat.replay(b"{}\n", transcript)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    repaired = chat.Transcript(tmp_path, BINDING, read_only=False)
    assert repaired.tail_removals == 1
    assert repaired.path.read_bytes() == (b"" if header_only else raw[: raw.rfind(b"\n") + 1])
    assert len(repaired.removals_path.read_bytes().splitlines()) == 1


def test_read_only_transcript_cannot_append(tmp_path: Path) -> None:
    transcript = chat.Transcript(tmp_path, BINDING)
    with pytest.raises(chat.StorageInterruption, match="read-only"):
        transcript.append(key(), chat.request(view()), response())
    assert list(tmp_path.iterdir()) == []
    client = Replies([])
    with pytest.raises(chat.StorageInterruption, match="read-only"):
        chat.Session(transcript, client, Decimal("1000")).decision(view(), GROUP)
    assert client.requests == []


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("point", ["open", "mkdir"])
def test_open_failure_is_storage_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial: bool, point: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    if not initial:
        transcript.append(key(), chat.request(view()), response())
    before = transcript.path.read_bytes() if transcript.path.exists() else None
    original: Any = Path.open

    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == transcript.path and args == ("ab",):
            raise OSError("cannot open")
        return original(path, *args, **kwargs)

    def mkdir(*args: Any, **kwargs: Any) -> None:
        raise OSError("cannot create")

    with monkeypatch.context() as patch:
        if point == "open":
            patch.setattr(Path, "open", opening)
        else:
            patch.setattr(Path, "mkdir", mkdir)
        with pytest.raises(chat.StorageInterruption):
            transcript.append(key(1), chat.request(view(1)), response())
    assert (transcript.path.read_bytes() if transcript.path.exists() else None) == before


def test_failed_rollback_is_consistency_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())

    def failed_sync(fd: int) -> None:
        raise OSError("sync unavailable")

    monkeypatch.setattr(chat.os, "fsync", failed_sync)
    with pytest.raises(chat.ConsistencyInterruption, match="rollback failed"):
        transcript.append(key(1), chat.request(view(1)), response())


@pytest.mark.parametrize("operation", ["open", "digest", "replay", "replay transcript"])
def test_read_failure_is_storage_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    original = Path.read_bytes

    def read(path: Path) -> bytes:
        if path == transcript.path:
            raise OSError("read failed")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(chat.StorageInterruption):
        if operation == "open":
            chat.Transcript(tmp_path, BINDING)
        elif operation == "digest":
            transcript.digest()
        elif operation == "replay":
            chat.replay(b"{}\n", transcript)
        else:
            chat.replay_transcript(transcript, "unused")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_preflight_retries_transient_statuses_with_bounded_pauses(status: int) -> None:
    statuses = iter([status] * 3 + [200])
    calls: list[Any] = []
    pauses: list[float] = []

    def metadata(*args: Any) -> Any:
        calls.append(args)
        return next(statuses), {}

    client = chat.VertexClient(
        generation_ok,
        token=lambda: "token",
        preflight_transport=metadata,
        pause=pauses.append,
    )
    client.preflight()
    assert len(calls) == 4 and pauses == [1, 2, 4]
    assert client.ask("question").text == "12"
    for _, headers, body in calls:
        assert headers == {
            "Authorization": "Bearer token",
            "Content-Type": "application/json",
            "x-goog-user-project": os.environ["ZOON_VERTEX_PROJECT"],
        }
        assert body == b""


def test_preflight_refreshes_rejected_bearer_and_generation_uses_same_headers() -> None:
    tokens = iter(["first", "second"])
    calls: list[Any] = []
    generated: list[Any] = []
    pauses: list[float] = []

    def metadata(*args: Any) -> Any:
        calls.append(args)
        return (401 if len(calls) == 1 else 200), {}

    def generation(*args: Any) -> Any:
        generated.append(args)
        return 200, payload()

    client = chat.VertexClient(
        generation, token=lambda: next(tokens), preflight_transport=metadata, pause=pauses.append
    )
    client.preflight()
    client.ask("question")
    assert len(calls) == 2 and pauses == [1]
    assert calls[0][1]["Authorization"] == "Bearer first"
    assert (
        calls[1][1]
        == generated[0][1]
        == {
            "Authorization": "Bearer second",
            "Content-Type": "application/json",
            "x-goog-user-project": os.environ["ZOON_VERTEX_PROJECT"],
        }
    )


@pytest.mark.parametrize("failure", [OSError, http.client.BadStatusLine, RuntimeError])
def test_preflight_exceptions_are_wrapped_and_retried(failure: type[Exception]) -> None:
    calls: list[int] = []
    pauses: list[float] = []
    secret = "private-error-sentinel"

    def metadata(*args: Any) -> Any:
        calls.append(1)
        raise failure(secret)

    client = chat.VertexClient(
        generation_ok,
        token=lambda: "token",
        preflight_transport=metadata,
        pause=pauses.append,
    )
    with pytest.raises(chat.TransportInterruption) as caught:
        client.preflight()
    assert_private(caught.value, secret)
    assert len(calls) == 4 and pauses == [1, 2, 4]
    with pytest.raises(chat.ConfigurationInterruption, match="preflight"):
        client.ask("question")


def test_project_is_required_for_preflight_and_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def metadata(*args: Any) -> Any:
        calls.append(1)
        return 200, {}

    client = chat.VertexClient(generation_ok, token=lambda: "token", preflight_transport=metadata)
    with monkeypatch.context() as patch:
        patch.delenv("ZOON_VERTEX_PROJECT")
        with pytest.raises(chat.ConfigurationInterruption, match="PROJECT"):
            client.preflight()
    assert calls == []
    client.preflight()
    monkeypatch.delenv("ZOON_VERTEX_PROJECT")
    with pytest.raises(chat.ConfigurationInterruption, match="PROJECT"):
        client.ask("question")


def test_generation_requires_preflight_in_current_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def generating(*args: Any) -> Any:
        calls.append(1)
        return 200, payload()

    client = chat.VertexClient(generating, token=lambda: "token", preflight_transport=preflight_ok)
    with pytest.raises(chat.ConfigurationInterruption, match="preflight"):
        client.ask("question")
    assert calls == []
    client.preflight()
    assert client.ask("question").text == "12"
    pid = os.getpid()
    monkeypatch.setattr(chat.os, "getpid", lambda: pid + 1)
    with pytest.raises(chat.ConfigurationInterruption, match="preflight"):
        client.ask("question")
    assert calls == [1]


def test_forbidden_generation_is_configuration_fault() -> None:
    client, calls, pauses = fake_client([(403, {})])
    with pytest.raises(chat.ConfigurationInterruption):
        client.ask("question")
    assert len(calls) == 1 and pauses == []


@pytest.mark.parametrize(
    "damage",
    [
        "line count",
        "decision request",
        "recall request",
        "disk entry",
        "round order",
        "attempt after acceptance",
    ],
)
def test_replay_transcript_structure_independently_of_digest(tmp_path: Path, damage: str) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    identity = (
        chat.Identity("2-AI", "recall", 0, 0, 1, "recall") if damage == "recall request" else key()
    )
    text = (
        chat.prompted.recall(GAMES["2-AI"]) if identity.kind == "recall" else chat.request(view())
    )
    entry = transcript.append(identity, text, response("0"))
    raw = transcript.path.read_bytes()
    if damage == "line count":
        raw += entry.bytes
    elif damage in ("decision request", "recall request"):
        entry = replace(entry, request="different", request_digest=chat.digest("different"))
        transcript.entries[identity] = entry
        raw = raw.split(b"\n", 1)[0] + b"\n" + entry.bytes
    elif damage == "disk entry":
        raw = raw.split(b"\n", 1)[0] + b"\n" + replace(entry, response=response("1")).bytes
    elif damage == "round order":
        changed_key = replace(identity, round=1)
        entry = replace(entry, identity=changed_key)
        transcript.entries = {changed_key: entry}
        transcript.lines = {changed_key: 2}
        raw = raw.split(b"\n", 1)[0] + b"\n" + entry.bytes
    else:
        changed_key = replace(identity, attempt=2)
        extra = replace(entry, identity=changed_key, response=response("no"))
        transcript.entries[changed_key] = extra
        transcript.lines[changed_key] = 3
        raw += extra.bytes
    transcript.path.write_bytes(raw)
    expected = transcript.digest()
    assert expected == chat.hashlib.sha256(raw).hexdigest()
    with pytest.raises(chat.Unusable) as caught:
        chat.replay_transcript(transcript, expected)
    assert caught.value.cause is chat.Cause.REPLAY


def test_interruption_base_is_never_raised() -> None:
    import ast
    import inspect

    module = ast.parse(inspect.getsource(chat))
    raised = [
        n.exc.func.id
        for n in ast.walk(module)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and isinstance(n.exc.func, ast.Name)
    ]
    assert "Interruption" not in raised
    assert {
        "StorageInterruption",
        "ConsistencyInterruption",
        "ConfigurationInterruption",
        "TransportInterruption",
    } <= set(raised)
