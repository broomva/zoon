"""Offline regressions for the independent chat reviews; no service access."""

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
    response,
    session,
    view,
)
from tests.test_interaction_chat import (
    offline as offline,
)
from zoon.interaction import chat, kernel
from zoon.interaction.games import GAMES


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
    ("text", "expected"), [("1.", None), ("36\r\n", 36), ("\x1f36", None), ("\x0b36", None)]
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
    resumed = chat.Session(chat.Transcript(tmp_path, BINDING), client, terminal=run.terminal)
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
    resumed = chat.Session(chat.Transcript(tmp_path, BINDING), client, terminal=run.terminal)
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


@pytest.mark.parametrize("failure", ["write", "short", "interrupt", "sync"])
def test_partial_write_rolls_back_and_next_append_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING)
    transcript.append(key(), chat.request(view()), response())
    before = transcript.path.read_bytes()
    original: Any = Path.open

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
            if failure == "interrupt":
                raise KeyboardInterrupt
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
        with pytest.raises(chat.StorageInterruption, match="write interrupted"):
            transcript.append(key(1), chat.request(view(1)), response())
    assert transcript.path.read_bytes() == before
    assert key(1) not in transcript.entries
    transcript.append(key(1), chat.request(view(1)), response())
    assert len(chat.Transcript(tmp_path, BINDING).entries) == 2


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
    assert session(tmp_path, Replies([])).tail_removals == 0


def test_append_refuses_torn_tail_without_modifying_it(tmp_path: Path) -> None:
    transcript = chat.Transcript(tmp_path, BINDING)
    transcript.append(key(), chat.request(view()), response())
    raw = transcript.path.read_bytes() + b"fragment"
    transcript.path.write_bytes(raw)
    with pytest.raises(chat.GuardRefusal, match="incomplete"):
        transcript.append(key(1), chat.request(view(1)), response())
    assert transcript.path.read_bytes() == raw


@pytest.mark.parametrize("damage", ["blank", "noncanonical header", "foreign line"])
def test_complete_damage_is_never_repaired(tmp_path: Path, damage: str) -> None:
    transcript = chat.Transcript(tmp_path, BINDING)
    transcript.append(key(), chat.request(view()), response())
    raw = transcript.path.read_bytes()
    if damage == "blank":
        raw += b"\n"
    elif damage == "noncanonical header":
        raw = b" " + raw
    else:
        other = chat.Transcript(tmp_path / "other", replace(BINDING, mode="live"))
        other.append(key(1), chat.request(view(1)), response())
        raw += other.path.read_bytes().split(b"\n")[1] + b"\n"
    raw += b"torn"
    transcript.path.write_bytes(raw)
    with pytest.raises(chat.GuardRefusal):
        chat.Transcript(tmp_path, BINDING)
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


def test_absent_output_counts_are_zero_only_without_candidate_text() -> None:
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
    tainted = chat.Transcript(tmp_path / "tainted", BINDING)
    tainted.append(key(), text, reply)
    if secret_kind == "project":
        monkeypatch.setenv("ZOON_VERTEX_PROJECT", secret)
    bearer = secret if secret_kind == "bearer" else ""
    clean = chat.Transcript(tmp_path / "clean", BINDING, bearer=bearer)
    with pytest.raises(chat.GuardRefusal, match="connection metadata refused"):
        clean.append(key(), text, reply)
    assert not clean.path.exists()
    raw = tainted.path.read_bytes()
    with pytest.raises(chat.GuardRefusal, match="connection metadata refused"):
        chat.Transcript(tmp_path / "tainted", BINDING, bearer=bearer)
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
    client = chat.VertexClient(chat.post, pause=pauses.append)
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
            )
        else:
            client = chat.VertexClient(chat.post, token=lambda: "safe", pause=lambda _: None)

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
        chat.Transcript(tmp_path, BINDING)


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
    restored = chat.Transcript(tmp_path, BINDING)
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
    transcript = chat.Transcript(tmp_path, BINDING)
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

    client = chat.VertexClient(transport, token=token)
    with pytest.raises(chat.ConfigurationInterruption) as caught:
        client.ask("question")
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
