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
from typing import Any

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
        == (Decimal(len(text.encode("utf-8")) + 400) * 4 + Decimal(2048) * 20) / 1_000_000
    )


def test_version_change_in_recall_is_paid_but_not_judged(tmp_path: Path) -> None:
    client = Replies([response("unknown"), response("Wang", version=chat.MODEL + "-002")])
    run = session(tmp_path, client)
    result = chat.recall(run)
    assert chat.status(run.transcript, run.cap) == chat.Cause.VERSION
    assert result[0].status == "judged"
    assert result[1].response == response("Wang", version=chat.MODEL + "-002")
    assert result[1].status == "not judged: version" and not result[1].flagged
    assert all(r.status == "not asked" for r in result[2:])
    assert len(run.transcript.entries) == 2
    assert run.transcript.spent == 2 * chat.cost(response())
    assert chat.recall(run) == result


def test_equal_cap_is_allowed_and_recorded_answer_obeys_status(tmp_path: Path) -> None:
    paid = response(input_tokens=100000, output_tokens=0, thinking_tokens=0)
    from zoon.interaction import budget

    empty = session(tmp_path / "empty", Replies([]), budget.chat_max_estimate())
    assert chat.status(empty.transcript, empty.cap) is None
    assert chat.status(empty.transcript, empty.cap, recall=True) is None
    run = session(tmp_path, Replies([paid]), chat.cost(paid) + budget.chat_max_estimate())
    assert run.decision(view(), GROUP).points == 12
    run.cap = Decimal("0")
    with pytest.raises(chat.Unusable) as caught:
        run.decision(view(), GROUP)
    assert caught.value.cause is chat.Cause.CAP


def test_usable_third_reply_is_accepted_and_later_decisions_continue(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([response("no"), response("no"), response("1"), response("2")]))
    assert run.decision(view(), GROUP).points == 1
    assert run.decision(view(1), GROUP).points == 2
    assert chat.status(run.transcript, run.cap) is None


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
                "maxOutputTokens": 2048,
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
        chat.replay(log, chat.replay_transcript(run.transcript, run.transcript.digest()))


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
            chat.replay(result.bytes, chat.replay_transcript(restored, expected))


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


@pytest.mark.parametrize("later", [chat.Cause.VERSION, chat.Cause.CAP, chat.Cause.WITHDRAWN])
def test_first_arm_cause_stands_when_something_else_stops_recall(
    tmp_path: Path, later: chat.Cause
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    for attempt in range(1, 4):
        transcript.append(key(attempt=attempt), chat.request(view()), response("no"))
    replies = (
        [response("Wang", version=chat.MODEL + "-002")]
        if later is chat.Cause.VERSION
        else [response("Wang", input_tokens=1_000_000)]
        if later is chat.Cause.CAP
        else [chat.Unusable(later)]
    )
    client = Replies(replies)
    run = chat.Session(transcript, client, Decimal("1"))
    result = chat.recall(run)
    assert chat.status(transcript, run.cap) is chat.Cause.REPLIES
    assert result.stopped_by is later
    assert len(client.requests) == 1
    if later is chat.Cause.VERSION:
        assert result[0].response == replies[0]
        assert result[0].status == "not judged: version" and not result[0].flagged
    elif later is chat.Cause.CAP:
        assert result[0].response == replies[0]
        assert result[0].status == "judged" and result[0].flagged
        assert run.transcript.spent > run.cap
        assert len(run.transcript.entries) == 4
    else:
        assert result[0].status == "not asked" and result[0].response is None
    assert all(r.status == "not asked" for r in result[1:])


def test_recorded_version_invalid_answer_is_never_returned_on_resume(tmp_path: Path) -> None:
    run = session(tmp_path, Replies([response(version="other")]))
    with pytest.raises(chat.Unusable):
        run.decision(view(), GROUP)
    resumed = session(tmp_path, Replies([]))
    with pytest.raises(chat.Unusable) as caught:
        resumed.decision(view(), GROUP)
    assert caught.value.cause is chat.status(resumed.transcript, resumed.cap) is chat.Cause.VERSION


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
    assert chat.status(run.transcript, run.cap) is result.stopped_by is chat.Cause.VERSION
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


@pytest.mark.parametrize("header_only", [False, True])
def test_read_only_opening_and_replay_refuse_torn_tail_without_writes(
    tmp_path: Path, header_only: bool
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    log = chat.play(GAMES["2-AI"], chat.Session(transcript, chat.StubClient()), GROUP).bytes
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
    with pytest.raises(chat.GuardRefusal, match="incomplete"):
        chat.replay(log, chat.replay_transcript(transcript, transcript.digest()))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    with pytest.raises(chat.GuardRefusal, match="incomplete"):
        chat.Transcript(tmp_path, BINDING, read_only=False)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


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


@pytest.mark.parametrize("operation", ["open", "digest", "replay", "replay transcript"])
def test_read_failure_is_storage_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    log = chat.play(GAMES["2-AI"], chat.Session(transcript, chat.StubClient()), GROUP).bytes
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
            chat.replay(log, chat.replay_transcript(transcript, transcript.digest()))
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
    if damage == "disk entry":
        chat.replay_transcript(transcript, expected)
    else:
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


@pytest.mark.parametrize("failure", ["write", "short", "flush", "sync", "interrupt", "exit"])
def test_failed_append_leaves_disk_and_requires_a_new_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    before = transcript.path.read_bytes()
    original: Any = Path.open
    interrupt = KeyboardInterrupt() if failure == "interrupt" else SystemExit()
    events: list[str] = []

    class Fault:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.handle.close()

        def fileno(self) -> int:
            return self.handle.fileno()

        def write(self, raw: bytes) -> int:
            events.append("write")
            complete = failure in ("flush", "sync")
            count = self.handle.write(raw if complete else raw[: len(raw) // 2])
            if failure in ("interrupt", "exit"):
                raise interrupt
            if failure == "write":
                raise OSError("failed write")
            return count

        def flush(self) -> None:
            events.append("flush")
            if failure == "flush":
                raise OSError("failed flush")
            self.handle.flush()

    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original(path, *args, **kwargs)
        return Fault(handle) if path == transcript.path and args == ("ab",) else handle

    def sync(fd: int) -> None:
        events.append("sync")
        raise OSError("failed sync")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", opening)
        patch.setattr(chat.os, "fsync", sync)
        expected = type(interrupt) if failure in ("interrupt", "exit") else chat.StorageInterruption
        with pytest.raises(expected):
            transcript.append(key(1), chat.request(view(1)), response())
    complete = failure in ("flush", "sync")
    assert events == (
        ["write", "flush", "sync"]
        if failure == "sync"
        else ["write", "flush"]
        if failure == "flush"
        else ["write"]
    )
    assert len(transcript.entries) == 1 and key(1) not in transcript.entries
    raw = transcript.path.read_bytes()
    assert raw.startswith(before) and len(raw) > len(before)
    for operation in (
        lambda: transcript.append(key(1), chat.request(view(1)), response()),
        lambda: transcript.spent,
        transcript.read_bytes,
        transcript.digest,
        lambda: chat.status(transcript, Decimal("1000")),
        lambda: chat.read_recall(transcript),
        lambda: chat.Session(transcript, Replies([])).ask(key(), chat.request(view())),
    ):
        with pytest.raises(chat.StorageInterruption):
            operation()
    if complete:
        reopened = chat.Transcript(tmp_path, BINDING)
        assert len(reopened.entries) == 2 and reopened.spent == 2 * chat.cost(response())
    else:
        with pytest.raises(chat.GuardRefusal, match="incomplete"):
            chat.Transcript(tmp_path, BINDING)
    assert transcript.path.read_bytes() == raw


@pytest.mark.parametrize("initial", [False, True])
def test_crash_after_each_append_byte_never_changes_a_complete_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial: bool
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    before = b""
    if not initial:
        transcript.append(key(), chat.request(view()), response())
        before = transcript.path.read_bytes()
    identity = key() if initial else key(1)
    text = chat.request(view(identity.seat))
    entry = chat.Entry(BINDING, identity, text, chat.digest(text), response())
    addition = (kernel.encoded(chat.asdict(BINDING)).encode() if initial else b"") + entry.bytes
    original: Any = Path.open
    cut = 0

    class Crash:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.handle.close()

        def write(self, raw: bytes) -> int:
            assert raw == addition
            self.handle.write(raw[:cut])
            raise KeyboardInterrupt

    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original(path, *args, **kwargs)
        return Crash(handle) if path == transcript.path and args == ("ab",) else handle

    for cut in range(len(addition) + 1):
        transcript.path.write_bytes(before)
        writer = chat.Transcript(tmp_path, BINDING, read_only=False)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", opening)
            with pytest.raises(KeyboardInterrupt):
                writer.append(identity, text, response())
        raw = before + addition[:cut]
        assert transcript.path.read_bytes() == raw
        for read_only in (False, True):
            if raw and not raw.endswith(b"\n"):
                with pytest.raises(
                    chat.GuardRefusal,
                    match=(
                        "incomplete transcript line; cut the file back to the byte "
                        "after its last line feed, or to empty if it has none, then "
                        "repeat the command"
                    ),
                ):
                    chat.Transcript(tmp_path, BINDING, read_only=read_only)
            else:
                opened = chat.Transcript(tmp_path, BINDING, read_only=read_only)
                assert len(opened.entries) == bool(before) + (cut == len(addition))
            assert transcript.path.read_bytes() == raw
        assert {p.name for p in tmp_path.iterdir()} == {"chat.jsonl"}


@pytest.mark.parametrize("ending", ["complete", "replies"])
def test_resume_at_every_entry_boundary(tmp_path: Path, ending: str) -> None:
    replies = [response("0")] * (40 if ending == "complete" else 7)
    if ending == "replies":
        replies += [response("no")] * 3
    replies += [response("Wang")] * 8
    baseline = session(tmp_path / "whole", Replies(replies))
    result: kernel.Run | None = None
    if ending == "complete":
        result = chat.play(GAMES["2-AI"], baseline, GROUP)
    else:
        with pytest.raises(chat.Unusable) as caught:
            chat.play(GAMES["2-AI"], baseline, GROUP)
        assert caught.value.cause is chat.Cause.REPLIES
    recall = chat.recall(baseline)
    expected = baseline.transcript.path.read_bytes()
    lines = expected.splitlines(keepends=True)
    for boundary in range(1, len(lines) + 1):
        directory = tmp_path / str(boundary)
        directory.mkdir()
        (directory / "chat.jsonl").write_bytes(b"".join(lines[:boundary]))
        client = Replies(replies[boundary - 1 :])
        resumed = session(directory, client)
        if ending == "complete":
            assert result is not None
            assert chat.play(GAMES["2-AI"], resumed, GROUP).bytes == result.bytes
        else:
            with pytest.raises(chat.Unusable) as caught:
                chat.play(GAMES["2-AI"], resumed, GROUP)
            assert caught.value.cause is chat.Cause.REPLIES
        assert chat.recall(resumed) == recall
        assert resumed.transcript.path.read_bytes() == expected
        assert len(client.requests) == len(lines) - boundary
        chat.replay_transcript(resumed.transcript, baseline.transcript.digest())


@pytest.mark.parametrize("cause", [chat.Cause.VERSION, chat.Cause.REPLIES, chat.Cause.CAP])
def test_status_agrees_with_fresh_sessions_at_many_prefixes(
    tmp_path: Path, cause: chat.Cause
) -> None:
    cap = Decimal("1")
    replies = [response("0")] * 12
    replies += (
        [response(version="foreign")]
        if cause is chat.Cause.VERSION
        else [response("no")] * 3
        if cause is chat.Cause.REPLIES
        else [response(input_tokens=1_000_000)]
    )
    writer = session(tmp_path, Replies(replies), cap)
    keys = [chat.Identity("2-AI", f"registered|2-AI|{n}", 0, 0, 1) for n in range(13)]
    if cause is chat.Cause.REPLIES:
        keys += [replace(keys[-1], attempt=n) for n in (2, 3)]
    for index, identity in enumerate(keys):
        expected = cause if index == len(keys) - 1 else None
        if expected:
            with pytest.raises(chat.Unusable) as caught:
                writer.ask(identity, chat.request(view()))
            assert caught.value.cause is expected
        else:
            writer.ask(identity, chat.request(view()))
        resumed = session(tmp_path, Replies([]), cap)
        assert (
            chat.status(writer.transcript, cap) is chat.status(resumed.transcript, cap) is expected
        )
        for run in (writer, resumed):
            if expected:
                with pytest.raises(chat.Unusable) as caught:
                    run.ask(keys[0], chat.request(view()))
                assert caught.value.cause is expected
            else:
                run.ask(keys[0], chat.request(view()))
    # Later causes never replace the first one in file order.
    writer.transcript.append(
        chat.Identity("2-AI", "recall", 0, 0, 1, "recall"),
        chat.prompted.recall(GAMES["2-AI"]),
        response(version="foreign", input_tokens=2_000_000),
    )
    assert chat.status(writer.transcript, cap) is cause
    assert chat.status(session(tmp_path, Replies([]), cap).transcript, cap) is cause


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_cap_refusal_depends_only_on_spending(tmp_path: Path, workers: int, reverse: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from zoon.interaction import budget

    addition = budget.chat_max_estimate()
    maximum = max(
        (
            budget.chat_bound_request(g, s, r)
            for g in GAMES.values()
            for s in range(g.size)
            for r in range(1, chat.ROUNDS + 1)
        ),
        key=chat.estimate,
    )
    paid = response(
        "0",
        input_tokens=chat.input_tokens(maximum),
        output_tokens=chat.MAX_OUTPUT_TOKENS,
        thinking_tokens=0,
    )
    assert chat.cost(paid) == addition
    cap = 3 * addition + (chat.estimate(chat.request(view())) + addition) / 2
    run = session(tmp_path, Replies([paid] * 3), cap)
    identities = [chat.Identity("2-AI", f"registered|2-AI|{n}", 0, 0, 1) for n in range(8)]
    texts = [chat.request(view()), maximum] * 4
    if reverse:
        texts.reverse()

    def ask(pair: tuple[chat.Identity, str]) -> bool:
        try:
            run.ask(*pair)
        except chat.Unusable as error:
            assert error.cause is chat.Cause.CAP
            return False
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ask, zip(identities, texts, strict=True)))
    assert sum(results) == 2 and run.transcript.spent == 3 * addition
    assert len(run.transcript.entries) == 3
    assert chat.status(run.transcript, run.cap) is chat.Cause.CAP
    assert chat.recall(run).stopped_by is chat.Cause.CAP
    assert not run.pending and run.reserved == 0
    for text in texts:
        resumed = session(tmp_path, Replies([]), run.cap)
        assert chat.status(resumed.transcript, run.cap) is chat.Cause.CAP
        with pytest.raises(chat.Unusable) as caught:
            resumed.ask(chat.Identity("2-AI", "registered|2-AI|99", 0, 0, 1), text)
        assert caught.value.cause is chat.Cause.CAP


@pytest.mark.parametrize("interrupted", [False, True])
def test_cap_waits_for_inflight_calls_and_wakes_after_completion(
    tmp_path: Path, interrupted: bool
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from zoon.interaction import budget

    entered, release, waited = Event(), Event(), Event()

    class Blocking:
        def ask(self, text: str) -> chat.Response:
            if text == "first":
                entered.set()
                assert release.wait(5)
                if interrupted:
                    raise chat.TransportInterruption("retry later")
            return response("0")

    run = session(tmp_path, Blocking(), budget.chat_max_estimate() + 2 * chat.cost(response("0")))
    wait = run._lock.wait  # pyright: ignore[reportPrivateUsage]

    def waiting(timeout: float | None = None) -> bool:
        waited.set()
        return wait(10 if timeout is None else timeout)

    run._lock.wait = waiting  # pyright: ignore[reportPrivateUsage]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run.ask, key(), "first")
        assert entered.wait(5)
        with pytest.raises(chat.ConsistencyInterruption, match="already in flight"):
            run.ask(key(), "first")
        second = pool.submit(run.ask, key(1), "second")
        try:
            assert waited.wait(5) and not second.done()
        finally:
            release.set()
        if interrupted:
            with pytest.raises(chat.TransportInterruption):
                first.result(timeout=5)
        else:
            assert first.result(timeout=5).response.text == "0"
        assert second.result(timeout=5).response.text == "0"
    assert not run.pending and run.reserved == 0


@pytest.mark.parametrize(
    "field", ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"]
)
def test_negative_reply_count_is_retried_and_can_recover(field: str) -> None:
    data = payload()
    data["usageMetadata"][field] = -1
    client, calls, pauses = fake_client([(200, data), (200, payload())])
    assert client.ask("question").text == "12"
    assert len(calls) == 2 and pauses == [1]
    client, calls, pauses = fake_client([(200, data)] * 4)
    with pytest.raises(chat.TransportInterruption):
        client.ask("question")
    assert len(calls) == 4 and pauses == [1, 2, 4]


@pytest.mark.parametrize("failure", ["status", "project"])
def test_failed_second_preflight_withdraws_the_first_success(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    states = iter([200, 403])
    calls: list[str] = []

    def generating(*args: Any) -> Any:
        calls.append("generate")
        return generation_ok()

    def metadata(*args: Any) -> tuple[int, dict[str, Any]]:
        return next(states), {}

    client = chat.VertexClient(generating, token=lambda: "token", preflight_transport=metadata)
    client.preflight()
    if failure == "project":
        monkeypatch.delenv("ZOON_VERTEX_PROJECT")
    with pytest.raises(chat.ConfigurationInterruption):
        client.preflight()
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "connection-location-sentinel")
    with pytest.raises(chat.ConfigurationInterruption, match="preflight"):
        client.ask("question")
    assert not calls


@pytest.mark.parametrize("cause", [chat.Cause.WITHDRAWN, chat.Cause.REPLAY])
def test_unrecorded_outcome_is_raised_without_session_state(
    tmp_path: Path, cause: chat.Cause
) -> None:
    run = session(tmp_path, Replies([chat.Unusable(cause), response("0")]))
    with pytest.raises(chat.Unusable) as caught:
        run.decision(view(), GROUP)
    assert caught.value.cause is cause
    assert not run.transcript.entries
    assert chat.status(run.transcript, run.cap) is None
    assert run.decision(view(), GROUP).points == 0
    assert session(tmp_path, Replies([])).decision(view(), GROUP).points == 0


def test_recall_reader_derives_all_labels_without_a_client(tmp_path: Path) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    cells = sorted(GAMES)
    for cell, reply in zip(
        cells[:2], [response("Wang"), response("Hilbe", version="foreign")], strict=True
    ):
        transcript.append(
            chat.Identity(cell, "recall", 0, 0, 1, "recall"),
            chat.prompted.recall(GAMES[cell]),
            reply,
        )
    before = transcript.path.read_bytes()
    replies = chat.read_recall(chat.Transcript(tmp_path, BINDING))
    assert len(replies) == 8
    assert (replies[0].status, replies[0].flagged) == ("judged", True)
    assert (replies[1].status, replies[1].flagged) == ("not judged: version", False)
    assert all(
        r.status == "not asked" and r.response is None and not r.flagged for r in replies[2:]
    )
    result = chat.recall(chat.Session(transcript, Replies([])))
    assert result.replies == replies and result.stopped_by is chat.Cause.VERSION
    assert transcript.path.read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "count",
        "request",
        "recall",
        "first attempt",
        "gap",
        "accepted",
        "excess",
        "round",
        "missing",
        "torn",
        "stale memory",
    ],
)
def test_replay_uses_disk_and_checks_complete_attempt_history(tmp_path: Path, damage: str) -> None:
    run = session(
        tmp_path,
        Replies([response("no"), response("0")] + [response("0")] * 39 + [response("unknown")] * 8),
    )
    log = chat.play(GAMES["2-AI"], run, GROUP).bytes
    chat.recall(run)
    transcript = run.transcript
    expected = transcript.digest()
    lines = transcript.path.read_bytes().splitlines(keepends=True)
    if damage == "digest":
        expected = "0" * 64
    elif damage == "count":
        lines.pop()
    elif damage in ("request", "recall"):
        index = 1 if damage == "request" else len(lines) - 1
        row = json.loads(lines[index])
        row["request"] = "different"
        row["request_digest"] = chat.digest("different")
        lines[index] = kernel.encoded(row).encode()
    elif damage in ("first attempt", "gap", "excess"):
        index = 1 if damage == "first attempt" else 2
        row = json.loads(lines[index])
        row["identity"]["attempt"] = {"first attempt": 2, "gap": 3, "excess": 4}[damage]
        lines[index] = kernel.encoded(row).encode()
    elif damage == "accepted":
        row = json.loads(lines[1])
        row["response"]["text"] = "0"
        lines[1] = kernel.encoded(row).encode()
    elif damage == "round":
        # Round two cannot precede the accepted reply of the last seat in round one.
        lines[3], lines[4] = lines[4], lines[3]
    elif damage == "torn":
        lines[-1] = lines[-1][:-4]
    elif damage == "stale memory":
        transcript.entries[key()] = replace(transcript.entries[key()], request="memory only")
    if damage == "missing":
        transcript.path.rename(tmp_path / "saved.jsonl")
    else:
        transcript.path.write_bytes(b"".join(lines))
    if damage not in ("digest", "missing", "torn"):
        expected = transcript.digest()
    if damage in ("missing", "torn"):
        for operation in (
            lambda: chat.replay(log, chat.replay_transcript(transcript, transcript.digest())),
            lambda: chat.replay_transcript(transcript, expected),
        ):
            with pytest.raises(chat.GuardRefusal):
                operation()
    elif damage == "stale memory":
        assert (
            chat.replay(log, chat.replay_transcript(transcript, transcript.digest())).bytes == log
        )
        chat.replay_transcript(transcript, expected)
    else:
        with pytest.raises(chat.Unusable) as caught:
            chat.replay_transcript(transcript, expected)
        assert caught.value.cause is chat.Cause.REPLAY
        if damage != "digest":
            with pytest.raises(chat.Unusable) as caught:
                chat.replay(log, chat.replay_transcript(transcript, transcript.digest()))
            assert caught.value.cause is chat.Cause.REPLAY


@pytest.mark.parametrize("cause", [chat.Cause.VERSION, chat.Cause.REPLIES, chat.Cause.CAP])
def test_unusable_arm_has_a_replayable_record_and_no_game_score(
    tmp_path: Path, cause: chat.Cause
) -> None:
    run = session(tmp_path, chat.StubClient())
    log = chat.play(GAMES["2-AI"], run, GROUP).bytes
    if cause is chat.Cause.REPLIES:
        for attempt in range(1, 4):
            run.transcript.append(
                chat.Identity("2-AI", "registered|2-AI|1", 0, 0, attempt),
                chat.request(view()),
                response("no", version=chat.MODEL + "-stub"),
            )
    else:
        run.transcript.append(
            chat.Identity("2-AI", "registered|2-AI|1", 0, 0, 1),
            chat.request(view()),
            response(
                version="foreign" if cause is chat.Cause.VERSION else chat.MODEL + "-stub",
                input_tokens=300_000_000 if cause is chat.Cause.CAP else 0,
            ),
        )
    if cause is chat.Cause.REPLIES:
        chat.recall(run)
    chat.replay_transcript(run.transcript, run.transcript.digest())
    assert chat.status(run.transcript, run.cap) is cause
    with pytest.raises(chat.Unusable) as caught:
        chat.replay(log, chat.replay_transcript(run.transcript, run.transcript.digest()), run.cap)
    assert caught.value.cause is cause


def test_replay_refuses_a_fourth_attempt_even_when_the_first_three_failed(tmp_path: Path) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    for attempt in range(1, 4):
        transcript.append(key(attempt=attempt), chat.request(view()), response("no"))
    transcript.append(key(1), chat.request(view(1)), response("0"))
    lines = transcript.path.read_bytes().splitlines(keepends=True)
    fourth = json.loads(lines[-1])
    fourth["identity"].update(seat=0, attempt=4)
    fourth["request"] = chat.request(view())
    fourth["request_digest"] = chat.digest(fourth["request"])
    lines[-1] = kernel.encoded(fourth).encode()
    transcript.path.write_bytes(b"".join(lines))
    with pytest.raises(chat.Unusable) as caught:
        chat.replay_transcript(transcript, transcript.digest())
    assert caught.value.cause is chat.Cause.REPLAY


def test_status_uses_file_order_before_identity_order(tmp_path: Path) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    for attempt in range(1, 4):
        transcript.append(
            chat.Identity("2-AI", "registered|2-AI|9", 0, 0, attempt),
            chat.request(view()),
            response("no"),
        )
    transcript.append(key(), chat.request(view()), response(version="foreign"))
    assert chat.status(transcript, Decimal("1000")) is chat.Cause.REPLIES


@pytest.mark.parametrize("kind", ["decision", "recall"])
def test_status_accumulates_spending_from_every_kind(tmp_path: Path, kind: str) -> None:
    from zoon.interaction import budget

    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    paid = response("0", input_tokens=10_000, output_tokens=0, thinking_tokens=0)
    cap = budget.chat_max_estimate() + chat.cost(paid) * Decimal("2.5")
    for index, cell in enumerate(sorted(GAMES)[:3]):
        identity = (
            chat.Identity(cell, "recall", 0, 0, 1, "recall")
            if kind == "recall"
            else chat.Identity("2-AI", f"registered|2-AI|{index}", 0, 0, 1)
        )
        text = chat.prompted.recall(GAMES[cell]) if kind == "recall" else chat.request(view())
        transcript.append(identity, text, paid)
        expected = chat.Cause.CAP if index == 2 else None
        assert chat.status(transcript, cap) is expected
        assert chat.status(transcript, cap, recall=True) is expected


@pytest.mark.parametrize("changed", [False, True])
def test_status_precedence_at_one_entry(tmp_path: Path, changed: bool) -> None:
    from zoon.interaction import budget

    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    for attempt in (1, 2):
        transcript.append(key(attempt=attempt), chat.request(view()), response("no"))
    cap = transcript.spent + budget.chat_max_estimate() + Decimal("0.1")
    transcript.append(
        key(attempt=3),
        chat.request(view()),
        response("no", version="foreign" if changed else chat.MODEL + "-001", input_tokens=100_000),
    )
    assert chat.status(transcript, cap) is (chat.Cause.VERSION if changed else chat.Cause.REPLIES)
    assert chat.status(transcript, cap, recall=True) is (
        chat.Cause.VERSION if changed else chat.Cause.CAP
    )


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("line", ["header", "entry"])
def test_transcript_checks_run_identity_on_every_line(
    tmp_path: Path, read_only: bool, line: str
) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    transcript.append(key(), chat.request(view()), response())
    lines = transcript.path.read_bytes().splitlines(keepends=True)
    index = 0 if line == "header" else 1
    row = json.loads(lines[index])
    (row if line == "header" else row["run"])["registered_commit"] = "b" * 40
    lines[index] = kernel.encoded(row).encode()
    raw = b"".join(lines)
    transcript.path.write_bytes(raw)
    with pytest.raises(chat.GuardRefusal, match="run identity differs"):
        chat.Transcript(tmp_path, BINDING, read_only=read_only)
    assert transcript.path.read_bytes() == raw


@pytest.mark.parametrize("recalling", [False, True])
def test_empty_transcript_can_already_have_reached_cap(tmp_path: Path, recalling: bool) -> None:
    from zoon.interaction import budget

    run = session(tmp_path, Replies([]), budget.chat_max_estimate() - Decimal("0.000001"))
    assert chat.status(run.transcript, run.cap, recall=recalling) is chat.Cause.CAP
    if recalling:
        result = chat.recall(run)
        assert result.stopped_by is chat.Cause.CAP
        assert all(r.status == "not asked" for r in result)
    else:
        with pytest.raises(chat.Unusable) as caught:
            run.decision(view(), GROUP)
        assert caught.value.cause is chat.Cause.CAP
    assert not run.transcript.entries and not run.transcript.path.exists()


def test_cap_refusal_is_recorded_status_for_replay_and_recall(tmp_path: Path) -> None:
    from zoon.interaction import budget

    run = session(tmp_path, chat.StubClient())
    log = chat.play(GAMES["2-AI"], run, GROUP).bytes
    paid = response("0", version=chat.MODEL + "-stub", input_tokens=10_000)
    run.cap = run.transcript.spent + budget.chat_max_estimate() + chat.cost(paid) / 2
    run.client = Replies([paid])
    with pytest.raises(chat.Unusable) as caught:
        run.ask(chat.Identity("2-AI", "registered|2-AI|1", 0, 0, 1), chat.request(view()))
    assert caught.value.cause is chat.Cause.CAP
    assert run.transcript.spent < run.cap
    verified = chat.replay_transcript(run.transcript, run.transcript.digest())
    for current in (run, session(tmp_path, Replies([]), run.cap)):
        assert chat.status(current.transcript, current.cap) is chat.Cause.CAP
        assert chat.recall(current).stopped_by is chat.Cause.CAP
    with pytest.raises(chat.Unusable) as caught:
        chat.replay(log, verified, run.cap)
    assert caught.value.cause is chat.Cause.CAP


@pytest.mark.parametrize("value", [None, "transcript", object()])
def test_replay_requires_verification(tmp_path: Path, value: Any) -> None:
    transcript = chat.Transcript(tmp_path, BINDING)
    with pytest.raises(chat.GuardRefusal, match="verified transcript"):
        chat.replay(b"unused", transcript if value == "transcript" else value)  # pyright: ignore[reportArgumentType]


def test_verification_reads_once_for_several_game_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import FrozenInstanceError

    run = session(tmp_path, chat.StubClient())
    logs = [chat.play(GAMES["2-AI"], run, f"registered|2-AI|{index}").bytes for index in range(3)]
    expected = run.transcript.digest()
    original = Path.read_bytes
    reads: list[Path] = []

    def read(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    verified = chat.replay_transcript(run.transcript, expected)
    assert verified.digest == expected and verified.transcript is not run.transcript
    with pytest.raises(FrozenInstanceError):
        verified.digest = "changed"  # pyright: ignore[reportAttributeAccessIssue]
    for log in logs:
        assert chat.replay(log, verified).bytes == log
    run.transcript.path.write_bytes(b"changed after verification\n")
    assert chat.replay(logs[0], verified).bytes == logs[0]
    assert reads == [run.transcript.path]


def test_recorded_recall_must_match_registered_question(tmp_path: Path) -> None:
    transcript = chat.Transcript(tmp_path, BINDING, read_only=False)
    for cell, game in sorted(GAMES.items()):
        transcript.append(
            chat.Identity(cell, "recall", 0, 0, 1, "recall"),
            "another question" if cell == "2-AI" else chat.prompted.recall(game),
            response("Wang"),
        )
    for current in (transcript, chat.Transcript(tmp_path, BINDING)):
        with pytest.raises(chat.GuardRefusal, match="recall request changed"):
            chat.read_recall(current)
        with pytest.raises(chat.GuardRefusal, match="recall request changed"):
            chat.recall(chat.Session(current, Replies([])))
