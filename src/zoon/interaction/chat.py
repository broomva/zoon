"""Point replies, per-decision transcripts and exact replay for the chat arm.

Round and seat identifiers are zero-based, as in the kernel. Reply attempts start
at one. Transport retries do not advance them. The run driver owns locking the
run directory and persisting arm outcomes; this module never reads human data.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Literal, Protocol, cast, overload

import numpy as np

from zoon.interaction import chat_prices, kernel, measure, prompted
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.pilot import recognition

MODEL = chat_prices.HIGH.model
LOCATION = "global"
TEMPERATURE = 1.0
THINKING_LEVEL = "low"
MAX_OUTPUT_TOKENS = 2048
TRANSPORT_ATTEMPTS = 4
TRANSPORT_PAUSES = (1.0, 2.0, 4.0)
REPLY_ATTEMPTS = 3
INPUT_BYTES_PER_TOKEN = 1
INPUT_OVERHEAD = 400
SMOKE_REPEATS = 3
SMOKE_DEFAULT = "stub"
# Instructions already end in a newline. No newline follows the question.
INSTRUCTION_SEPARATOR = "\n"
SECTION_SEPARATOR = "\n\n"
ROUND_SEPARATOR = SECTION_SEPARATOR
EMPTY_HISTORY = "Round 1: there is no earlier history."
RECALL_KEYS = ("wang", "hilbe", "zhang", "asymmetric public goods")
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_" + category, "threshold": "OFF"}
    for category in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")
]
HTTP = Callable[[str, dict[str, str], bytes], tuple[int, dict[str, Any]]]


def request(view: kernel.View) -> str:
    # Use the existing renderer to validate the complete, open-channel history.
    prompted.request(view)
    history = ROUND_SEPARATOR.join(
        f"Round {r}\n"
        + prompted.round_text(
            view.game, tuple(dict(o.contributions)[s] for s in range(view.game.size))
        )
        for r, o in enumerate(view.history, 1)
    )
    return (
        prompted.instructions(view.game)
        + INSTRUCTION_SEPARATOR
        + f"You are Player {view.seat + 1}."
        + SECTION_SEPARATOR
        + (history or EMPTY_HISTORY)
        + SECTION_SEPARATOR
        + f"How many of your {view.game.endowments[view.seat]} points do you contribute "
        + f"to the common pool in round {len(view.history) + 1}? "
        + "Answer with one whole number of points and nothing else."
    )


def points(text: str, endowment: int) -> int | None:
    """Trim only space, tab, LF and CR; accept ASCII digits including leading zeroes."""
    text = text.strip(" \t\n\r")
    if not re.fullmatch(r"[0-9]+", text):
        return None
    # Avoid Python's integer-string length limit even for pathological replies.
    digits = text.lstrip("0") or "0"
    if len(digits) > len(str(endowment)):
        return None
    value = int(digits)
    return value if value <= endowment else None


def input_tokens(text: str) -> int:
    """One token per UTF-8 byte, plus an allowance for the single user envelope.

    This assumes byte-fallback tokenisation, not a measured tokenizer guarantee.
    """
    size = len(text.encode("utf-8"))
    return (size + INPUT_BYTES_PER_TOKEN - 1) // INPUT_BYTES_PER_TOKEN + INPUT_OVERHEAD


@dataclass(frozen=True)
class Response:
    text: str
    version: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0


def cost(response: Response, price: chat_prices.Price = chat_prices.HIGH) -> Decimal:
    return price.dollars(response.input_tokens, response.output_tokens + response.thinking_tokens)


def estimate(text: str) -> Decimal:
    return chat_prices.HIGH.dollars(input_tokens(text), MAX_OUTPUT_TOKENS)


class Cause(Enum):
    REPLIES = "chat unusable: three unusable replies for one decision"
    WITHDRAWN = "chat unusable: model withdrawn"
    VERSION = "chat unusable: model version changed or invalid first version"
    CAP = "chat unusable: spending cap reached"
    REPLAY = "chat unusable: game failed replay"


class Unusable(Exception):
    def __init__(self, cause: Cause) -> None:
        self.cause = cause
        super().__init__(cause.value)


class Interruption(Exception):
    """A failure that consumes no reply attempt; subclasses identify the remedy."""


class TransportInterruption(Interruption):
    """A transient transport fault; repeating the step may succeed."""


class StorageInterruption(Interruption):
    """Storage failed; repeat the command to reopen the transcript."""


class ConfigurationInterruption(Interruption):
    """Credentials or configuration must be corrected before repeating."""


class ConsistencyInterruption(Interruption):
    """Internal or stored data is invalid; repeating cannot repair it."""


class GuardRefusal(Exception):
    pass


class TransientError(Exception):
    pass


class ModelMissing(TransientError):
    pass


def retry(call: Callable[[], Response], pause: Callable[[float], None] = time.sleep) -> Response:
    if TRANSPORT_ATTEMPTS < 1:
        raise ConsistencyInterruption("chat transport attempts must be positive")
    missing = 0
    for attempt in range(TRANSPORT_ATTEMPTS):
        try:
            return call()
        except (
            TransientError,
            TimeoutError,
            ConnectionError,
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
        ) as error:
            missing += isinstance(error, ModelMissing)
        # Raise outside the handler: suppressed context can still contain secrets.
        if attempt + 1 < TRANSPORT_ATTEMPTS:
            pause(TRANSPORT_PAUSES[attempt])
    if missing == TRANSPORT_ATTEMPTS:
        raise Unusable(Cause.WITHDRAWN)
    raise TransportInterruption("chat transport failure after bounded retries")


def protect(value: object, bearer: str = "") -> None:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    for secret in (os.environ.get("ZOON_VERTEX_PROJECT", ""), bearer):
        if secret and secret in encoded:
            raise GuardRefusal("connection metadata refused; no record was written")


def validate(response: Response, bearer: str = "") -> None:
    if (
        not isinstance(cast(object, response.text), str)
        or not isinstance(cast(object, response.version), str)
        or any(
            type(value) is not int or value < 0
            for value in (response.input_tokens, response.output_tokens, response.thinking_tokens)
        )
    ):
        raise ConsistencyInterruption("invalid chat response or token usage")
    protect(asdict(response), bearer)


def _http(
    url: str, headers: dict[str, str], body: bytes, method: str
) -> tuple[int, dict[str, Any]]:
    try:
        outgoing = urllib.request.Request(
            url, data=body if method == "POST" else None, headers=headers, method=method
        )
        with urllib.request.urlopen(outgoing, timeout=30) as reply:
            value = json.loads(reply.read())
            if not isinstance(value, dict):
                raise ValueError
            return reply.status, cast(dict[str, Any], value)
    except urllib.error.HTTPError as error:
        return error.code, {}
    except (ValueError, TypeError):
        pass
    raise TransientError("invalid chat JSON response")


def post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
    """The live generating transport; callers must explicitly inject it."""
    return _http(url, headers, body, "POST")


def get(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
    """The non-generating metadata transport."""
    return _http(url, headers, body, "GET")


def access_token() -> str:
    try:
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        token = result.stdout.strip()
        if not token:
            raise ValueError
        return token
    except Exception:
        pass
    raise ConfigurationInterruption("chat credentials unavailable")


class Client(Protocol):
    def ask(self, text: str) -> Response: ...


class VertexClient:
    """A step must call preflight before its first generating request.

    Preflight queries model metadata without generating text. Missing metadata
    permits generation; only four missing generating responses mean withdrawn.
    """

    def __init__(
        self,
        transport: HTTP,
        *,
        token: Callable[[], str] = access_token,
        preflight_transport: HTTP = get,
        pause: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport, self.token, self.pause = transport, token, pause
        self.preflight_transport = preflight_transport
        self._preflight_pid: int | None = None
        self._bearer: str | None = None
        self._lock = Lock()

    def bearer(self, rejected: str | None = None) -> str:
        with self._lock:
            if self._bearer is None or self._bearer == rejected:
                try:
                    self._bearer = self.token()
                    if not self._bearer:
                        raise ValueError
                except Exception:
                    self._bearer = None
            if self._bearer is not None:
                return self._bearer
        raise ConfigurationInterruption("chat credentials unavailable")

    def preflight(self) -> None:
        self._preflight_pid = None
        if not os.environ.get("ZOON_VERTEX_PROJECT", ""):
            raise ConfigurationInterruption("ZOON_VERTEX_PROJECT is required")

        def probe() -> Response:
            try:
                bearer = self.bearer()
                status, _ = self.preflight_transport(
                    "https://aiplatform.googleapis.com/v1/publishers/google/models/" + MODEL,
                    {
                        "Authorization": "Bearer " + bearer,
                        "Content-Type": "application/json",
                        "x-goog-user-project": os.environ["ZOON_VERTEX_PROJECT"],
                    },
                    b"",
                )
                if status == 401:
                    self.bearer(rejected=bearer)
                    raise TransientError
                if status == 429 or 500 <= status < 600:
                    raise TransientError
                if status not in (200, 404, 410):
                    raise ConfigurationInterruption("chat metadata request refused")
                return Response("", MODEL, 0, 0)
            except (TransientError, Interruption, GuardRefusal, OSError, http.client.HTTPException):
                raise
            except Exception:
                pass
            raise TransientError("chat metadata processing failed")

        retry(probe, self.pause)
        self._preflight_pid = os.getpid()

    def ask(self, text: str) -> Response:
        if self._preflight_pid != os.getpid():
            raise ConfigurationInterruption("chat preflight is required in this process")
        project = os.environ.get("ZOON_VERTEX_PROJECT", "")
        if not project:
            raise ConfigurationInterruption("ZOON_VERTEX_PROJECT is required")
        protect(text, self.bearer())
        endpoint = (
            "https://aiplatform.googleapis.com/v1/projects/"
            + project
            + "/locations/"
            + LOCATION
            + "/publishers/google/models/"
            + MODEL
            + ":generateContent"
        )
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": text}]}],
                "generationConfig": {
                    "temperature": TEMPERATURE,
                    "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
                    "maxOutputTokens": MAX_OUTPUT_TOKENS,
                },
                "safetySettings": SAFETY_SETTINGS,
            }
        ).encode()

        def call() -> Response:
            try:
                bearer = self.bearer()
                status, data = self.transport(
                    endpoint,
                    {
                        "Authorization": "Bearer " + bearer,
                        "Content-Type": "application/json",
                        "x-goog-user-project": project,
                    },
                    body,
                )
                if status == 401:
                    self.bearer(rejected=bearer)
                if status in (404, 410):
                    raise ModelMissing
                if (400 <= status < 500 and status != 403) or 500 <= status < 600:
                    raise TransientError
                if status != 200:
                    raise ConfigurationInterruption("chat request refused")
                protect(data, bearer)
                texts: list[str] = []
                candidates = data.get("candidates", [])
                for item in cast(list[Any], candidates) if isinstance(candidates, list) else []:
                    content = (
                        cast(dict[str, Any], item).get("content")
                        if isinstance(item, dict)
                        else None
                    )
                    parts = (
                        cast(dict[str, Any], content).get("parts")
                        if isinstance(content, dict)
                        else None
                    )
                    for part in cast(list[Any], parts) if isinstance(parts, list) else []:
                        text = (
                            cast(dict[str, Any], part).get("text")
                            if isinstance(part, dict)
                            else None
                        )
                        if isinstance(text, str):
                            texts.append(text)
                protect("".join(texts), bearer)
                usage = data.get("usageMetadata", {})
                candidates = data.get("candidates", [])
                candidate: Any = candidates[0] if candidates else {}
                blocked = (
                    "blockReason" in data.get("promptFeedback", {})
                    or "blockReason" in candidate
                    or candidate.get("finishReason") not in (None, "STOP", "MAX_TOKENS")
                    or any(
                        rating.get("blocked")
                        for source in (data.get("promptFeedback", {}), candidate)
                        for rating in source.get("safetyRatings", [])
                    )
                )
                parts = candidate.get("content", {}).get("parts", [])
                reply = "" if blocked else "".join(p.get("text", "") for p in parts)
                version = data.get("modelVersion")
                if not isinstance(version, str) or not version.strip():
                    raise TransientError
                if "promptTokenCount" not in usage or any(
                    type(usage[name]) is not int or usage[name] < 0
                    for name in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")
                    if name in usage
                ):
                    raise TransientError
                response = Response(
                    reply,
                    version,
                    usage["promptTokenCount"],
                    usage.get("candidatesTokenCount", 0),
                    usage.get("thoughtsTokenCount", 0),
                )
                return response
            except (
                TransientError,
                Interruption,
                GuardRefusal,
                OSError,
                http.client.HTTPException,
            ):
                raise
            except Exception:
                pass
            raise TransientError("chat response processing failed")

        return retry(call, self.pause)


class StubClient:
    """A stateful seeded stream: equal requests still cause separate calls."""

    def __init__(self, seed: int = prompted.FIXTURE_SEED) -> None:
        self.rng = random.Random(seed)

    def ask(self, text: str) -> Response:
        match = re.search(r"How many of your ([0-9]+) points", text)
        reply = str(self.rng.randrange(int(match[1]) + 1)) if match else "unknown"
        return Response(reply, MODEL + "-stub", 12, 1)


@dataclass(frozen=True)
class Identity:
    game: str
    group: str
    round: int
    seat: int
    attempt: int
    kind: Literal["decision", "recall"] = "decision"

    def __post_init__(self) -> None:
        game = GAMES.get(self.game)
        if game is None:
            raise GuardRefusal("unknown chat game")
        if any(type(v) is not int for v in (self.round, self.seat, self.attempt)):
            raise GuardRefusal("invalid chat decision coordinates")
        if self.kind == "recall":
            valid = (self.group, self.round, self.seat, self.attempt) == ("recall", 0, 0, 1)
        else:
            prefix = f"registered|{game.cell}|"
            suffix = self.group.removeprefix(prefix)
            valid = (
                self.kind == "decision"
                and self.group.startswith(prefix)
                and suffix in {str(n) for n in range(100)}
                and 0 <= self.round < ROUNDS
                and 0 <= self.seat < game.size
                and 1 <= self.attempt <= REPLY_ATTEMPTS
            )
        if not valid:
            raise GuardRefusal("invalid chat decision identity")


@dataclass(frozen=True)
class RunIdentity:
    registered_commit: str
    mode: Literal["stub", "live"]
    arm: str = "chat"

    def __post_init__(self) -> None:
        if (
            not re.fullmatch(r"[0-9a-f]{40}", self.registered_commit)
            or self.mode not in ("stub", "live")
            or self.arm != "chat"
        ):
            raise GuardRefusal("invalid chat run identity")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def complete(raw: bytes) -> bytes:
    """Refuse transcript bytes that end in an incomplete line."""
    if raw and not raw.endswith(b"\n"):
        raise GuardRefusal(
            "incomplete transcript line; cut the file back to the byte after "
            "its last line feed, or to empty if it has none, then repeat the "
            "command"
        )
    return raw


@dataclass(frozen=True)
class Entry:
    run: RunIdentity
    identity: Identity
    request: str
    request_digest: str
    response: Response

    @property
    def bytes(self) -> bytes:
        return kernel.encoded(asdict(self)).encode()


class Transcript:
    def __init__(
        self,
        directory: Path,
        identity: RunIdentity,
        *,
        bearer: str = "",
        read_only: bool = True,
        _raw: bytes | None = None,
    ) -> None:
        self.path = directory / "chat.jsonl"
        self.identity = identity
        self.bearer = bearer
        self.read_only = read_only
        self._failed = False
        self.entries: dict[Identity, Entry] = {}
        self.lines: dict[Identity, int] = {}
        protect(asdict(identity))
        storage_failed = False
        try:
            raw = self.read_bytes() if _raw is None else complete(_raw)
            if raw:
                protect(raw.decode(), self.bearer)
                lines = raw.split(b"\n")[:-1]
                if lines[0] + b"\n" != kernel.encoded(asdict(identity)).encode():
                    raise GuardRefusal("chat transcript run identity differs or is noncanonical")
                for line in lines[1:]:
                    row = json.loads(line)
                    entry = Entry(
                        RunIdentity(**row["run"]),
                        Identity(**row["identity"]),
                        row["request"],
                        row["request_digest"],
                        Response(**row["response"]),
                    )
                    self.check(entry)
                    if line + b"\n" != entry.bytes:
                        raise GuardRefusal("noncanonical chat transcript line")
                    self.entries[entry.identity] = entry
                    self.lines[entry.identity] = len(self.entries) + 1
        except OSError:
            storage_failed = True
        except (GuardRefusal, Interruption):
            raise
        except Exception:
            pass
        else:
            return
        if storage_failed:
            raise StorageInterruption("chat transcript could not be read")
        raise GuardRefusal("invalid chat transcript")

    def usable(self) -> None:
        if self._failed:
            raise StorageInterruption("chat transcript storage failed; repeat the command")

    def check(self, entry: Entry) -> None:
        self.usable()
        validate(entry.response)
        protect(asdict(entry), self.bearer)
        if entry.run != self.identity:
            raise GuardRefusal("chat transcript entry run identity differs")
        if entry.request_digest != digest(entry.request) or entry.identity in self.entries:
            raise GuardRefusal("invalid or duplicate chat transcript identity or digest")

    def append(self, identity: Identity, text: str, response: Response) -> Entry:
        self.usable()
        if self.read_only:
            raise StorageInterruption("chat transcript is read-only")
        entry = Entry(self.identity, identity, text, digest(text), response)
        self.check(entry)
        before = self.read_bytes()
        lines = entry.bytes
        if not before:
            lines = kernel.encoded(asdict(self.identity)).encode() + lines
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab", buffering=0) as handle:
                if handle.write(lines) != len(lines):
                    raise OSError("short write")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException as error:
            self._failed = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        else:
            self.entries[identity] = entry
            self.lines[identity] = len(self.entries) + 1
            return entry
        raise StorageInterruption("chat transcript append failed; repeat the command")

    @property
    def spent(self) -> Decimal:
        self.usable()
        return sum((cost(e.response) for e in self.entries.values()), Decimal())

    def read_bytes(self) -> bytes:
        """Read without changing any bytes."""
        self.usable()
        try:
            raw = self.path.read_bytes() if self.path.exists() else b""
        except OSError:
            pass
        else:
            return complete(raw)
        raise StorageInterruption("chat transcript could not be read")

    def digest(self) -> str:
        return hashlib.sha256(self.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Choice:
    points: int
    line: int
    entry_digest: str
    attempts: tuple[str, ...]


def status(transcript: Transcript, cap: Decimal, *, recall: bool = False) -> Cause | None:
    """First cause in file order; recall excludes the decision-reply rule."""
    from zoon.interaction.budget import chat_max_estimate

    transcript.usable()
    version: str | None = None
    spent = Decimal()
    addition = chat_max_estimate()
    if spent + addition > cap:
        return Cause.CAP
    for key, entry in transcript.entries.items():
        reported = entry.response.version
        if version is None:
            version = reported
        if not version.startswith(MODEL) or reported != version:
            return Cause.VERSION
        if (
            not recall
            and key.kind == "decision"
            and key.attempt == REPLY_ATTEMPTS
            and points(entry.response.text, GAMES[key.game].endowments[key.seat]) is None
        ):
            return Cause.REPLIES
        spent += cost(entry.response)
        if spent + addition > cap:
            return Cause.CAP
    return None


class Session:
    """A client and spending reservations over an append-only transcript."""

    def __init__(self, transcript: Transcript, client: Client, cap: Decimal | None = None) -> None:
        from zoon.interaction.budget import chat_cap, chat_max_estimate

        if cap is None:
            cap = chat_cap()
        if not cap.is_finite() or cap < 0:
            raise ValueError("chat cap must be finite and non-negative")
        self.transcript, self.client, self.cap = transcript, client, cap
        self.reserved = Decimal()
        self.pending: set[Identity] = set()
        self._lock = Condition()
        self._estimate = chat_max_estimate()

    def check(self, *, recall: bool = False) -> None:
        cause = status(self.transcript, self.cap, recall=recall)
        if cause is not None:
            raise Unusable(cause)

    def ask(self, identity: Identity, text: str) -> Entry:
        with self._lock:
            while True:
                self.check(recall=identity.kind == "recall")
                recorded = self.transcript.entries.get(identity)
                if recorded is not None:
                    if recorded.request_digest != digest(text):
                        raise GuardRefusal("chat decision request changed on resume")
                    return recorded
                if identity in self.pending:
                    raise ConsistencyInterruption("chat decision already in flight")
                if self.transcript.read_only:
                    raise StorageInterruption("chat transcript is read-only")
                if self.transcript.spent + self.reserved + self._estimate <= self.cap:
                    break
                if not self.pending:
                    raise ConsistencyInterruption("chat admission and status disagree")
                self._lock.wait()
            self.reserved += self._estimate
            self.pending.add(identity)
        try:
            try:
                response = self.client.ask(text)
            except (Unusable, Interruption, GuardRefusal):
                raise
            except Exception:
                pass
            else:
                with self._lock:
                    entry = self.transcript.append(identity, text, response)
                    self.check(recall=identity.kind == "recall")
                    return entry
            raise ConsistencyInterruption("chat client failed; internal fault")
        finally:
            with self._lock:
                self.reserved -= self._estimate
                self.pending.remove(identity)
                self._lock.notify_all()

    def decision(self, view: kernel.View, group: str) -> Choice:
        text = request(view)
        attempts: list[str] = []
        for attempt in range(1, REPLY_ATTEMPTS + 1):
            key = Identity(view.game.cell, group, len(view.history), view.seat, attempt)
            entry = self.ask(key, text)
            attempts.append(digest(entry.bytes.decode()))
            value = points(entry.response.text, view.game.endowments[view.seat])
            if value is not None:
                return Choice(
                    value, self.transcript.lines[key], digest(entry.bytes.decode()), tuple(attempts)
                )
        raise Unusable(Cause.REPLIES)


def _play(
    game: Game,
    group: str,
    identity: RunIdentity,
    choose: Callable[[kernel.View, str], Choice],
) -> kernel.Run:
    """The kernel's probability-only play cannot accept points or transcript links.

    Keep only that small adapter here; reuse its views, balance check, encoding
    and run container unchanged. All seats see the same completed-round prefix.
    """
    Identity(game.cell, group, 0, 0, 1)
    log = [kernel.encoded({"kind": "start", "cell": game.cell, "group": group, **asdict(identity)})]
    history: tuple[kernel.Observation, ...] = ()
    rounds: list[tuple[int, ...]] = []
    for r in range(ROUNDS):
        choices = [choose(kernel.View(game, s, history), group) for s in range(game.size)]
        row = tuple(choice.points for choice in choices)
        payoffs = kernel.balance(game, row)
        pool = sum((p * c for p, c in zip(game.productivities, row, strict=True)), Fraction())
        for s, choice in enumerate(choices):
            log.append(kernel.encoded({"kind": "intent", "round": r, "seat": s, **asdict(choice)}))
        rounds.append(row)
        for s in range(game.size):
            log.append(
                kernel.encoded(
                    {
                        "kind": "transfer",
                        "round": r,
                        "seat": s,
                        "kept": str(Fraction(game.endowments[s] - row[s])),
                        "pool": str(pool),
                        "share": str(pool / game.size),
                        "payoff": str(payoffs[s]),
                    }
                )
            )
        history += (kernel.Observation(tuple(enumerate(row)), payoffs),)
    result = kernel.Run(tuple(rounds), tuple(log))
    protect(result.bytes.decode())
    return result


def play(game: Game, session: Session, group: str) -> kernel.Run:
    session.check()
    return _play(game, group, session.transcript.identity, session.decision)


@dataclass(frozen=True)
class _VerifiedTranscript:
    transcript: Transcript
    digest: str


def replay(log: bytes, verified: _VerifiedTranscript, cap: Decimal | None = None) -> kernel.Run:
    """No client, no sampling: rebuild requests and ledger and require identical bytes."""
    from zoon.interaction.budget import chat_cap

    if not isinstance(cast(object, verified), _VerifiedTranscript):
        raise GuardRefusal("chat replay requires a verified transcript")
    transcript = verified.transcript
    cause = status(transcript, chat_cap() if cap is None else cap)
    if cause is not None:
        raise Unusable(cause)
    try:
        header = json.loads(log.splitlines()[0])
        if any(header.get(k) != v for k, v in asdict(transcript.identity).items()):
            raise GuardRefusal("chat log run identity differs")
        game = GAMES[header["cell"]]

        def recorded(view: kernel.View, group: str) -> Choice:
            text = request(view)
            attempts: list[str] = []
            for attempt in range(1, REPLY_ATTEMPTS + 1):
                key = Identity(game.cell, group, len(view.history), view.seat, attempt)
                entry = transcript.entries[key]
                if entry.request_digest != digest(text):
                    raise GuardRefusal("chat transcript request differs from replay")
                attempts.append(digest(entry.bytes.decode()))
                value = points(entry.response.text, game.endowments[view.seat])
                if value is not None:
                    return Choice(
                        value, transcript.lines[key], digest(entry.bytes.decode()), tuple(attempts)
                    )
            raise Unusable(Cause.REPLIES)

        result = _play(game, header["group"], transcript.identity, recorded)
        if result.bytes != log:
            raise Unusable(Cause.REPLAY)
        return result
    except Exception:
        pass
    raise Unusable(Cause.REPLAY)


@dataclass(frozen=True)
class Recall:
    game: str
    response: Response | None
    flagged: bool
    line: int | None
    status: Literal["judged", "not judged: version", "not asked"]
    entry_digest: str | None


@dataclass(frozen=True)
class RecallResult(Sequence[Recall]):
    replies: tuple[Recall, ...]
    stopped_by: Cause | None

    def __len__(self) -> int:
        return len(self.replies)

    @overload
    def __getitem__(self, index: int) -> Recall: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Recall, ...]: ...

    def __getitem__(self, index: int | slice) -> Recall | tuple[Recall, ...]:
        return self.replies[index]


def read_recall(transcript: Transcript) -> tuple[Recall, ...]:
    """Derive all recall labels and string-match flags without a client."""
    transcript.usable()
    replies: list[Recall] = []
    version = next((e.response.version for e in transcript.entries.values()), None)
    for cell in sorted(GAMES):
        key = Identity(cell, "recall", 0, 0, 1, "recall")
        entry = transcript.entries.get(key)
        if entry and entry.request_digest != digest(prompted.recall(GAMES[cell])):
            raise GuardRefusal("chat recall request changed on resume")
        response = entry.response if entry else None
        invalid = bool(
            response and (not response.version.startswith(MODEL) or response.version != version)
        )
        label = "not asked" if response is None else "not judged: version" if invalid else "judged"
        replies.append(
            Recall(
                cell,
                response,
                bool(response and not invalid and recognition.flagged(response.text, RECALL_KEYS)),
                transcript.lines.get(key),
                label,
                digest(entry.bytes.decode()) if entry else None,
            )
        )
    return tuple(replies)


def recall(session: Session) -> RecallResult:
    stopped_by = status(session.transcript, session.cap, recall=True)
    for cell, game in sorted(GAMES.items()):
        if stopped_by is not None:
            break
        key = Identity(cell, "recall", 0, 0, 1, "recall")
        if key not in session.transcript.entries:
            try:
                session.ask(key, prompted.recall(game))
            except Unusable as error:
                stopped_by = error.cause
    return RecallResult(read_recall(session.transcript), stopped_by)


def replay_transcript(transcript: Transcript, expected_digest: str) -> _VerifiedTranscript:
    """Verify the whole-file SHA-256 and every retained attempt and recall request.

    The run machinery records Transcript.digest() after each completed calling
    step. A paid, version-invalid reply remains an attempt but is never judged.
    Partial games need no invented completed game log to check their attempts.
    """
    raw = transcript.read_bytes()
    if not raw:
        raise GuardRefusal("missing chat transcript")
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise Unusable(Cause.REPLAY)
    disk_lines = raw.split(b"\n")[:-1]
    if len(disk_lines) != len(transcript.entries) + bool(raw):
        raise Unusable(Cause.REPLAY)
    directory, identity, bearer = transcript.path.parent, transcript.identity, transcript.bearer
    histories: dict[tuple[str, str], tuple[kernel.Observation, ...]] = {}
    rows: dict[tuple[str, str], dict[int, int]] = {}
    attempts: dict[tuple[str, str, int, int], int] = {}
    try:
        transcript = Transcript(directory, identity, bearer=bearer, _raw=raw)
        for key, entry in transcript.entries.items():
            if key.kind == "recall":
                text = prompted.recall(GAMES[key.game])
            else:
                decision = (key.game, key.group, key.round, key.seat)
                if key.attempt != attempts.get(decision, 0) + 1:
                    raise GuardRefusal("chat attempts are not consecutive")
                attempts[decision] = key.attempt
                group = (key.game, key.group)
                history = histories.get(group, ())
                game = GAMES[key.game]
                if key.round != len(history):
                    raise GuardRefusal("chat attempt history differs")
                text = request(kernel.View(game, key.seat, history))
                row = rows.setdefault(group, {})
                if key.seat in row:
                    raise GuardRefusal("attempt after accepted decision")
                value = points(entry.response.text, game.endowments[key.seat])
                if value is not None:
                    row[key.seat] = value
                if len(row) == game.size:
                    contributions = tuple(row[s] for s in range(game.size))
                    histories[group] = (
                        *history,
                        kernel.Observation(
                            tuple(enumerate(contributions)), game.payoffs(contributions)
                        ),
                    )
                    rows[group] = {}
            if entry.request != text or entry.request_digest != digest(text):
                raise GuardRefusal("chat transcript request differs from replay")
    except StorageInterruption:
        raise
    except Exception:
        pass
    else:
        return _VerifiedTranscript(transcript, expected_digest)
    raise Unusable(Cause.REPLAY)


def fixtures() -> dict[str, str]:
    result: dict[str, str] = {}
    rng = np.random.default_rng(prompted.FIXTURE_SEED)
    for cell, game in sorted(GAMES.items()):
        group = Group(
            cell,
            "made-up",
            tuple(tuple(int(rng.integers(e + 1)) for e in game.endowments) for _ in range(ROUNDS)),
        )
        seats = {measure.seat_type(game, s): s for s in reversed(range(game.size))}
        for kind, seat in sorted(seats.items()):
            for r in (1, ROUNDS):
                result[f"{cell}-type-{kind}-round-{r}.txt"] = prompted.FIXTURE_LABEL + request(
                    prompted.training_view(group, seat, r)
                )
    return result


def write_fixtures() -> None:
    directory = prompted.DIRECTORY / "chat-requests"
    directory.mkdir(exist_ok=True)
    digests: dict[str, str] = {}
    for name, text in fixtures().items():
        (directory / name).write_bytes(text.encode())
        digests[name] = digest(text)
    (directory / "sha256.json").write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")


def smoke(client: Client) -> list[str]:
    lines: list[str] = []
    for cell in prompted.SMOKE_GAMES:
        for r in (1, ROUNDS):
            name = f"{cell}-type-0-round-{r}.txt"
            text = (prompted.DIRECTORY / "chat-requests" / name).read_text(encoding="utf-8")
            text = text.removeprefix(prompted.FIXTURE_LABEL)
            for repeat in range(1, SMOKE_REPEATS + 1):
                response = client.ask(text)
                validate(response)
                value = points(response.text, GAMES[cell].endowments[0])
                lines.append(
                    f"[smoke {name} {repeat}] version = {response.version} | "
                    f"text = {json.dumps(response.text, ensure_ascii=False)} | "
                    f"usable = {value is not None} | points = {value} | "
                    f"input tokens = {response.input_tokens} | "
                    f"output tokens = {response.output_tokens} | "
                    f"thinking tokens = {response.thinking_tokens} | "
                    f"USD higher = {cost(response):.8f} | "
                    f"USD lower = {cost(response, chat_prices.LOW):.8f}"
                )
    return lines


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fixtures", "smoke"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stub", action="store_const", dest="mode", const="stub")
    mode.add_argument("--live", action="store_const", dest="mode", const="live")
    parser.set_defaults(mode=SMOKE_DEFAULT)
    args = parser.parse_args(arguments)
    if args.command == "fixtures":
        write_fixtures()
    else:
        if args.mode == "live":
            client = VertexClient(post)
            client.preflight()
        else:
            client = StubClient()
        print("\n".join(smoke(client)))


if __name__ == "__main__":
    main()
