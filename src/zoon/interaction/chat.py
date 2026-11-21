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
from threading import Lock
from typing import Any, Literal, NoReturn, Protocol, cast, overload

import numpy as np

from zoon.interaction import chat_prices, kernel, measure, prompted
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.pilot import recognition

MODEL = chat_prices.HIGH.model
LOCATION = "global"
TEMPERATURE = 1.0
THINKING_LEVEL = "low"
MAX_OUTPUT_TOKENS = 256
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
    """A transcript write was rolled back; storage must work before repeating."""


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


def validate(response: Response) -> None:
    if (
        not isinstance(cast(object, response.text), str)
        or not isinstance(cast(object, response.version), str)
        or any(
            type(value) is not int or value < 0
            for value in (response.input_tokens, response.output_tokens, response.thinking_tokens)
        )
    ):
        raise ConsistencyInterruption("invalid chat response or token usage")
    protect(asdict(response))


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

    Preflight queries model metadata without generating text. Missing metadata is
    a configuration fault; only four missing generating responses mean withdrawn.
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
        if not os.environ.get("ZOON_VERTEX_PROJECT", ""):
            raise ConfigurationInterruption("ZOON_VERTEX_PROJECT is required")

        def probe() -> Response:
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
            if status != 200:
                raise ConfigurationInterruption("chat metadata request refused")
            return Response("", MODEL, 0, 0)

        retry(probe, self.pause)

    def ask(self, text: str) -> Response:
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
                required = ["promptTokenCount"]
                if any(p.get("text", "") for p in parts):
                    required += ["candidatesTokenCount", "thoughtsTokenCount"]
                if any(type(usage.get(name)) is not int for name in required):
                    raise TransientError
                response = Response(
                    reply,
                    version,
                    usage["promptTokenCount"],
                    usage.get("candidatesTokenCount", 0),
                    usage.get("thoughtsTokenCount", 0),
                )
                validate(response)
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
    def __init__(self, directory: Path, identity: RunIdentity, *, bearer: str = "") -> None:
        self.path = directory / "chat.jsonl"
        self.identity = identity
        self.bearer = bearer
        self.tail_removals = 0
        self.entries: dict[Identity, Entry] = {}
        self.lines: dict[Identity, int] = {}
        protect(asdict(identity))
        try:
            if self.path.exists() and self.path.stat().st_size:
                raw = self.path.read_bytes()
                complete = raw[: raw.rfind(b"\n") + 1]
                protect(complete.decode(), self.bearer)
                lines = complete.split(b"\n")[:-1]
                if lines and lines[0] + b"\n" != kernel.encoded(asdict(identity)).encode():
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
                if complete != raw:
                    with self.path.open("r+b") as handle:
                        handle.truncate(len(complete))
                        handle.flush()
                        os.fsync(handle.fileno())
                    self.tail_removals = 1
        except (GuardRefusal, Interruption):
            raise
        except Exception:
            pass
        else:
            return
        raise GuardRefusal("invalid chat transcript")

    def check(self, entry: Entry) -> None:
        validate(entry.response)
        protect(asdict(entry), self.bearer)
        if entry.run != self.identity:
            raise GuardRefusal("chat transcript entry run identity differs")
        if entry.request_digest != digest(entry.request) or entry.identity in self.entries:
            raise GuardRefusal("invalid or duplicate chat transcript identity or digest")
        key = entry.identity
        if key.attempt > 1:
            previous = Identity(key.game, key.group, key.round, key.seat, key.attempt - 1)
            prior = self.entries.get(previous)
            if (
                prior is None
                or prior.request != entry.request
                or points(prior.response.text, GAMES[key.game].endowments[key.seat]) is not None
            ):
                raise GuardRefusal("chat reply attempts are not consecutive unusable replies")

    def append(self, identity: Identity, text: str, response: Response) -> Entry:
        entry = Entry(self.identity, identity, text, digest(text), response)
        self.check(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        before = self.path.stat().st_size if self.path.exists() else 0
        if before:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise GuardRefusal("incomplete chat transcript line; reopen before appending")
        lines = [entry.bytes]
        if not before:
            lines.insert(0, kernel.encoded(asdict(self.identity)).encode())
        try:
            # Unbuffered writes cannot flush a failed buffered tail after rollback.
            with self.path.open("ab", buffering=0) as handle:
                for line in lines:
                    if handle.write(line) != len(line):
                        raise OSError("short write")
                os.fsync(handle.fileno())
        except BaseException:
            pass
        else:
            self.entries[identity] = entry
            self.lines[identity] = len(self.entries) + 1
            return entry
        try:
            with self.path.open("r+b", buffering=0) as handle:
                handle.truncate(before)
                os.fsync(handle.fileno())
        except BaseException:
            pass
        else:
            raise StorageInterruption("chat transcript write interrupted; no record was written")
        raise ConsistencyInterruption("chat transcript rollback failed; storage repair required")

    @property
    def spent(self) -> Decimal:
        return sum((cost(e.response) for e in self.entries.values()), Decimal())

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes() if self.path.exists() else b"").hexdigest()


@dataclass(frozen=True)
class Choice:
    points: int
    line: int
    entry_digest: str
    attempts: tuple[str, ...]


class Session:
    def __init__(
        self,
        transcript: Transcript,
        client: Client,
        cap: Decimal | None = None,
        *,
        terminal: Cause | None = None,
    ) -> None:
        if cap is None:
            from zoon.interaction.budget import chat_cap

            cap = chat_cap()
        if not cap.is_finite() or cap < 0:
            raise ValueError("chat cap must be finite and non-negative")
        self.transcript, self.client, self.cap = transcript, client, cap
        self.reserved = Decimal()
        self.pending: set[Identity] = set()
        self._lock = Lock()
        self.terminal = terminal

    @property
    def tail_removals(self) -> int:
        return self.transcript.tail_removals

    def stop(self, cause: Cause) -> NoReturn:
        self.terminal = cause
        raise Unusable(cause)

    def check(self, *, recall: bool = False) -> None:
        if self.terminal is not None and (
            not recall or self.terminal in (Cause.WITHDRAWN, Cause.VERSION, Cause.CAP)
        ):
            self.stop(self.terminal)
        self.check_version()
        if self.transcript.spent > self.cap:
            self.stop(Cause.CAP)
        if not recall and any(
            key.kind == "decision"
            and key.attempt == REPLY_ATTEMPTS
            and points(entry.response.text, GAMES[key.game].endowments[key.seat]) is None
            for key, entry in self.transcript.entries.items()
        ):
            self.stop(Cause.REPLIES)

    def check_version(self) -> None:
        versions = [e.response.version for e in self.transcript.entries.values()]
        if versions and (
            not versions[0].startswith(MODEL) or any(v != versions[0] for v in versions)
        ):
            self.stop(Cause.VERSION)

    def ask(self, identity: Identity, text: str) -> Entry:
        with self._lock:
            recorded = self.transcript.entries.get(identity)
            if recorded is not None:
                if recorded.request_digest != digest(text):
                    raise GuardRefusal("chat decision request changed on resume")
                self.check_version()
                return recorded
            self.check(recall=identity.kind == "recall")
            if identity in self.pending:
                raise Interruption("chat decision already in flight")
            addition = estimate(text)
            if self.transcript.spent + self.reserved + addition > self.cap:
                self.stop(Cause.CAP)
            self.reserved += addition
            self.pending.add(identity)
        try:
            response: Response | None = None
            try:
                response = self.client.ask(text)
            except Unusable as error:
                self.terminal = error.cause
                raise
            except (Interruption, GuardRefusal):
                raise
            except Exception:
                failed = True
            else:
                failed = False
            if failed or response is None:
                raise ConsistencyInterruption("chat client failed; internal fault")
            with self._lock:
                entry = self.transcript.append(identity, text, response)
                self.check(recall=identity.kind == "recall")
                return entry
        finally:
            with self._lock:
                self.reserved -= addition
                self.pending.remove(identity)

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
        self.stop(Cause.REPLIES)


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
    return _play(game, group, session.transcript.identity, session.decision)


def replay(log: bytes, transcript: Transcript) -> kernel.Run:
    """No client, no sampling: rebuild requests and ledger and require identical bytes."""
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
    terminal: Cause | None

    def __len__(self) -> int:
        return len(self.replies)

    @overload
    def __getitem__(self, index: int) -> Recall: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Recall, ...]: ...

    def __getitem__(self, index: int | slice) -> Recall | tuple[Recall, ...]:
        return self.replies[index]


def recall(session: Session) -> RecallResult:
    replies: list[Recall] = []
    stopped = False
    versions = [e.response.version for e in session.transcript.entries.values()]
    version = versions[0] if versions else None
    for cell, game in sorted(GAMES.items()):
        key = Identity(cell, "recall", 0, 0, 1, "recall")
        if not stopped:
            try:
                session.ask(key, prompted.recall(game))
            except Unusable:
                stopped = True
        entry = session.transcript.entries.get(key)
        response = entry.response if entry else None
        if version is None and response:
            version = response.version
        invalid = bool(
            response and (not response.version.startswith(MODEL) or response.version != version)
        )
        status = "not asked" if response is None else "not judged: version" if invalid else "judged"
        replies.append(
            Recall(
                cell,
                response,
                bool(response and not invalid and recognition.flagged(response.text, RECALL_KEYS)),
                session.transcript.lines.get(key),
                status,
                digest(entry.bytes.decode()) if entry else None,
            )
        )
    return RecallResult(tuple(replies), session.terminal)


def replay_transcript(transcript: Transcript, expected_digest: str) -> None:
    """Verify the whole-file SHA-256 and every retained attempt and recall request.

    The run machinery records Transcript.digest() after each completed calling
    step. A paid, version-invalid reply remains an attempt but is never judged.
    Partial games need no invented completed game log to check their attempts.
    """
    raw = transcript.path.read_bytes() if transcript.path.exists() else b""
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise Unusable(Cause.REPLAY)
    disk_lines = raw.split(b"\n")[:-1]
    if len(disk_lines) != len(transcript.entries) + bool(raw):
        raise Unusable(Cause.REPLAY)
    histories: dict[tuple[str, str], tuple[kernel.Observation, ...]] = {}
    rows: dict[tuple[str, str], dict[int, int]] = {}
    seen: set[Identity] = set()
    try:
        for key, entry in transcript.entries.items():
            if key in seen or entry.run != transcript.identity:
                raise GuardRefusal("unowned transcript entry")
            seen.add(key)
            if key.kind == "recall":
                text = prompted.recall(GAMES[key.game])
            else:
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
            # Also bind in-memory entries to the canonical lines on disk.
            if disk_lines[transcript.lines[key] - 1] + b"\n" != entry.bytes:
                raise GuardRefusal("chat transcript entry differs from disk")
    except Exception:
        pass
    else:
        return
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
        client = VertexClient(post) if args.mode == "live" else StubClient()
        print("\n".join(smoke(client)))


if __name__ == "__main__":
    main()
