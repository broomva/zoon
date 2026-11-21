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
from typing import Any, Literal, Protocol, cast

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
    """Only trimmed ASCII digits, including leading zeroes; truncation has no exemption."""
    text = text.strip()
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

    This deliberately exceeds the pilot's character ratio and the two measured
    chat requests. It assumes byte-fallback tokenisation, not a measured tokenizer
    guarantee. Measured chat inputs were 462 and 1,944 tokens; the corresponding
    decision estimates were 1,172 and 2,588. No such sample proves a universal bound.
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
    """A resumable failure; no reply attempt is consumed."""


class GuardRefusal(Exception):
    pass


class TransientError(Exception):
    pass


class ModelMissing(TransientError):
    pass


def retry(call: Callable[[], Response], pause: Callable[[float], None] = time.sleep) -> Response:
    missing = 0
    for attempt in range(TRANSPORT_ATTEMPTS):
        try:
            return call()
        except (
            TransientError,
            TimeoutError,
            ConnectionError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            OSError,
        ) as error:
            missing += isinstance(error, ModelMissing)
            if attempt + 1 == TRANSPORT_ATTEMPTS:
                if missing == TRANSPORT_ATTEMPTS:
                    raise Unusable(Cause.WITHDRAWN) from None
                raise Interruption("chat transport failure after bounded retries") from None
            pause(TRANSPORT_PAUSES[attempt])
    raise Interruption("chat transport attempts must be positive")


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
        raise Interruption("invalid chat response or token usage")
    protect(asdict(response))


def post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
    """The live transport; callers must explicitly inject it."""
    try:
        outgoing = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(outgoing, timeout=30) as reply:
            value = json.loads(reply.read())
            if not isinstance(value, dict):
                raise TransientError
            return reply.status, cast(dict[str, Any], value)
    except urllib.error.HTTPError as error:
        return error.code, {}
    except (ValueError, TypeError):
        raise TransientError from None


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
        raise Interruption("chat credentials unavailable") from None


class Client(Protocol):
    def ask(self, text: str) -> Response: ...


class VertexClient:
    def __init__(
        self,
        transport: HTTP,
        *,
        token: Callable[[], str] = access_token,
        pause: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport, self.token, self.pause = transport, token, pause
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
                    raise Interruption("chat credentials unavailable") from None
            return self._bearer

    def ask(self, text: str) -> Response:
        project = os.environ.get("ZOON_VERTEX_PROJECT", "")
        if not project:
            raise Interruption("ZOON_VERTEX_PROJECT is required")
        protect(text)
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
                    raise Interruption("chat request refused")
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
                response = Response(
                    reply,
                    data.get("modelVersion", ""),
                    usage.get("promptTokenCount", 0),
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
                http.client.IncompleteRead,
            ):
                raise
            except Exception:
                raise Interruption("chat response processing failed") from None

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
    identity: Identity
    request: str
    request_digest: str
    response: Response

    @property
    def bytes(self) -> bytes:
        return kernel.encoded(asdict(self)).encode()


class Transcript:
    def __init__(self, directory: Path, identity: RunIdentity) -> None:
        self.path = directory / "chat.jsonl"
        self.identity = identity
        self.entries: dict[Identity, Entry] = {}
        self.lines: dict[Identity, int] = {}
        protect(asdict(identity))
        try:
            if self.path.exists() and self.path.stat().st_size:
                raw = self.path.read_bytes()
                if not raw.endswith(b"\n"):
                    raise GuardRefusal("incomplete chat transcript line")
                protect(raw.decode())
                lines = raw.splitlines()
                if json.loads(lines[0]) != asdict(identity):
                    raise GuardRefusal("chat transcript run identity differs")
                for line in lines[1:]:
                    row = json.loads(line)
                    entry = Entry(
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
        except (GuardRefusal, Interruption):
            raise
        except Exception:
            raise GuardRefusal("invalid chat transcript") from None

    def check(self, entry: Entry) -> None:
        validate(entry.response)
        protect(asdict(entry))
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
        entry = Entry(identity, text, digest(text), response)
        self.check(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("ab") as handle:
            if new:
                handle.write(kernel.encoded(asdict(self.identity)).encode())
            handle.write(entry.bytes)
            handle.flush()
            os.fsync(handle.fileno())
        self.entries[identity] = entry
        self.lines[identity] = len(self.entries) + 1
        return entry

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


class Session:
    def __init__(self, transcript: Transcript, client: Client, cap: Decimal | None = None) -> None:
        if cap is None:
            from zoon.interaction.budget import chat_cap

            cap = chat_cap()
        if not cap.is_finite() or cap < 0:
            raise ValueError("chat cap must be finite and non-negative")
        self.transcript, self.client, self.cap = transcript, client, cap
        self.reserved = Decimal()
        self.pending: set[Identity] = set()
        self._lock = Lock()
        self.terminal: Cause | None = None

    def check(self, *, recall: bool = False) -> None:
        if self.terminal is not None and (
            not recall or self.terminal in (Cause.WITHDRAWN, Cause.VERSION, Cause.CAP)
        ):
            raise Unusable(self.terminal)
        versions = [e.response.version for e in self.transcript.entries.values()]
        if versions and (
            not versions[0].startswith(MODEL) or any(v != versions[0] for v in versions)
        ):
            raise Unusable(Cause.VERSION)
        if self.transcript.spent >= self.cap:
            raise Unusable(Cause.CAP)
        if not recall and any(
            key.kind == "decision"
            and key.attempt == REPLY_ATTEMPTS
            and points(entry.response.text, GAMES[key.game].endowments[key.seat]) is None
            for key, entry in self.transcript.entries.items()
        ):
            raise Unusable(Cause.REPLIES)

    def ask(self, identity: Identity, text: str) -> Entry:
        with self._lock:
            self.check(recall=identity.kind == "recall")
            recorded = self.transcript.entries.get(identity)
            if recorded is not None:
                if recorded.request_digest != digest(text):
                    raise GuardRefusal("chat decision request changed on resume")
                return recorded
            if identity in self.pending:
                raise Interruption("chat decision already in flight")
            addition = estimate(text)
            if self.transcript.spent + self.reserved + addition > self.cap:
                raise Unusable(Cause.CAP)
            self.reserved += addition
            self.pending.add(identity)
        try:
            try:
                response = self.client.ask(text)
            except Unusable as error:
                self.terminal = error.cause
                raise
            except (Interruption, GuardRefusal):
                raise
            except Exception:
                raise Interruption("chat client failed; decision is resumable") from None
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
        for attempt in range(1, REPLY_ATTEMPTS + 1):
            key = Identity(view.game.cell, group, len(view.history), view.seat, attempt)
            entry = self.ask(key, text)
            value = points(entry.response.text, view.game.endowments[view.seat])
            if value is not None:
                return Choice(value, self.transcript.lines[key], digest(entry.bytes.decode()))
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
            for attempt in range(1, REPLY_ATTEMPTS + 1):
                key = Identity(game.cell, group, len(view.history), view.seat, attempt)
                entry = transcript.entries[key]
                if entry.request_digest != digest(text):
                    raise GuardRefusal("chat transcript request differs from replay")
                value = points(entry.response.text, game.endowments[view.seat])
                if value is not None:
                    return Choice(value, transcript.lines[key], digest(entry.bytes.decode()))
            raise Unusable(Cause.REPLIES)

        result = _play(game, header["group"], transcript.identity, recorded)
        if result.bytes != log:
            raise Unusable(Cause.REPLAY)
        return result
    except Exception:
        raise Unusable(Cause.REPLAY) from None


@dataclass(frozen=True)
class Recall:
    game: str
    response: Response | None
    flagged: bool
    line: int | None


def recall(session: Session) -> tuple[Recall, ...]:
    replies: list[Recall] = []
    stopped = False
    for cell, game in sorted(GAMES.items()):
        key = Identity(cell, "recall", 0, 0, 1, "recall")
        if not stopped:
            try:
                session.ask(key, prompted.recall(game))
            except Unusable:
                stopped = True
        entry = session.transcript.entries.get(key)
        response = entry.response if entry else None
        replies.append(
            Recall(
                cell,
                response,
                bool(response and recognition.flagged(response.text, RECALL_KEYS)),
                session.transcript.lines.get(key),
            )
        )
    return tuple(replies)


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
    mode.add_argument("--stub", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args(arguments)
    if args.command == "fixtures":
        write_fixtures()
    else:
        client = VertexClient(post) if args.live else StubClient()
        print("\n".join(smoke(client)))


if __name__ == "__main__":
    main()
