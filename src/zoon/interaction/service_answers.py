"""Regenerate the chat service answer table without a network connection."""

from __future__ import annotations

import argparse
import http.client
import io
import json
import os
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from zoon.interaction import chat
from zoon.interaction.steps import PreflightClient

TABLE = Path(__file__).resolve().parents[3] / "interaction" / "service-answers.md"
STAGES = ("metadata check", "generating request")


def reply() -> dict[str, Any]:
    return {
        "modelVersion": chat.MODEL + "-stub",
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1},
        "candidates": [{"content": {"parts": [{"text": "3"}]}}],
    }


@dataclass(frozen=True)
class Answer:
    status: int | None
    kind: str = "JSON object"

    def body(self) -> bytes:
        if self.kind == "not a JSON object":
            return b"[]"
        if self.kind == "empty body":
            return b""
        value = reply()
        if self.kind == "no model version":
            del value["modelVersion"]
        elif self.kind == "bad token counts":
            value["usageMetadata"]["promptTokenCount"] = -1
        elif self.kind == "secret in text":
            value["candidates"][0]["content"]["parts"][0]["text"] = "fixture-project"
        if "non-finite" in self.kind:
            value["unused"] = float("nan")
            if "secret" in self.kind:
                value["candidates"][0]["content"]["parts"][0]["text"] = "fixture-project"
        return json.dumps(value).encode()


@dataclass(frozen=True)
class Outcome:
    category: str
    requests: int
    acceptable: bool


@contextmanager
def network(stage: str, answers: Sequence[Answer]) -> Generator[list[str]]:
    """Fake only socket establishment; urllib and HTTP response parsing still run.

    Answers repeat after the supplied sequence ends. Other stages and reachable
    redirect targets return a valid reply. No real socket is ever established.
    """
    requests: list[str] = []
    selected = 0

    class Socket:
        def __init__(self) -> None:
            self.sent = b""

        def sendall(self, raw: bytes) -> None:
            self.sent += raw

        def makefile(self, mode: str) -> io.BytesIO:
            nonlocal selected
            assert mode == "rb"
            line = self.sent.split(b"\r\n", 1)[0].decode()
            requests.append(line)
            target = line.split()[1]
            current = "generating request" if ":generateContent" in target else "metadata check"
            if target == "/unreachable":
                raise ConnectionError("connection unavailable")
            answer = Answer(200)
            if current == stage and target != "/reachable":
                answer = answers[min(selected, len(answers) - 1)]
                selected += 1
            if target == "/loop":
                answer = Answer(302, "redirect to loop")
            if answer.status is None:
                raise ConnectionError("connection unavailable")
            headers = b""
            if answer.kind.startswith("redirect to "):
                destination = answer.kind.removeprefix("redirect to ")
                headers = b"Location: https://service.invalid/" + destination.encode() + b"\r\n"
            body = answer.body()

            class BodyTimeout(io.BytesIO):
                def read(self, size: int = -1) -> bytes:
                    super().read(max(1, size // 2))
                    raise TimeoutError("body read timed out")

            stream = BodyTimeout if answer.kind == "body timeout" else io.BytesIO
            return stream(
                f"HTTP/1.1 {answer.status} Answer\r\nContent-Length: {len(body)}\r\n".encode()
                + headers
                + b"Connection: close\r\n\r\n"
                + body
            )

        def close(self) -> None:
            pass

    def connect(connection: http.client.HTTPSConnection) -> None:
        connection.sock = Socket()  # type: ignore[assignment]

    with (
        patch.dict(os.environ, {"ZOON_VERTEX_PROJECT": "fixture-project"}),
        patch.object(http.client.HTTPSConnection, "connect", connect),
    ):
        yield requests


def client() -> chat.VertexClient:
    return chat.VertexClient(
        chat.post, preflight_transport=chat.get, token=lambda: "dummy-bearer", pause=lambda _: None
    )


def observe(stage: str, answer: Answer) -> Outcome:
    answers = (
        [Answer(401), Answer(200 if answer.kind == "refreshed credential succeeds" else 401)]
        if answer.kind.startswith("refreshed credential")
        else [answer]
    )
    with network(stage, answers) as requests:
        try:
            PreflightClient(client()).ask("hello")
            category = "reply accepted"
        except chat.Unusable as error:
            assert error.cause == chat.Cause.WITHDRAWN
            category = "withdrawal counted"
        except chat.TransportInterruption:
            category = "transport interruption"
        except chat.ConfigurationInterruption:
            category = "configuration refusal"
        except chat.SecretRefusal:
            category = "secret refusal"
    return Outcome(
        category,
        len(requests),
        category in ("transport interruption", "configuration refusal", "secret refusal"),
    )


def cases() -> Iterator[tuple[str, list[Answer]]]:
    yield "Continue then invalid status line", [Answer(100)]
    yield "JSON object", [Answer(status) for status in range(101, 300)]
    yield "redirect without Location", [Answer(status) for status in range(300, 400)]
    yield "JSON object", [Answer(status) for status in range(400, 600)]
    for kind in ("not a JSON object", "empty body"):
        yield kind, [Answer(status, kind) for status in range(200, 300)]
    for kind in ("no model version", "bad token counts", "secret in text"):
        yield kind, [Answer(200, kind)]
    for target in ("reachable", "unreachable"):
        yield (
            "redirect to " + target,
            [Answer(status, "redirect to " + target) for status in (301, 302, 303, 307, 308)],
        )
    yield "failed connection", [Answer(None)]
    yield "redirect loop", [Answer(302, "redirect to loop")]
    for kind in ("body timeout", "non-finite number", "non-finite number with secret"):
        yield kind, [Answer(200, kind)]
    for kind in ("refreshed credential succeeds", "refreshed credential rejected"):
        yield kind, [Answer(401, kind)]


def render() -> bytes:
    lines = [
        "# Chat service answers",
        "",
        "Generated by `python -m zoon.interaction.service_answers`; "
        "no network calls or model calls.",
        "Requests include the metadata check, bounded retries and redirect targets.",
        "Credentials are dummy values; refresh rows give the answers after refresh.",
        "The JSON object has a valid reply envelope; response-check variants change that envelope.",
        "The flag column concerns the next attempt after this persistent answer, "
        "with no saved ending.",
        "A metadata reply accepted here permits the generating request; it is not a chat decision.",
        "",
        "| Stage | Answer | Outcome | Requests | Flag acceptable next attempt |",
        "|---|---|---|---:|---|",
    ]
    for stage in STAGES:
        for kind, answers in cases():
            rows: list[tuple[Answer, Answer, Outcome]] = []
            for answer in answers:
                result = observe(stage, answer)
                if (
                    rows
                    and rows[-1][2] == result
                    and answer.status is not None
                    and rows[-1][1].status == answer.status - 1
                ):
                    first, _, outcome = rows[-1]
                    rows[-1] = first, answer, outcome
                else:
                    rows.append((answer, answer, result))
            for first, last, result in rows:
                status = str(first.status) if first.status is not None else ""
                if first.status != last.status:
                    status += "-" + str(last.status)
                label = (status + " " + kind).strip()
                lines.append(
                    f"| {stage} | {label} | {result.category} | {result.requests} | "
                    + ("yes" if result.acceptable else "no")
                    + " |"
                )
    return ("\n".join(lines) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TABLE)
    args = parser.parse_args()
    args.output.write_bytes(render())


if __name__ == "__main__":
    main()
