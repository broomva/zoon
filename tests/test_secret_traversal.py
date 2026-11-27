"""Malformed neighboring fields cannot hide a secret split across readable parts."""

import json
from typing import Any

import pytest

from zoon.interaction import chat, service_answers
from zoon.interaction.service_answers import Answer


@pytest.mark.parametrize("secret", ["fixture-project", "dummy-bearer"])
@pytest.mark.parametrize(
    "shape", ["second candidate", "null parts", "integer part", "numeric text"]
)
def test_secret_traversal_skips_unreadable_neighbors(secret: str, shape: str) -> None:
    value = service_answers.reply()
    first, second = secret[: len(secret) // 2], secret[len(secret) // 2 :]
    if shape in ("second candidate", "null parts"):
        value["candidates"] = [
            {"content": {"parts": [{"text": first}]}},
            None if shape == "second candidate" else {"content": {"parts": None}},
            {"content": {"parts": [{"text": second}]}},
        ]
    else:
        value["candidates"][0]["content"]["parts"] = [
            {"text": first},
            7 if shape == "integer part" else {"text": 5},
            {"text": second},
        ]

    class Malformed(Answer):
        def body(self) -> bytes:
            return json.dumps(value).encode()

    with (
        service_answers.network("generating request", [Malformed(200)]) as requests,
        pytest.raises(chat.GuardRefusal, match="no record was written"),
    ):
        client = service_answers.client()
        client.preflight()
        client.ask("hello")
    assert len(requests) == 2


@pytest.mark.parametrize("neighbor", [None, {"content": {"parts": None}}])
def test_unreadable_later_candidate_does_not_fault_the_text_scan(neighbor: Any) -> None:
    value = service_answers.reply()
    value["candidates"].append(neighbor)

    class Malformed(Answer):
        def body(self) -> bytes:
            return json.dumps(value).encode()

    with service_answers.network("generating request", [Malformed(200)]) as requests:
        client = service_answers.client()
        client.preflight()
        assert client.ask("hello").text == "3"
    assert len(requests) == 2
