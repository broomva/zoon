"""The registered recognition flag: adjacent runs of letters and digits."""

import re
from collections.abc import Sequence
from typing import cast

from zoon.pilot import outside, rewrites


def keys(dilemma: str, records: Sequence[rewrites.Record]) -> tuple[str, ...]:
    record = next(record for record in records if record["id"] == dilemma)
    return (*cast(list[str], record["removed"]), outside.originals()[dilemma])


def runs(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", text.lower()))


def flagged(reply: str, keys: Sequence[str]) -> bool:
    words = runs(reply)
    return any(
        key_words
        and any(
            words[i : i + len(key_words)] == key_words
            for i in range(len(words) - len(key_words) + 1)
        )
        for key in keys
        if (key_words := runs(key))
    )
