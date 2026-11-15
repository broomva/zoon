"""Validate draft judge stimuli without fetching sources or reading participant data.

Run with ``python -m zoon.pilot.rewrites``. Source word counts include the vignette and
question, exclude headings and outcomes, and count whitespace-separated words. Mechanical
checks cannot establish that a rewrite preserves the moral trade-off or prevents recognition.
"""

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from zoon.pilot.fetch import DILEMMAS, REPO
from zoon.pilot.outside import originals, published

DIRECTORY = REPO / "pilot" / "rewrites"
COUNTS = REPO / "pilot" / "source-word-counts.json"
PEOPLE_COUNTS = REPO / "pilot" / "people-counts.json"
REQUIRED = ("id", "set", "source", "suggested_action", "kept", "removed", "generalised", "rewrite")
SOURCE_FIELDS = (
    "authors",
    "title",
    "url",
    "materials_url",
    "locator",
    "license",
    "license_url",
    "changes",
)
# Positive and qualified-positive answers in Tables 1 and 2. "Not really" is negative.
REASONABLE = frozenset(("yes", "rather yes", "somewhat"))
FRAMING_CODES = REASONABLE | {"no", "not really"}
YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")
Record = dict[str, object]


def word_count(text: str) -> int:
    """Count whitespace-separated words, including the final question."""
    return len(text.split())


def _record(value: object) -> Record:
    if not isinstance(value, dict):
        raise ValueError("record must be a JSON object")
    return cast(Record, value)


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _items(record: Record, key: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    items = cast(list[object], value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{key} must contain non-empty strings")
    return cast(list[str], items)


def expected() -> dict[str, tuple[str, bool]]:
    """The exact task IDs and factual outside IDs, with the original paper's reversal code.

    The task's codes differ for Tax and RobinHood. Reversal follows the original paper for
    both sets, as specified in the arms note, rather than the adaptation's coding.
    """
    table = published()
    mapping = originals()
    task_originals = set(mapping.values())
    names = {name: ("task", mapping[name]) for name in DILEMMAS}
    names.update(
        (name, ("outside", name))
        for name, dilemma in table.items()
        if dilemma.kind == "factual" and name not in task_originals
    )
    result: dict[str, tuple[str, bool]] = {}
    for name, (group, original) in names.items():
        codes = table[original].codes
        if codes is None or codes.opposite_framing.casefold() not in FRAMING_CODES:
            raise ValueError(f"{name}: unknown opposite framing code")
        result[name] = group, codes.opposite_framing.casefold() in REASONABLE
    return result


def load(directory: Path = DIRECTORY) -> list[Record]:
    """Read one JSON object per stimulus; reject unexpected files rather than skipping them."""
    result: list[Record] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            raise ValueError(f"unexpected stimulus file: {path.name}")
        value: object = json.loads(path.read_text(encoding="utf-8"))
        result.append(_record(value))
    return result


def load_counts(path: Path = COUNTS) -> Record:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return _record(value)


def load_people_counts(path: Path = PEOPLE_COUNTS) -> Record:
    """Read source people counts and the stimulus phrases expressing each side.

    Unnumbered sides are described separately; absence of a number never means zero.
    Mentions locate a count in context, including singular people and lower bounds.
    They do not prove semantic equivalence to the source.
    """
    return _record(json.loads(path.read_text(encoding="utf-8")))


def _normal(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _generalisations(record: Record) -> list[Record]:
    value = record["generalised"]
    if not isinstance(value, list):
        raise ValueError("generalised must be a list")
    result = [_record(item) for item in cast(list[object], value)]
    for item in result:
        for key in ("from", "to", "why"):
            _text(item, key)
    if not result:
        notes = _text(record, "notes")
        prefix = "No further generalisation:"
        if not notes.startswith(prefix) or not notes.removeprefix(prefix).strip():
            raise ValueError(
                "empty generalised needs notes explaining why nothing can be generalised"
            )
    return result


def _people_mentions(value: object) -> list[list[str]]:
    table = _record(value)
    _text(table, "unspecified")
    counts = table.get("counts")
    if not isinstance(counts, list):
        raise ValueError("people counts must be a list")
    mentions: list[list[str]] = []
    for value in cast(list[object], counts):
        entry = _record(value)
        _text(entry, "side")
        people = entry.get("people")
        if not (
            (type(people) is int and people > 0)
            or (isinstance(people, str) and people in {"thousands", "millions"})
        ):
            raise ValueError("people count must be a positive integer or a source magnitude")
        phrases = _items(entry, "mentions")
        if people != 1:
            forms = [str(people)]
            if isinstance(people, int):
                forms.append(f"{people:,}")
                words = [
                    "zero",
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                ]
                if people < len(words):
                    forms.append(words[people])
                if people == 2:
                    forms.append("both")
            if any(not _contains_mention(phrase, forms) for phrase in phrases):
                raise ValueError("people count mentions must express their recorded number")
        mentions.append(phrases)
    return mentions


def _contains_mention(text: str, mentions: Sequence[str]) -> bool:
    return any(
        re.search(r"(?<!\w)" + re.escape(_normal(mention)) + r"(?!\w)", _normal(text))
        for mention in mentions
    )


def _question(text: str) -> tuple[str, str]:
    body, separator, question = text.rpartition("Do you ")
    if (
        not separator
        or not body.strip()
        or not question.endswith("?")
        or "?" in body + question[:-1]
    ):
        raise ValueError("rewrite must end with one Do you yes/no question")
    return body, separator + question


def validate(
    records: Sequence[Record],
    counts: Mapping[str, object],
    people_counts: Mapping[str, object] | None = None,
) -> None:
    """Raise on any invalid draft; this does not confer human approval or freeze stimuli."""
    contract = expected()
    if set(counts) != set(contract):
        raise ValueError("source count IDs must match the task and outside IDs exactly")
    people_counts = load_people_counts() if people_counts is None else people_counts
    if set(people_counts) != set(contract):
        raise ValueError("people count IDs must match the task and outside IDs exactly")
    people = {name: _people_mentions(value) for name, value in people_counts.items()}
    seen: set[str] = set()
    for record in records:
        missing = set(REQUIRED) - record.keys()
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")
        name = _text(record, "id")
        if name not in contract:
            raise ValueError(f"unknown dilemma ID: {name}")
        if name in seen:
            raise ValueError(f"duplicate dilemma ID: {name}")
        seen.add(name)
        group, reversible = contract[name]
        if _text(record, "set") != group:
            raise ValueError(f"{name}: set does not match the committed tables")
        source = _record(record["source"])
        for key in SOURCE_FIELDS:
            _text(source, key)
        if source["license"] != "CC BY 4.0":
            raise ValueError(f"{name}: source license must be CC BY 4.0")
        _text(record, "suggested_action")
        _items(record, "kept")
        removed = _items(record, "removed")
        generalised = _generalisations(record)
        if "notes" in record:
            _text(record, "notes")
        for key in ("reversed", "reversed_suggested_action"):
            if (key in record) != reversible:
                raise ValueError(f"{name}: {key} presence must match the paper's reversal code")
        if reversible:
            _text(record, "reversed_suggested_action")
        count = counts[name]
        if type(count) is not int or count <= 0:
            raise ValueError(f"{name}: source word count must be a positive integer")
        rewrite_body, question = _question(_text(record, "rewrite"))
        for key in ("rewrite", "reversed") if reversible else ("rewrite",):
            text = _text(record, key)
            if any(_normal(item) in _normal(text) for item in removed):
                raise ValueError(f"{name}: {key} contains a removed string")
            if any(_normal(str(item["from"])) in _normal(text) for item in generalised):
                raise ValueError(f"{name}: {key} contains a generalised source phrase")
            if YEAR.search(_normal(text)):
                raise ValueError(f"{name}: {key} contains a four-digit year-like number")
            if not 7 * count <= 10 * word_count(text) <= 13 * count:
                raise ValueError(f"{name}: {key} word count must be within 70-130% of source")
            if any(not _contains_mention(text, mentions) for mentions in people[name]):
                raise ValueError(f"{name}: {key} is missing a recorded people count")
            body, final = _question(text)
            if key == "reversed" and (body != rewrite_body or final == question):
                raise ValueError(f"{name}: reversed must change only the final question")
    if seen != set(contract):
        raise ValueError(f"missing dilemma IDs: {sorted(set(contract) - seen)}")


if __name__ == "__main__":
    drafts = load()
    validate(drafts, load_counts())
    print(f"Validated {len(drafts)} draft rewrites; human checks and reading remain pending.")
