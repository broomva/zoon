"""Corrupt otherwise valid stimuli so every validation guard has a failing example.

These fixtures contain no source transcripts or participant data and need no network.
"""

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from zoon.pilot import rewrites


@pytest.fixture
def records() -> list[rewrites.Record]:
    return rewrites.load()


def one(records: list[rewrites.Record], name: str) -> rewrites.Record:
    return next(record for record in records if record["id"] == name)


def check(records: list[rewrites.Record]) -> None:
    rewrites.validate(records, rewrites.load_counts())


def test_committed_rewrites() -> None:
    records = rewrites.load()
    check(records)
    assert len(records) == 38
    assert sum(record["set"] == "task" for record in records) == 13
    assert sum(record["set"] == "outside" for record in records) == 25
    assert {str(record["id"]) for record in records if "reversed" in record} == {
        "Suicide",
        "Medicine",
        "RAF",
        "Rwanda",
        "Ransom",
        "Tax",
        "Vet",
        "Endow",
        "Organ Law",
        "Mare Nostrum",
        "Aid for Syria",
        "Test Tube Sibling",
        "Ashley Treatment",
        "Bishops",
        "Trail of Tears",
        "Hogesa (Hooligan Demo)",
        "Mass Rape",
    }


@pytest.mark.parametrize(
    "key", ["id", "set", "source", "suggested_action", "kept", "removed", "rewrite"]
)
def test_missing_required_field(records: list[rewrites.Record], key: str) -> None:
    del records[0][key]
    with pytest.raises(ValueError, match="missing required fields"):
        check(records)


@pytest.mark.parametrize("key", ["id", "set", "suggested_action", "rewrite", "notes"])
@pytest.mark.parametrize("value", ["", "  ", 7, [], None])
def test_text_fields(records: list[rewrites.Record], key: str, value: object) -> None:
    records[0][key] = value
    with pytest.raises(ValueError, match=f"{key} must be a non-empty string"):
        check(records)


@pytest.mark.parametrize("key", ["kept", "removed"])
@pytest.mark.parametrize("value", [[], "not a list", [""], [3], [None]])
def test_fact_lists(records: list[rewrites.Record], key: str, value: object) -> None:
    records[0][key] = value
    with pytest.raises(ValueError, match=key):
        check(records)


@pytest.mark.parametrize(
    "key",
    ["authors", "title", "url", "materials_url", "locator", "license", "license_url", "changes"],
)
def test_source_fields(records: list[rewrites.Record], key: str) -> None:
    source = records[0]["source"]
    assert isinstance(source, dict)
    del source[key]
    with pytest.raises(ValueError, match=key):
        check(records)


def test_source_license(records: list[rewrites.Record]) -> None:
    source = records[0]["source"]
    assert isinstance(source, dict)
    source["license"] = "unspecified"
    with pytest.raises(ValueError, match="source license"):
        check(records)


@pytest.mark.parametrize("value", [[], "citation", None])
def test_source_is_an_object(records: list[rewrites.Record], value: object) -> None:
    records[0]["source"] = value
    with pytest.raises(ValueError, match="JSON object"):
        check(records)


@pytest.mark.parametrize("field", ["rewrite", "reversed"])
@pytest.mark.parametrize("token", ["cAnAdA", "\uff23\uff41\uff4e\uff41\uff44\uff41"])
def test_removed_string(records: list[rewrites.Record], field: str, token: str) -> None:
    record = one(records, "Suicide")
    record[field] = str(record[field]).replace("You are", token + ": You are", 1)
    with pytest.raises(ValueError, match="contains a removed string"):
        check(records)


@pytest.mark.parametrize("field", ["rewrite", "reversed"])
@pytest.mark.parametrize("year", ["1987", "\uff11\uff19\uff18\uff17", "1990s"])
def test_four_digit_number(records: list[rewrites.Record], field: str, year: str) -> None:
    record = one(records, "Suicide")
    record[field] = str(record[field]).replace("You are", year + ": You are", 1)
    with pytest.raises(ValueError, match="four-digit"):
        check(records)


def test_grouped_number_is_not_a_year(records: list[rewrites.Record]) -> None:
    record = one(records, "Suicide")
    for field in ("rewrite", "reversed"):
        record[field] = str(record[field]).replace("You are", "4,000: You are", 1)
    check(records)


@pytest.mark.parametrize("field", ["rewrite", "reversed"])
@pytest.mark.parametrize("long", [True, False])
def test_length_bounds(records: list[rewrites.Record], field: str, long: bool) -> None:
    record = one(records, "Suicide")
    record[field] = "You are deciding. " + ("word " * 250 if long else "") + "Do you act?"
    with pytest.raises(ValueError, match="word count"):
        check(records)


@pytest.mark.parametrize("size,valid", [(74, False), (75, True), (125, True), (126, False)])
def test_exact_length_boundaries(records: list[rewrites.Record], size: int, valid: bool) -> None:
    record = one(records, "Ladder")
    counts = rewrites.load_counts()
    counts["Ladder"] = 100
    record["rewrite"] = "You " + "word " * (size - 4) + "Do you act?"
    if valid:
        rewrites.validate(records, counts)
    else:
        with pytest.raises(ValueError, match="word count"):
            rewrites.validate(records, counts)


@pytest.mark.parametrize("name", ["Ladder", "Organ Law"])
def test_missing_id_from_either_set(records: list[rewrites.Record], name: str) -> None:
    records.remove(one(records, name))
    with pytest.raises(ValueError, match="missing dilemma IDs"):
        check(records)


@pytest.mark.parametrize("name", ["Ladder", "Organ Law"])
def test_unknown_id_in_either_set(records: list[rewrites.Record], name: str) -> None:
    one(records, name)["id"] = "renamed"
    with pytest.raises(ValueError, match="unknown dilemma ID"):
        check(records)


@pytest.mark.parametrize("name", ["Ladder", "Organ Law"])
def test_wrong_set(records: list[rewrites.Record], name: str) -> None:
    record = one(records, name)
    record["set"] = "outside" if record["set"] == "task" else "task"
    with pytest.raises(ValueError, match="set does not match"):
        check(records)


def test_duplicate_id(records: list[rewrites.Record]) -> None:
    records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match="duplicate dilemma ID"):
        check(records)


@pytest.mark.parametrize("field", ["reversed", "reversed_suggested_action"])
@pytest.mark.parametrize("name", ["Suicide", "Medicine", "Tax", "Bishops", "Organ Law"])
def test_required_reversals(records: list[rewrites.Record], field: str, name: str) -> None:
    del one(records, name)[field]
    with pytest.raises(ValueError, match="presence must match"):
        check(records)


@pytest.mark.parametrize("field", ["reversed", "reversed_suggested_action"])
@pytest.mark.parametrize("name", ["Ladder", "RobinHood", "Conjoined Twins"])
def test_unexpected_reversals(records: list[rewrites.Record], field: str, name: str) -> None:
    one(records, name)[field] = "unexpected"
    with pytest.raises(ValueError, match="presence must match"):
        check(records)


@pytest.mark.parametrize("field", ["reversed", "reversed_suggested_action"])
def test_reversal_nonempty(records: list[rewrites.Record], field: str) -> None:
    one(records, "Suicide")[field] = ""
    with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
        check(records)


@pytest.mark.parametrize("change", ["body", "same question"])
def test_reversal_changes_only_question(records: list[rewrites.Record], change: str) -> None:
    record = one(records, "Suicide")
    if change == "body":
        record["reversed"] = str(record["reversed"]).replace("You are", "You were", 1)
    else:
        record["reversed"] = record["rewrite"]
    with pytest.raises(ValueError, match="change only the final question"):
        check(records)


@pytest.mark.parametrize("field", ["rewrite", "reversed"])
def test_final_question(records: list[rewrites.Record], field: str) -> None:
    record = one(records, "Suicide")
    record[field] = str(record[field]).replace("Do you ", "Would it be appropriate to ")
    with pytest.raises(ValueError, match="yes/no question"):
        check(records)


@pytest.mark.parametrize("value", [0, -1, 3.5, True, "100"])
def test_source_count_positive_integer(records: list[rewrites.Record], value: object) -> None:
    counts = rewrites.load_counts()
    counts["Ladder"] = value
    with pytest.raises(ValueError, match="positive integer"):
        rewrites.validate(records, counts)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_source_count_coverage(records: list[rewrites.Record], change: str) -> None:
    counts = rewrites.load_counts()
    if change == "missing":
        del counts["Ladder"]
    else:
        counts["unexpected"] = 100
    with pytest.raises(ValueError, match="source count IDs"):
        rewrites.validate(records, counts)


def test_word_count() -> None:
    assert rewrites.word_count("  One\n two\tthree-word  ") == 3


def test_loader_rejects_extra_file(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("not a stimulus")
    with pytest.raises(ValueError, match="unexpected stimulus file"):
        rewrites.load(tmp_path)


def test_loader_rejects_array(tmp_path: Path) -> None:
    (tmp_path / "array.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        rewrites.load(tmp_path)


def test_one_file_per_dilemma() -> None:
    paths = list(rewrites.DIRECTORY.glob("*.json"))
    assert len(paths) == 38
    assert all(isinstance(json.loads(path.read_text()), dict) for path in paths)


def test_documented_inventory_and_notes_match_files() -> None:
    note = (rewrites.REPO / "docs" / "pilot" / "rewrites.md").read_text()
    counts = rewrites.load_counts()
    records = rewrites.load()
    inventory = [line for line in note.splitlines() if line.startswith("| ")][1:]
    assert len(inventory) == 38
    for record in records:
        name = str(record["id"])
        text = str(record["rewrite"])
        kept, removed = record["kept"], record["removed"]
        assert isinstance(kept, list) and isinstance(removed, list)
        reversed_words = (
            rewrites.word_count(str(record["reversed"])) if "reversed" in record else "—"
        )
        row = (
            f"| {name} | {record['set']} | {counts[name]} | {rewrites.word_count(text)} | "
            f"{reversed_words} | {len(cast(list[object], kept))} | "
            f"{len(cast(list[object], removed))} |"
        )
        assert row in inventory
        if "notes" in record:
            assert f"- **{name}**: {record['notes']}" in note
