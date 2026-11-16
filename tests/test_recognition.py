"""Recognition uses whole runs, adjacent in order, and unknown vetoes a flag."""

from typing import cast

import pytest

from zoon.pilot import fetch, outside, recognition, rewrites


@pytest.mark.parametrize("reply", ["unknown", "Unknown. Medicine Costs", "UNKNOWN: Belgian"])
def test_unknown_veto(reply: str) -> None:
    assert not recognition.flagged(reply, recognition.keys("Medicine", rewrites.load()))


@pytest.mark.parametrize(
    "dilemma,reply,expected",
    [
        ("Medicine", "Medicine Costs in Belgium", True),
        ("Ransom", "The Sahara ransom case, 2003", True),
        ("RobinHood", "A German case", False),
        ("RobinHood", "Germany", True),
        ("RobinHood", "Germanywide", False),
        ("Medicine", "Medicine Other Costs", False),
        ("Medicine", "Costs Medicine", False),
    ],
)
def test_registered_examples(dilemma: str, reply: str, expected: bool) -> None:
    assert recognition.flagged(reply, recognition.keys(dilemma, rewrites.load())) == expected


def test_all_keys_and_punctuation() -> None:
    records = {str(r["id"]): r for r in rewrites.load()}
    for dilemma in fetch.DILEMMAS:
        assert recognition.keys(dilemma, rewrites.load()) == (
            *cast(list[str], records[dilemma]["removed"]),
            outside.originals()[dilemma],
        )
    assert recognition.flagged("Sahara—RANSOM, 2003", ["(Sahara) Ransom"])
    assert recognition.flagged("Abc_123", ["abc 123"])
    assert not recognition.flagged("anything", ["", "!!!"])
