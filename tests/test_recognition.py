"""Recognition uses whole runs, adjacent in order."""

from typing import cast

import pytest

from zoon.pilot import fetch, outside, recognition, rewrites


@pytest.mark.parametrize(
    "reply,expected",
    [("unknown", False), ("Unknown. Medicine Costs", True), ("UNKNOWN: Belgian", True)],
)
def test_unknown_does_not_veto_a_key(reply: str, expected: bool) -> None:
    assert recognition.flagged(reply, recognition.keys("Medicine", rewrites.load())) == expected
    assert recognition.flagged("Unknown, but possibly the 1977 kidnapping", ["1977"])


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
