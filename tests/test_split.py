"""The split: deterministic, about 30% held out, and unmoved by new participants."""

from pathlib import Path

from tests.synthetic import write_raw
from zoon.pilot import fetch, split


def test_split_is_deterministic_and_pinned() -> None:
    """The same identifier always lands in the same part. The pinned values fail if the
    salt, the hash or the threshold changes, any of which would silently redraw the split."""
    labels = [str(n) for n in range(200)]
    assert [split.assign(3, x) for x in labels] == [split.assign(3, x) for x in labels]
    assert split.assign(1, "synthetic-a") == "train"
    assert split.assign(1, "synthetic-b") == "held_out"
    assert split.assign(2, "synthetic-b") == "train"
    assert split.participant_index(1, "synthetic-a") == "cf18d7f821a32b25"


def test_about_thirty_percent_is_held_out_in_each_experiment() -> None:
    for experiment in fetch.SOURCES:
        parts = [split.assign(experiment, str(n)) for n in range(20_000)]
        assert 0.28 < parts.count("held_out") / len(parts) < 0.32


def test_adding_participants_moves_nobody(tmp_path: Path) -> None:
    write_raw(tmp_path / "before", people=60)
    write_raw(tmp_path / "after", people=90)
    before = set(fetch.load(tmp_path / "before")["participant"])
    after = set(fetch.load(tmp_path / "after")["participant"])
    assert before < after
    newcomers = {
        split.participant_index(experiment, str(label))
        for experiment in fetch.SOURCES
        for label in range(61, 91)
    }
    assert after - before <= newcomers
    held_before = fetch.held_out_counts(tmp_path / "before")
    held_after = fetch.held_out_counts(tmp_path / "after")
    assert all(held_before[e] <= held_after[e] for e in fetch.SOURCES)
