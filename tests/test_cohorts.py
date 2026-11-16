"""The cohort draw follows an independently written version of the registered pseudocode."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.synthetic import write_raw
from zoon.pilot import cohorts, fetch, power, split


def check_cohorts(people: list[cohorts.Participant], document: cohorts.Cohorts) -> None:
    by_id = {(p.experiment, p.participant): p for p in people}
    calibration = {(p["experiment"], p["participant"]) for p in document["calibration"]}
    assert len(calibration) == 300
    assert calibration <= by_id.keys()
    assert len(document["rotations"]) == 13
    for rotation, pair in zip(document["rotations"], power.held_out_sets(True, 13), strict=True):
        assert rotation["held_out"] == [fetch.DILEMMAS[d] for d in pair]
        assert len(rotation["examples"]) == 2
        seen: set[tuple[int, str]] = set()
        for example in rotation["examples"]:
            key = (example["experiment"], example["participant"])
            assert key not in calibration and key not in seen
            seen.add(key)
            person = by_id[key]
            first = next(
                i for i, t in enumerate(person.trials, 1) if t.dilemma in rotation["held_out"]
            )
            assert 2 <= example["last"] < first
            assert all(
                t.dilemma not in rotation["held_out"] for t in person.trials[: example["last"]]
            )


def test_pseudocode_and_seal(tmp_path: Path) -> None:
    sealed = frozenset(
        (e, str(i))
        for e in fetch.SOURCES
        for i in range(1, 221)
        if split.assign(e, str(i)) == "held_out"
    )
    intended = write_raw(tmp_path, 220, unreadable=sealed)
    people = cohorts.participants(fetch.load(tmp_path))
    training_ids = {
        (e, split.participant_index(e, label))
        for e, label in intended
        if split.assign(e, label) == "train"
    }
    assert {(p.experiment, p.participant) for p in people} == training_ids
    made = cohorts.generate(people)
    check_cohorts(people, made)
    ids = sorted(training_ids)
    rng = np.random.Generator(np.random.PCG64(20261115))
    order = rng.permutation(len(ids))
    assert [(p["experiment"], p["participant"]) for p in made["calibration"]] == [
        ids[int(i)] for i in order[:300]
    ]
    rest = [ids[int(i)] for i in order[300:]]
    records = {(p.experiment, p.participant): p.trials for p in people}
    for rotation in made["rotations"]:
        first = {
            key: next(i for i, t in enumerate(records[key], 1) if t.dilemma in rotation["held_out"])
            for key in rest
        }
        eligible = [key for key in rest if first[key] >= 3]
        a, b = rng.choice(len(eligible), size=2, replace=False)
        for draw, example in zip((a, b), rotation["examples"], strict=True):
            key = eligible[int(draw)]
            assert (example["experiment"], example["participant"]) == key
            assert example["last"] == int(rng.integers(2, first[key]))
    assert cohorts.serialise(cohorts.generate(list(reversed(people)))) == cohorts.serialise(made)


def test_committed_cohorts_regenerate() -> None:
    if not all((fetch.RAW_DIR / s.filename).exists() for s in fetch.SOURCES.values()):
        pytest.skip("training raw files absent; cohort regeneration requires the published data")
    people = cohorts.load_training()
    document = cohorts.generate(people)
    check_cohorts(people, document)
    assert cohorts.serialise(document) == cohorts.FILE.read_bytes()
    assert cohorts.load() == json.loads(cohorts.FILE.read_text())


def test_incomplete_and_noninteger_records_are_rejected(tmp_path: Path) -> None:
    write_raw(tmp_path, 5)
    table = fetch.load(tmp_path)
    with pytest.raises(ValueError, match="complete"):
        cohorts.participants(table.iloc[1:])
    table.loc[0, "valence"] = 0.123
    with pytest.raises(ValueError, match="integer"):
        cohorts.participants(table)


def test_invalid_draw() -> None:
    with pytest.raises(ValueError, match="insufficient"):
        cohorts.generate([])
    person = cohorts.Participant(1, "index", ())
    with pytest.raises(ValueError, match="unique"):
        cohorts.generate([person, person], 1)
