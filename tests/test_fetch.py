"""The loader: the seal on held-out participants, and personal data kept out."""

import inspect
import os
from pathlib import Path

import pytest

from tests.synthetic import PERSONAL_COLUMNS, SECRET, write_raw
from zoon.pilot import fetch, split

PEOPLE = 80


def _parts(intended: dict[tuple[int, str], dict[str, int]]) -> tuple[set[str], set[str]]:
    """Opaque indices of the included participants, by the part the split gives them."""
    train: set[str] = set()
    held_out: set[str] = set()
    for experiment, label in intended:
        part = train if split.assign(experiment, label) == "train" else held_out
        part.add(split.participant_index(experiment, label))
    return train, held_out


def test_load_returns_training_participants_only(tmp_path: Path) -> None:
    """THE SEAL. Every loaded participant is a training participant, no held-out participant
    is loaded, and the only thing known about the held-out ones is how many there are."""
    intended = write_raw(tmp_path, PEOPLE)
    train, held_out = _parts(intended)
    assert held_out, "the synthetic data must contain held-out participants"

    loaded = set(fetch.load(tmp_path)["participant"])
    assert loaded == train
    assert not loaded & held_out

    counts = fetch.held_out_counts(tmp_path)
    assert sum(counts.values()) == len(held_out)
    assert all(type(count) is int for count in counts.values())


def test_held_out_records_are_never_parsed(tmp_path: Path) -> None:
    """Held-out records with no dilemma order and no responses load without error, because
    the loader never reads those cells. The same damage to a training record raises."""
    everyone = [(e, str(n)) for e in fetch.SOURCES for n in range(1, PEOPLE + 1)]
    held_out = frozenset(key for key in everyone if split.assign(*key) == "held_out")
    write_raw(tmp_path / "sealed", PEOPLE, unreadable=held_out)
    assert len(fetch.load(tmp_path / "sealed")) > 0

    included = write_raw(tmp_path / "damaged", PEOPLE)
    one_training = next(key for key in included if split.assign(*key) == "train")
    write_raw(tmp_path / "damaged", PEOPLE, unreadable=frozenset({one_training}))
    with pytest.raises(ValueError, match="unexpected dilemma order"):
        fetch.load(tmp_path / "damaged")


def test_nothing_can_ask_for_held_out_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training loaders cannot be unsealed by arguments or connection settings."""
    assert list(inspect.signature(fetch.load).parameters) == ["raw_dir"]
    assert list(inspect.signature(fetch.held_out_counts).parameters) == ["raw_dir"]

    for module in Path(inspect.getfile(fetch)).parent.glob("*.py"):
        source = module.read_text()
        if module.name != "clients.py":
            assert "environ" not in source and "getenv" not in source, module.name

    public = {name for name in vars(fetch) | vars(split) if "held" in name.lower()}
    assert public == {"held_out_counts", "HELD_OUT_FRACTION"}

    write_raw(tmp_path, PEOPLE)
    expected = fetch.load(tmp_path)
    for name in ("ZOON_UNSEAL", "UNSEAL", "HELD_OUT", "ZOON_HELD_OUT", "ZOON_SPLIT"):
        monkeypatch.setenv(name, "1")
    assert os.environ["ZOON_UNSEAL"] == "1"
    assert fetch.load(tmp_path).equals(expected)


def test_personal_data_cannot_reach_the_table(tmp_path: Path) -> None:
    """The raw files carry personal columns filled with a marker. No such column, no marker
    and no raw row label appears in the loaded table."""
    write_raw(tmp_path, PEOPLE)
    for source in fetch.SOURCES.values():
        header = (tmp_path / source.filename).read_text(encoding="latin-1").splitlines()[0]
        assert all(column in header for column in PERSONAL_COLUMNS)

    table = fetch.load(tmp_path)
    assert list(table.columns) == [
        "experiment",
        "participant",
        "condition",
        "dilemma",
        "trial",
        "cbr_choice",
        "action_choice",
        "success",
        "valence",
    ]
    assert not any(fetch.FORBIDDEN_COLUMN.search(column) for column in table.columns)
    assert SECRET not in table.to_csv()
    assert all(len(index) == 16 and not index.isdigit() for index in table["participant"])


@pytest.mark.parametrize("column", PERSONAL_COLUMNS)
def test_allowlist_refuses_personal_columns(column: str) -> None:
    """Putting a personal column on the list of columns to read is an error, not a leak."""
    source = fetch.Source("x.csv", "https://example.org", ",", (column, "FL_1_DO"), ())
    with pytest.raises(ValueError, match="personal-data columns"):
        fetch.allowed_columns(source)
    for real in fetch.SOURCES.values():
        assert not any(fetch.FORBIDDEN_COLUMN.search(name) for name in fetch.allowed_columns(real))


def test_table_matches_what_participants_did(tmp_path: Path) -> None:
    """Choices, trial order and outcomes survive the trip through the raw layout."""
    intended = write_raw(tmp_path, PEOPLE)
    table = fetch.load(tmp_path)
    by_index = {
        split.participant_index(experiment, label): chosen
        for (experiment, label), chosen in intended.items()
    }
    for index, trials in table.groupby("participant"):
        assert list(trials["trial"]) == list(range(1, 14))
        assert sorted(trials["dilemma"]) == sorted(fetch.DILEMMAS)
        assert (
            dict(zip(trials["dilemma"], trials["cbr_choice"], strict=True)) == by_index[str(index)]
        )
        cbr_succeeds = trials["condition"].iloc[0] == "cbr_success"
        assert list(trials["success"]) == [int(c == cbr_succeeds) for c in trials["cbr_choice"]]
        for dilemma, cbr, action in zip(
            trials["dilemma"], trials["cbr_choice"], trials["action_choice"], strict=True
        ):
            assert bool(action) == (bool(cbr) == fetch.CBR_IS_ACTION[dilemma])
    assert table["valence"].between(-1, 1).all()


def test_fetch_records_then_verifies_hashes(tmp_path: Path) -> None:
    """With the files already present nothing is downloaded. The first run records each
    file's hash; a later run refuses a file that has changed."""
    raw_dir, manifest = tmp_path / "raw", tmp_path / "data-manifest.json"
    write_raw(raw_dir, 5)
    fetch.fetch(raw_dir, manifest)
    recorded = manifest.read_text()
    assert all(source.filename in recorded for source in fetch.SOURCES.values())
    fetch.fetch(raw_dir, manifest)
    assert manifest.read_text() == recorded

    changed = raw_dir / fetch.SOURCES[1].filename
    changed.write_text(changed.read_text(encoding="latin-1") + "extra\n", encoding="latin-1")
    with pytest.raises(ValueError, match="does not match"):
        fetch.fetch(raw_dir, manifest)
