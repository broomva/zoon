"""The group loader: the seal on held-out groups, and the cells that are never read."""

import inspect
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from tests.synthetic_groups import SENTINEL, read_synthetic_fours, write_raw
from zoon.interaction import data, split
from zoon.interaction.games import GAMES, ROUNDS

GROUPS = 11


@pytest.fixture(autouse=True)
def synthetic_fours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "read_fours_table", read_synthetic_fours)


def _sealed(intended: Mapping[tuple[str, str], object]) -> set[tuple[str, str]]:
    sealed: set[tuple[str, str]] = set()
    for cell in GAMES:
        raw = [group for game, group in intended if game == cell]
        sealed |= {(cell, group) for group in split.held_out(cell, raw)}
    return sealed


def test_load_returns_training_groups_only(tmp_path: Path) -> None:
    """THE SEAL. Every loaded group is a training group with the contributions that were
    written, and the only thing known about the held-out groups is how many there are."""
    intended = write_raw(tmp_path, GROUPS)
    sealed = _sealed(intended)
    expected = {
        (cell, split.group_index(cell, raw)): rounds
        for (cell, raw), rounds in intended.items()
        if (cell, raw) not in sealed
    }
    loaded = {(group.cell, group.index): group.rounds for group in data.load(tmp_path)}
    assert loaded == expected

    counts = data.held_out_counts(tmp_path)
    assert counts == {cell: 5 for cell in sorted(GAMES)}
    assert sum(counts.values()) == len(sealed)


def test_held_out_groups_are_never_parsed(tmp_path: Path) -> None:
    """Held-out groups whose cells are not numbers load without error, because the loader
    never turns them into numbers. The same damage to one training group raises."""
    intended = write_raw(tmp_path / "plain", GROUPS)
    sealed = _sealed(intended)
    write_raw(tmp_path / "sealed", GROUPS, unreadable=frozenset(sealed))
    assert len(data.load(tmp_path / "sealed")) == len(intended) - len(sealed)

    for size in ("2", "4"):
        one = next(key for key in sorted(intended) if key not in sealed and key[0][0] == size)
        write_raw(tmp_path / f"damaged{size}", GROUPS, unreadable=frozenset({one}))
        with pytest.raises(data.RecordFault, match="not a number"):
            data.load(tmp_path / f"damaged{size}")


def test_other_sessions_rounds_and_answers_are_never_read(tmp_path: Path) -> None:
    """The second session, the rounds after the twentieth and the questionnaire hold a value
    no seat may contribute. Loading succeeds and the value appears nowhere."""
    write_raw(tmp_path, GROUPS)
    groups = data.load(tmp_path)
    assert all(len(group.rounds) == ROUNDS for group in groups)
    assert SENTINEL not in {points for group in groups for row in group.rounds for points in row}


def test_nothing_can_ask_for_held_out_groups() -> None:
    """The training loaders take a directory and nothing that could lift the seal."""
    assert list(inspect.signature(data.load).parameters) == ["raw_dir"]
    assert list(inspect.signature(data.held_out_counts).parameters) == ["raw_dir"]
    assert not [
        name
        for name in vars(data)
        if "held_out" in name and name != "held_out_counts" and name != "held_out"
    ]


def test_a_contribution_must_be_a_whole_number_within_the_endowment() -> None:
    assert data._points(12.0, 12) == 12  # pyright: ignore[reportPrivateUsage]
    for bad in (12.5, -1, 13, float("nan"), "12", True, None):
        with pytest.raises(data.RecordFault):
            data._points(bad, 12)  # pyright: ignore[reportPrivateUsage]


def test_matlab_text_cells_are_unwrapped() -> None:
    assert data._scalar(np.array(["FE"])) == "FE"  # pyright: ignore[reportPrivateUsage]
    assert data._scalar(np.float64(3.0)) == 3.0  # pyright: ignore[reportPrivateUsage]
    assert data._scalar("AI") == "AI"  # pyright: ignore[reportPrivateUsage]


def test_fetch_refuses_a_file_that_differs_from_the_manifest(tmp_path: Path) -> None:
    """With both files already present nothing is downloaded. The first check records them;
    a later change to either one is refused."""
    raw, manifest = tmp_path / "raw", tmp_path / "manifest.json"
    write_raw(raw, 2)
    data.fetch(raw, manifest)
    data.fetch(raw, manifest)
    (raw / data.PAIRS_FILE).write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        data.fetch(raw, manifest)
