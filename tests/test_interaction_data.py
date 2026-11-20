"""The group loader: the seal on held-out groups, and the cells that are never asked for."""

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.synthetic_groups import (
    SENTINEL,
    UNREADABLE,
    first_row,
    read_synthetic_fours,
    write_raw,
)
from zoon.interaction import data, split
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS

GROUPS = 11


@pytest.fixture(autouse=True)
def synthetic_fours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "fours_table", read_synthetic_fours)


def _pin(monkeypatch: pytest.MonkeyPatch, raw: Path) -> None:
    """Record the files of `raw` in a manifest of their own and make it the manifest."""
    manifest = raw.parent / f"{raw.name}-manifest.json"
    data.fetch(raw, manifest)
    monkeypatch.setattr(data, "MANIFEST", manifest)


def _sealed(intended: Mapping[tuple[str, str], object]) -> set[tuple[str, str]]:
    sealed: set[tuple[str, str]] = set()
    for cell in GAMES:
        raw = [group for game, group in intended if game == cell]
        sealed |= {(cell, group) for group in split.held_out(cell, raw)}
    return sealed


def test_load_returns_training_groups_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE SEAL. Every loaded group is a training group with the contributions that were
    written, and the only thing known about the held-out groups is how many there are."""
    intended = write_raw(tmp_path / "raw", GROUPS)
    _pin(monkeypatch, tmp_path / "raw")
    sealed = _sealed(intended)
    expected = {
        (cell, split.group_index(cell, raw)): rounds
        for (cell, raw), rounds in intended.items()
        if (cell, raw) not in sealed
    }
    loaded = {(group.cell, group.index): group.rounds for group in data.load(tmp_path / "raw")}
    assert loaded == expected

    counts = data.held_out_counts(tmp_path / "raw")
    assert counts == {cell: 5 for cell in sorted(GAMES)}
    assert sum(counts.values()) == len(sealed)


def test_no_function_returns_a_held_out_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every function of the module that returns groups, private ones included, returns
    training groups only. There is no reader to call for the rest."""
    raw = tmp_path / "raw"
    intended = write_raw(raw, GROUPS)
    _pin(monkeypatch, raw)
    sealed = {(cell, split.group_index(cell, group)) for cell, group in _sealed(intended)}
    module: Any = data
    returned: list[Group] = [
        *data.load(raw),
        *module._training(raw)[0],
        *module._training_pairs(raw / data.PAIRS_FILE)[0],
        *module._training_fours(raw / data.FOURS_FILE)[0],
    ]
    assert returned and not {(group.cell, group.index) for group in returned} & sealed
    makers = {
        name
        for name, value in vars(data).items()
        if inspect.isfunction(value)
        and value.__module__ == data.__name__
        and "Group" in str(inspect.signature(value).return_annotation)
    }
    assert makers == {"load", "_training", "_training_pairs", "_training_fours"}
    assert list(inspect.signature(data.load).parameters) == ["raw_dir"]
    assert list(inspect.signature(data.held_out_counts).parameters) == ["raw_dir"]


def test_held_out_groups_are_never_turned_into_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held-out groups whose cells are text load without error. The same damage to one
    training group raises, and the message does not repeat the cell."""
    intended = write_raw(tmp_path / "plain", GROUPS)
    sealed = _sealed(intended)
    write_raw(tmp_path / "sealed", GROUPS, unreadable=frozenset(sealed))
    _pin(monkeypatch, tmp_path / "sealed")
    assert len(data.load(tmp_path / "sealed")) == len(intended) - len(sealed)

    for size in ("2", "4"):
        one = next(key for key in sorted(intended) if key not in sealed and key[0][0] == size)
        raw = tmp_path / f"damaged{size}"
        write_raw(raw, GROUPS, unreadable=frozenset({one}))
        _pin(monkeypatch, raw)
        with pytest.raises(data.RecordFault, match="not a number") as fault:
            data.load(raw)
        assert UNREADABLE not in str(fault.value)


def test_only_the_cells_of_training_pairs_are_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every read of the spreadsheet is recorded. Round cells are asked for only in the two
    first-session rows of training pairs: never for a held-out pair, a second-session row, a
    round after the twentieth or a questionnaire column."""
    raw = tmp_path / "raw"
    intended = write_raw(raw, GROUPS)
    _pin(monkeypatch, raw)
    sealed = _sealed(intended)
    asked: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    real = data.cells

    def recording(
        sheet: Any, rows: tuple[int, int], columns: tuple[int, int]
    ) -> list[list[object]]:
        asked.append((str(sheet.title), rows, columns))
        return real(sheet, rows, columns)

    monkeypatch.setattr(data, "cells", recording)
    data.load(raw)

    assert {sheet for sheet, _, _ in asked} == set(data.PAIR_SHEETS.values())
    assert max(columns[1] for _, _, columns in asked) == 2 + ROUNDS
    for treatment, sheet in data.PAIR_SHEETS.items():
        cell = f"2-{treatment}"
        training = {
            (first_row(n), first_row(n) + 1)
            for n in range(1, GROUPS + 1)
            if (cell, str(n)) not in sealed
        }
        round_reads = {
            rows
            for name, rows, columns in asked
            if name == sheet and columns[0] > 2 and rows != (1, 1)
        }
        assert round_reads == training
        assert len(training) == GROUPS - 5


def test_other_sessions_rounds_and_answers_are_never_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second session, the rounds after the twentieth and the questionnaire hold text.
    Loading succeeds, so none of it was turned into a contribution."""
    raw = tmp_path / "raw"
    write_raw(raw, GROUPS)
    _pin(monkeypatch, raw)
    groups = data.load(raw)
    assert all(len(group.rounds) == ROUNDS for group in groups)
    assert all(type(points) is int for group in groups for row in group.rounds for points in row)
    table = read_synthetic_fours(raw / data.FOURS_FILE)
    assert SENTINEL in set(table["Contribution"]), "the synthetic second session is text"


def test_a_changed_file_is_refused_before_the_split_can_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split ranks the groups that are in the file. If a file changes, a held-out group
    could rank as a training group, so both loaders refuse files that differ from the
    manifest."""
    raw = tmp_path / "raw"
    write_raw(raw, GROUPS)
    _pin(monkeypatch, raw)
    assert data.load(raw)
    write_raw(raw, GROUPS - 1)
    with pytest.raises(ValueError, match="does not match"):
        data.load(raw)
    with pytest.raises(ValueError, match="does not match"):
        data.held_out_counts(raw)


def test_a_contribution_must_be_a_whole_number_within_the_endowment() -> None:
    module: Any = data
    assert module._points(12.0, 12) == 12
    for bad in (12.5, -1, 13, float("nan"), "12", True, None):
        with pytest.raises(data.RecordFault):
            module._points(bad, 12)


def test_matlab_text_cells_are_unwrapped() -> None:
    module: Any = data
    assert module._scalar(np.array(["FE"])) == "FE"
    assert module._scalar(np.float64(3.0)) == 3.0
    assert module._scalar("AI") == "AI"


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
