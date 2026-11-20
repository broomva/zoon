"""The group loader: the seal on held-out groups, and the cells that are never asked for."""

import dataclasses
import importlib
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from tests.synthetic_groups import SENTINEL, UNREADABLE, FakeMatio, first_row, write_raw
from zoon.interaction import data, split
from zoon.interaction.games import GAMES, ROUNDS

GROUPS = 11


@pytest.fixture(autouse=True)
def synthetic_fours(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MATLAB reader library is replaced by one that reads the synthetic file."""
    monkeypatch.setitem(sys.modules, "matio", FakeMatio)


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


PLAIN = (type(None), bool, int, float, str)


def _leaves(value: object) -> list[object]:
    """Flatten a returned value to its plain leaves. Anything that is not plain data, a
    container of plain data or a `Group` is returned as it is, for the caller to reject."""
    if isinstance(value, PLAIN):
        return [value]
    if isinstance(value, data.Group):
        return _leaves(dataclasses.astuple(value))
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        return [leaf for pair in items.items() for part in pair for leaf in _leaves(part)]
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = list[object](cast(Any, value))
        return [leaf for part in parts for leaf in _leaves(part)]
    return [value]


def test_nothing_any_function_returns_holds_a_held_out_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE BOUNDARY. Every function of the module, private ones included, is called on files
    whose held-out and unused cells are marked text. Whatever it returns or raises is plain
    data that holds no marked cell: no held-out contribution, no opened spreadsheet and no
    decoded table leaves the module."""
    raw = tmp_path / "raw"
    intended = write_raw(raw, GROUPS, unreadable=frozenset(_sealed(write_raw(raw, GROUPS))))
    _pin(monkeypatch, raw)
    arguments = (raw, raw / data.PAIRS_FILE, raw / data.FOURS_FILE)
    functions = {
        name: value
        for name, value in vars(data).items()
        if inspect.isfunction(value) and value.__module__ == data.__name__
    }
    takes_a_path = {
        name: function
        for name, function in functions.items()
        if list(inspect.signature(function).parameters) in (["raw_dir"], ["path"])
    }
    assert set(takes_a_path) == {
        "load",
        "held_out_counts",
        "verify",
        "_training",
        "_training_pairs",
        "_training_fours",
        "_sha256",
    }
    assert set(functions) - set(takes_a_path) == {
        "fetch",
        "main",
        "_url",
        "_points",
        "_whole",
        "_by_seat",
        "_scalar",
    }, "a new function must be added to one of the two lists and judged"
    for name, function in takes_a_path.items():
        for argument in arguments:
            try:
                leaves = _leaves(function(argument))
            except Exception as fault:
                leaves = [str(fault)]
            strange = [leaf for leaf in leaves if not isinstance(leaf, PLAIN)]
            assert not strange, f"{name} returned something that is not plain data"
            text = " ".join(leaf for leaf in leaves if isinstance(leaf, str))
            assert UNREADABLE not in text and SENTINEL not in text, name
    assert len(data.load(raw)) == len(intended) - len(_sealed(intended))


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


class _RecordingSheet:
    def __init__(self, sheet: Any, asked: list[tuple[str, tuple[int, int], tuple[int, int]]]):
        self._sheet, self._asked = sheet, asked
        self.max_row = sheet.max_row

    def iter_rows(self, *, min_row: int, max_row: int, min_col: int, max_col: int, **rest: Any):
        self._asked.append((str(self._sheet.title), (min_row, max_row), (min_col, max_col)))
        return self._sheet.iter_rows(
            min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col, **rest
        )


class _RecordingBook:
    def __init__(self, book: Any, asked: list[tuple[str, tuple[int, int], tuple[int, int]]]):
        self._book, self._asked = book, asked

    def __getitem__(self, name: str) -> _RecordingSheet:
        return _RecordingSheet(self._book[name], self._asked)

    def close(self) -> None:
        self._book.close()


def test_only_the_cells_of_training_pairs_are_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spreadsheet library is wrapped so that every range asked of a worksheet is
    recorded. Round cells are asked for only in the two first-session rows of training pairs:
    never for a held-out pair, a second-session row, a round after the twentieth or a
    questionnaire column."""
    raw = tmp_path / "raw"
    intended = write_raw(raw, GROUPS)
    _pin(monkeypatch, raw)
    sealed = _sealed(intended)
    asked: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    openpyxl: Any = importlib.import_module("openpyxl")
    real = openpyxl.load_workbook

    def recording(*positional: Any, **named: Any) -> _RecordingBook:
        return _RecordingBook(real(*positional, **named), asked)

    monkeypatch.setattr(openpyxl, "load_workbook", recording)
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
    table = pd.read_pickle(raw / data.FOURS_FILE)
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
