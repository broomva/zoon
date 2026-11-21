"""The training file: what `seal` writes, what it never asks for, and what `load` reads."""

import ast
import importlib
import inspect
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.synthetic_groups import SENTINEL, UNREADABLE, FakeMatio, first_row, write_raw
from zoon.interaction import data, split
from zoon.interaction.games import GAMES, ROUNDS

GROUPS = 11
PACKAGE = Path(data.__file__).parent


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets its own manifest, and a MATLAB reader that reads the synthetic file."""
    monkeypatch.setattr(data, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setitem(sys.modules, "matio", FakeMatio)


def _sealed(intended: Mapping[tuple[str, str], object]) -> set[tuple[str, str]]:
    sealed: set[tuple[str, str]] = set()
    for cell in GAMES:
        raw = [group for game, group in intended if game == cell]
        sealed |= {(cell, group) for group in split.held_out(cell, raw)}
    return sealed


def _prepare(tmp_path: Path, **damage: Any) -> tuple[Path, Path, dict[tuple[str, str], Any]]:
    """Write synthetic raw files, pin them, and return their place and the training file's."""
    raw, training = tmp_path / "raw", tmp_path / "training.json"
    intended = write_raw(raw, GROUPS, **damage)
    data.fetch(raw)
    return raw, training, dict(intended)


def test_the_training_file_holds_the_training_groups_and_nothing_of_the_rest(
    tmp_path: Path,
) -> None:
    """THE GUARANTEE. Held-out and unused cells are marked text. The training file holds
    every training group as written, a count of the others, and no marked cell."""
    plain = dict(write_raw(tmp_path / "plain", GROUPS))
    sealed = _sealed(plain)
    raw, training, intended = _prepare(tmp_path, unreadable=frozenset(sealed))
    data.seal(raw, training)

    text = training.read_text()
    assert UNREADABLE not in text and SENTINEL not in text
    assert set(json.loads(text)) == {"dataset", "groups", "held_out"}
    expected = {
        (cell, split.group_index(cell, group)): rounds
        for (cell, group), rounds in intended.items()
        if (cell, group) not in sealed
    }
    loaded = {(group.cell, group.index): group.rounds for group in data.load(training)}
    assert loaded == expected
    assert not {index for _, index in loaded} & {split.group_index(c, g) for c, g in sealed}
    assert data.held_out_counts(training) == {cell: 5 for cell in sorted(GAMES)}


def test_loading_needs_no_raw_file_and_no_reader_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the raw files deleted and both reader libraries made unusable, the training
    groups still load. So the analysis, which only loads, cannot reach a raw file."""
    raw, training, _ = _prepare(tmp_path)
    data.seal(raw, training)
    before = data.load(training)

    shutil.rmtree(raw)
    monkeypatch.setattr(data, "RAW_DIR", tmp_path / "gone")
    monkeypatch.setitem(sys.modules, "matio", None)
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    assert data.load(training) == before
    assert sum(data.held_out_counts(training).values()) == 5 * len(GAMES)
    with pytest.raises((ImportError, FileNotFoundError)):
        data.seal(tmp_path / "gone", tmp_path / "again.json")


def test_only_seal_names_the_raw_files() -> None:
    """In the package, the raw files, their directory, the reader libraries and the functions
    that open them are named in `data.py` only, and there only outside `load`, `_read` and
    `held_out_counts`."""
    raw_names = {"RAW_DIR", "PAIRS_FILE", "FOURS_FILE", "FILES", "PAIR_SHEETS", "openpyxl"}
    raw_names |= {"matio", "seal", "fetch", "_pairs", "_fours", "_check"}

    def named(tree: ast.AST) -> set[str]:
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        text = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
        return names | {value for value in text if isinstance(value, str)}

    others = [path for path in PACKAGE.glob("*.py") if path.name != "data.py"]
    assert {path.name for path in others} >= {"measure.py", "games.py", "split.py"}
    for path in others:
        assert not named(ast.parse(path.read_text())) & raw_names, path.name
    module = ast.parse((PACKAGE / "data.py").read_text())
    readers = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in {"load", "_read", "held_out_counts"}
    ]
    assert len(readers) == 3
    for function in readers:
        assert not named(function) & raw_names, function.name


def test_a_changed_raw_file_is_refused_by_every_function_that_opens_one(tmp_path: Path) -> None:
    """The split ranks the groups that are in the file, so a changed file could turn a
    held-out group into a training group. `seal` refuses it, and so does each of the two
    functions that open a raw file when called on its own."""
    raw, training, _ = _prepare(tmp_path)
    data.seal(raw, training)
    write_raw(raw, GROUPS - 1)
    module: Any = data
    for refused in (
        lambda: data.seal(raw, tmp_path / "moved.json"),
        lambda: module._pairs(raw / data.PAIRS_FILE),
        lambda: module._fours(raw / data.FOURS_FILE),
        lambda: data.fetch(raw),
    ):
        with pytest.raises(ValueError, match="does not match"):
            refused()
    assert not (tmp_path / "moved.json").exists()


def test_a_changed_training_file_is_refused(tmp_path: Path) -> None:
    """The training file's hash is pinned when it is first written. A different file, or one
    written from other raw files, is refused by both readers and by `seal`."""
    raw, training, _ = _prepare(tmp_path)
    data.seal(raw, training)
    data.seal(raw, training)
    pinned = json.loads(data.MANIFEST.read_text())["training"]
    assert pinned["groups"] == len(data.load(training))

    document = json.loads(training.read_text())
    document["groups"] = document["groups"][1:]
    training.write_text(json.dumps(document, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        data.load(training)
    with pytest.raises(ValueError, match="does not match"):
        data.held_out_counts(training)

    manifest = json.loads(data.MANIFEST.read_text())
    manifest["training"]["sha256"] = "0" * 64
    data.MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        data.seal(raw, training)
    assert json.loads(data.MANIFEST.read_text())["training"]["sha256"] == "0" * 64


def test_the_committed_manifest_pins_both_raw_files_and_the_training_file() -> None:
    committed = json.loads((data.REPO / "interaction" / "data-manifest.json").read_text())
    assert set(committed["files"]) == set(data.FILES)
    assert committed["training"]["groups"] == 210
    assert len(committed["training"]["sha256"]) == 64


def test_the_training_file_has_the_same_bytes_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text mode ends lines with a carriage return on some platforms. With text mode made to
    behave that way, sealing again still matches the pinned hash, so the training file is
    written as bytes."""
    raw, training, _ = _prepare(tmp_path)
    data.seal(raw, training)
    pinned = training.read_bytes()

    def carriage_returns(self: Path, text: str, *_: Any, **__: Any) -> int:
        return self.write_bytes(text.replace("\n", "\r\n").encode())

    monkeypatch.setattr(Path, "write_text", carriage_returns)
    data.seal(raw, training)
    assert training.read_bytes() == pinned
    assert pinned.endswith(b"\n") and b"\r" not in pinned


def test_a_training_group_that_cannot_be_read_stops_the_seal_without_echoing_it(
    tmp_path: Path,
) -> None:
    plain = dict(write_raw(tmp_path / "plain", GROUPS))
    sealed = _sealed(plain)
    for size in ("2", "4"):
        one = next(key for key in sorted(plain) if key not in sealed and key[0][0] == size)
        raw, training, _ = _prepare(tmp_path / size, unreadable=frozenset({one}))
        with pytest.raises(data.RecordFault, match="not a number") as fault:
            data.seal(raw, training)
        assert UNREADABLE not in str(fault.value)
        assert not training.exists()
        data.MANIFEST.unlink()


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
    raw, training, intended = _prepare(tmp_path)
    sealed = _sealed(intended)
    asked: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    openpyxl: Any = importlib.import_module("openpyxl")
    real = openpyxl.load_workbook

    def recording(*positional: Any, **named: Any) -> _RecordingBook:
        return _RecordingBook(real(*positional, **named), asked)

    monkeypatch.setattr(openpyxl, "load_workbook", recording)
    data.seal(raw, training)

    assert {sheet for sheet, _, _ in asked} == set(data.PAIR_SHEETS.values())
    assert max(columns[1] for _, _, columns in asked) == 2 + ROUNDS
    for treatment, sheet in data.PAIR_SHEETS.items():
        cell = f"2-{treatment}"
        wanted = {
            (first_row(n), first_row(n) + 1)
            for n in range(1, GROUPS + 1)
            if (cell, str(n)) not in sealed
        }
        round_reads = {
            rows
            for name, rows, columns in asked
            if name == sheet and columns[0] > 2 and rows != (1, 1)
        }
        assert round_reads == wanted
        assert len(wanted) == GROUPS - 5


def test_other_sessions_rounds_and_answers_are_never_used(tmp_path: Path) -> None:
    """The second session, the rounds after the twentieth and the questionnaire hold text.
    Sealing succeeds, so none of it was turned into a contribution."""
    raw, training, _ = _prepare(tmp_path)
    data.seal(raw, training)
    groups = data.load(training)
    assert all(len(group.rounds) == ROUNDS for group in groups)
    assert all(type(points) is int for group in groups for row in group.rounds for points in row)
    table = pd.read_pickle(raw / data.FOURS_FILE)
    assert SENTINEL in set(table["Contribution"]), "the synthetic second session is text"


def test_the_loaders_take_a_file_and_nothing_that_could_widen_them() -> None:
    assert list(inspect.signature(data.load).parameters) == ["training"]
    assert list(inspect.signature(data.held_out_counts).parameters) == ["training"]


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


def test_fetch_reuses_hash_correct_files_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    raw = tmp_path / "cached"
    raw.mkdir()
    files: dict[str, object] = {}
    for filename in data.FILES:
        content = b"synthetic cached data"
        (raw / filename).write_bytes(content)
        files[filename] = {
            "url": f"https://zenodo.org/api/records/16918146/files/{filename}/content",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    data.MANIFEST.write_text(json.dumps({"files": files}))

    def refuse_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("cached data must not be downloaded")

    monkeypatch.setattr(data.urllib.request, "urlopen", refuse_network)
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    data.fetch(raw)
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == before
    assert json.loads(data.MANIFEST.read_text())["files"] == files


def test_fetch_downloads_a_missing_file_with_a_time_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import io

    content = b"synthetic download"
    files = {
        filename: {
            "url": f"https://zenodo.org/api/records/16918146/files/{filename}/content",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for filename in data.FILES
    }
    data.MANIFEST.write_text(json.dumps({"files": files}))
    asked: list[tuple[str, object]] = []

    def download(url: str, *, timeout: object = None) -> io.BytesIO:
        asked.append((url, timeout))
        return io.BytesIO(content)

    monkeypatch.setattr(data.urllib.request, "urlopen", download)
    raw = tmp_path / "fresh"
    data.fetch(raw)
    assert asked == [(files[filename]["url"], data.DOWNLOAD_TIMEOUT) for filename in data.FILES]
    assert data.DOWNLOAD_TIMEOUT > 0
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == dict.fromkeys(data.FILES, content)


@pytest.mark.parametrize("length", [19, 21])
def test_a_seat_has_exactly_the_registered_round_count(length: int) -> None:
    module: Any = data
    with pytest.raises(data.RecordFault, match="20 rounds"):
        module._by_seat(GAMES["2-FE"], [[0] * ROUNDS, [0] * length])
