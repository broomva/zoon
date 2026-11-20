"""Fetch the public goods data from Zenodo and load the training groups.

Data: Wang, Hilbe and Zhang, "Dataset and computer code for 'The dynamics of cooperation in
asymmetric public goods games'", https://zenodo.org/records/16918146, CC BY 4.0.
Paper: https://doi.org/10.1073/pnas.2525760123

The raw files are never committed. `fetch` downloads them into `data/groups/raw/` and checks
each against `interaction/data-manifest.json`. `load` returns the training groups.

What is read, and what is not:

- Only the linear game, and only its first session. In the second session the same people
  play again in new groups, and the four-player file does not say who is who across
  sessions, so a second-session group cannot be kept apart from the first-session groups
  its members came from.
- Only the first 20 rounds, the ones the authors analysed. The two-player sessions ran for
  22 rounds.
- The threshold game, the second session and the questionnaire answers are never read.
- A held-out group contributes one to a count and nothing else. Its contributions are never
  turned into numbers.
"""

import hashlib
import importlib
import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.split import group_index, held_out

REPO = Path(__file__).resolve().parents[3]
RAW_DIR = REPO / "data" / "groups" / "raw"
MANIFEST = REPO / "interaction" / "data-manifest.json"

RECORD = "https://zenodo.org/records/16918146"
PAIRS_FILE = "LinearPGG_ExperimentalData.xlsx"
FOURS_FILE = "LinearPGG_4P_ExperimentalData.mat"
FILES: tuple[str, ...] = (PAIRS_FILE, FOURS_FILE)

# One sheet per treatment holds every first-session pair once: the sheet whose focal
# participant sat in the first seat. The spellings are the file's own.
PAIR_SHEETS: dict[str, str] = {
    "FE": "Full_equlity_Type1",
    "EI": "Endowment_inequality_Type1",
    "PI": "Productivity_inequality_Type1",
    "AI": "Aligned_inequality_Type1",
    "MI": "Misaliged_inequality_Type1",
}
FOCAL_LABEL = "me"
ROWS_PER_PARTICIPANT = 4
FOUR_COLUMNS: tuple[str, ...] = (
    "Treatment",
    "Session",
    "GroupID",
    "PlayerID",
    "Round",
    "Contribution",
)

Rounds = tuple[tuple[int, ...], ...]
# For each game, each raw group identifier and a function that reads that group's rounds.
# Nothing about a group is turned into numbers until its function is called.
Unread = dict[str, dict[str, Callable[[], Rounds]]]


class RecordFault(ValueError):
    """A group's record cannot be read as 20 rounds of valid contributions."""


@dataclass(frozen=True)
class Group:
    """One training group: its game, an opaque index, and each round's contributions by seat."""

    cell: str
    index: str
    rounds: Rounds


def _url(filename: str) -> str:
    return f"https://zenodo.org/api/records/16918146/files/{filename}/content"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(raw_dir: Path = RAW_DIR, manifest: Path = MANIFEST) -> None:
    """Download any missing raw file, then check every file against the manifest.

    A file with no manifest entry is recorded (URL, size, sha256). A file whose hash differs
    from its entry raises, so a changed upstream file cannot pass unnoticed.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, dict[str, str | int]] = (
        json.loads(manifest.read_text())["files"] if manifest.exists() else {}
    )
    for filename in FILES:
        path = raw_dir / filename
        if not path.exists():
            with urllib.request.urlopen(_url(filename)) as response:
                path.write_bytes(response.read())
        found = {"url": _url(filename), "size": path.stat().st_size, "sha256": _sha256(path)}
        expected = recorded.setdefault(filename, found)
        if expected != found:
            raise ValueError(f"{filename} does not match {manifest.name}: {found}")
    document = {"dataset": RECORD, "licence": "CC BY 4.0", "files": recorded}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _points(cell: object, endowment: int) -> int:
    """Read one contribution: a whole number of points from nothing to the endowment."""
    if isinstance(cell, (str, bool)) or not isinstance(cell, (int, float)):
        raise RecordFault(f"a contribution is not a number: {cell!r}")
    value = float(cell)
    if not value.is_integer() or not 0 <= value <= endowment:
        raise RecordFault(f"a contribution of {cell!r} is not a whole number in 0..{endowment}")
    return int(value)


def _whole(cell: object, what: str) -> int:
    if isinstance(cell, (str, bool)) or not isinstance(cell, (int, float)):
        raise ValueError(f"{what} is not a number: {cell!r}")
    value = float(cell)
    if not value.is_integer():
        raise ValueError(f"{what} is not a whole number: {cell!r}")
    return int(value)


def _by_seat(game: Game, seats: list[list[object]]) -> Rounds:
    """Turn one row of cells per seat into one tuple of contributions per round."""
    if any(len(row) != ROUNDS for row in seats):
        raise RecordFault(f"a seat of {game.cell} does not have {ROUNDS} rounds")
    points = [
        [_points(cell, endowment) for cell in row]
        for row, endowment in zip(seats, game.endowments, strict=True)
    ]
    return tuple(tuple(seat[r] for seat in points) for r in range(ROUNDS))


def _rows(frame: pd.DataFrame) -> list[list[object]]:
    return cast(list[list[object]], frame.astype(object).to_numpy().tolist())


def _unread_pairs(path: Path) -> Unread:
    """Index the first-session pairs of the two-player file without reading a contribution."""
    unread: Unread = {}
    for treatment, sheet in PAIR_SHEETS.items():
        game = GAMES[f"2-{treatment}"]
        frame = cast(
            pd.DataFrame, cast(Any, pd).read_excel(path, sheet_name=sheet, header=0, dtype=object)
        )
        columns = [str(column) for column in frame.columns]
        wanted = [f"R_{r}" for r in range(1, ROUNDS + 1)]
        if columns[2 : 2 + ROUNDS] != wanted:
            raise ValueError(f"{sheet} does not start with rounds 1 to {ROUNDS}")
        rows = _rows(frame)
        if len(rows) % ROWS_PER_PARTICIPANT:
            raise ValueError(f"{sheet} is not made of blocks of {ROWS_PER_PARTICIPANT} rows")
        groups: dict[str, Callable[[], Rounds]] = {}
        for start in range(0, len(rows), ROWS_PER_PARTICIPANT):
            first, second = rows[start], rows[start + 1]
            if first[1] != FOCAL_LABEL or second[1] == FOCAL_LABEL:
                raise ValueError(f"{sheet} row {start + 2}: the first seat is not the focal one")
            raw_group = str(_whole(first[0], f"{sheet} participant number"))
            if raw_group in groups:
                raise ValueError(f"{sheet} repeats participant {raw_group}")
            seats = [first[2 : 2 + ROUNDS], second[2 : 2 + ROUNDS]]
            groups[raw_group] = lambda game=game, seats=seats: _by_seat(game, seats)
        unread[game.cell] = groups
    return unread


def read_fours_table(path: Path) -> pd.DataFrame:
    """Read the four-player file, a MATLAB table, with the optional `mat-io` package."""
    matio: Any = importlib.import_module("matio")
    return cast(pd.DataFrame, matio.load_from_mat(str(path))["all_data"])


def _scalar(cell: object) -> object:
    """Unwrap the one-element arrays MATLAB text cells arrive in."""
    value: object = cast(Any, cell).tolist() if hasattr(cell, "tolist") else cell
    while isinstance(value, list):
        items = cast(list[object], value)
        if len(items) != 1:
            break
        value = items[0]
    return cast(object, value)


def _unread_fours(path: Path) -> Unread:
    """Index the first-session groups of the four-player file without reading a contribution."""
    frame = read_fours_table(path)
    missing = [column for column in FOUR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks the columns {missing}")
    by_group: dict[tuple[str, str], list[tuple[int, int, object]]] = {}
    for treatment, session, group, player, round_, cell in _rows(frame[list(FOUR_COLUMNS)]):
        if _whole(_scalar(session), "a session") != 1:
            continue
        key = (str(_scalar(treatment)), str(_whole(_scalar(group), "a group number")))
        seat = _whole(_scalar(player), "a player number")
        by_group.setdefault(key, []).append((seat, _whole(_scalar(round_), "a round"), cell))

    def rounds_of(game: Game, cells: list[tuple[int, int, object]]) -> Rounds:
        placed = {(seat, round_): cell for seat, round_, cell in cells}
        expected = {(s, r) for s in range(1, game.size + 1) for r in range(1, ROUNDS + 1)}
        if len(cells) != len(expected) or set(placed) != expected:
            raise RecordFault(f"a group of {game.cell} is not {game.size} seats by {ROUNDS} rounds")
        seats = [
            [placed[seat, r] for r in range(1, ROUNDS + 1)] for seat in range(1, game.size + 1)
        ]
        return _by_seat(game, seats)

    unread: Unread = {}
    for (treatment, raw_group), cells in by_group.items():
        cell_name = f"4-{treatment}"
        if cell_name not in GAMES:
            raise ValueError(f"{path.name} has an unknown treatment {treatment!r}")
        game = GAMES[cell_name]
        unread.setdefault(cell_name, {})[raw_group] = lambda game=game, cells=cells: rounds_of(
            game, cells
        )
    return unread


def _unread(raw_dir: Path) -> Unread:
    unread = {**_unread_pairs(raw_dir / PAIRS_FILE), **_unread_fours(raw_dir / FOURS_FILE)}
    if set(unread) != set(GAMES):
        raise ValueError(f"the files hold the games {sorted(unread)}, not {sorted(GAMES)}")
    return unread


def load(raw_dir: Path = RAW_DIR) -> list[Group]:
    """Return the training groups of all eight games, in a fixed order."""
    groups: list[Group] = []
    for cell, readers in sorted(_unread(raw_dir).items()):
        sealed = held_out(cell, readers)
        for raw_group, read in readers.items():
            if raw_group not in sealed:
                groups.append(Group(cell, group_index(cell, raw_group), read()))
    return sorted(groups, key=lambda group: (group.cell, group.index))


def held_out_counts(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Return how many groups of each game are held out. Nothing else is known about them."""
    return {
        cell: len(held_out(cell, readers)) for cell, readers in sorted(_unread(raw_dir).items())
    }


def main() -> None:
    """Fetch the files, check them, and print how many groups each game has in each part."""
    fetch()
    training: dict[str, int] = {}
    for group in load():
        training[group.cell] = training.get(group.cell, 0) + 1
    sealed = held_out_counts()
    print("game  training  held-out")
    for cell in sorted(sealed):
        print(f"{cell}  {training[cell]:8d}  {sealed[cell]:8d}")
    print(f"all   {sum(training.values()):8d}  {sum(sealed.values()):8d}")


if __name__ == "__main__":
    main()
