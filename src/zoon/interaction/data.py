"""Fetch the public goods data from Zenodo and load the training groups.

Data: Wang, Hilbe and Zhang, "Dataset and computer code for 'The dynamics of cooperation in
asymmetric public goods games'", https://zenodo.org/records/16918146, CC BY 4.0.
Paper: https://doi.org/10.1073/pnas.2525760123

The raw files are never committed. `fetch` downloads them into `data/groups/raw/` and checks
each against `interaction/data-manifest.json`. `load` returns the training groups.

What is used, and what is not:

- Only the linear game, and only its first session. In the second session the same people
  play again in new groups, and the four-player file does not say who is who across
  sessions, so a second-session group cannot be kept apart from the first-session groups
  its members came from.
- Only the first 20 rounds, the ones the authors analysed. The two-player sessions ran for
  22 rounds.
- The threshold game, the second session and the questionnaire answers are never used.
- A held-out group contributes one to a count and nothing else.

How far the seal goes, by file:

- The two-player spreadsheet is read cell range by cell range. The loader asks for the two
  identifier columns of a sheet, and then for the 20 round cells of the two first-session
  rows of each training pair. It never asks for a held-out pair's rounds, a second-session
  row, a round after the twentieth or a questionnaire column.
- The four-player file is a MATLAB table that its reader library decodes whole. The loader
  takes the identifier columns, works out which rows belong to first-session training
  groups, and takes the contributions of those rows only. Nothing else is kept or turned
  into a number.

Both loaders check the files against the manifest first, so the split cannot move with a
changed file. No function in this module returns a held-out group.
"""

import hashlib
import importlib
import json
import urllib.request
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
FOUR_KEYS: tuple[str, ...] = ("Treatment", "Session", "GroupID", "PlayerID", "Round")
FOUR_VALUE = "Contribution"

Rounds = tuple[tuple[int, ...], ...]


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


def verify(raw_dir: Path) -> None:
    """Refuse to read files that are missing from the manifest or differ from it."""
    recorded: dict[str, dict[str, str | int]] = json.loads(MANIFEST.read_text())["files"]
    for filename in FILES:
        path = raw_dir / filename
        entry = recorded.get(filename)
        if entry is None or entry["sha256"] != _sha256(path):
            raise ValueError(f"{filename} does not match {MANIFEST.name}")


def _points(cell: object, endowment: int) -> int:
    """Read one contribution: a whole number of points from nothing to the endowment.

    The message never repeats the cell.
    """
    if isinstance(cell, (str, bool)) or not isinstance(cell, (int, float)):
        raise RecordFault("a contribution is not a number")
    value = float(cell)
    if not value.is_integer() or not 0 <= value <= endowment:
        raise RecordFault(f"a contribution is not a whole number in 0..{endowment}")
    return int(value)


def _whole(cell: object, what: str) -> int:
    if isinstance(cell, (str, bool)) or not isinstance(cell, (int, float)):
        raise ValueError(f"{what} is not a number")
    value = float(cell)
    if not value.is_integer():
        raise ValueError(f"{what} is not a whole number")
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


def cells(sheet: Any, rows: tuple[int, int], columns: tuple[int, int]) -> list[list[object]]:
    """Return the values of one rectangle of a worksheet, rows and columns counted from 1.

    Every read of the spreadsheet goes through this function, so a test can record exactly
    which cells were asked for.
    """
    found = sheet.iter_rows(
        min_row=rows[0], max_row=rows[1], min_col=columns[0], max_col=columns[1], values_only=True
    )
    return [list(row) for row in found]


ID_COLUMNS = (1, 2)
ROUND_COLUMNS = (3, 2 + ROUNDS)


def _training_pairs(path: Path) -> tuple[list[Group], dict[str, int]]:
    """Read the first-session training pairs, and count the held-out ones."""
    openpyxl: Any = importlib.import_module("openpyxl")
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    groups: list[Group] = []
    counts: dict[str, int] = {}
    try:
        for treatment, name in PAIR_SHEETS.items():
            game, sheet = GAMES[f"2-{treatment}"], book[name]
            header = cells(sheet, (1, 1), ROUND_COLUMNS)[0]
            if header != [f"R_{r}" for r in range(1, ROUNDS + 1)]:
                raise ValueError(f"{name} does not start with rounds 1 to {ROUNDS}")
            labels = cells(sheet, (2, sheet.max_row), ID_COLUMNS)
            if len(labels) % ROWS_PER_PARTICIPANT:
                raise ValueError(f"{name} is not made of blocks of {ROWS_PER_PARTICIPANT} rows")
            first_row: dict[str, int] = {}
            for start in range(0, len(labels), ROWS_PER_PARTICIPANT):
                first, second = labels[start], labels[start + 1]
                if first[1] != FOCAL_LABEL or second[1] == FOCAL_LABEL:
                    raise ValueError(f"{name} row {start + 2}: the first seat is not the focal one")
                raw_group = str(_whole(first[0], f"{name} participant number"))
                if raw_group in first_row:
                    raise ValueError(f"{name} repeats participant {raw_group}")
                first_row[raw_group] = start + 2
            sealed = held_out(game.cell, first_row)
            counts[game.cell] = len(sealed)
            for raw_group, row in first_row.items():
                if raw_group not in sealed:
                    rounds = _by_seat(game, cells(sheet, (row, row + 1), ROUND_COLUMNS))
                    groups.append(Group(game.cell, group_index(game.cell, raw_group), rounds))
    finally:
        book.close()
    return groups, counts


def fours_table(path: Path) -> pd.DataFrame:
    """Decode the four-player file, a MATLAB table, with the optional `mat-io` package."""
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


def _training_fours(path: Path) -> tuple[list[Group], dict[str, int]]:
    """Read the first-session training groups of four, and count the held-out ones.

    The identifier columns decide which rows are wanted. Only those rows' contributions are
    taken out of the table.
    """
    frame = fours_table(path)
    missing = [column for column in (*FOUR_KEYS, FOUR_VALUE) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks the columns {missing}")
    keys = cast(list[list[object]], frame[list(FOUR_KEYS)].astype(object).to_numpy().tolist())
    places: dict[tuple[str, str], dict[tuple[int, int], int]] = {}
    for position, (treatment, session, group, player, round_) in enumerate(keys):
        if _whole(_scalar(session), "a session") != 1:
            continue
        key = (f"4-{_scalar(treatment)}", str(_whole(_scalar(group), "a group number")))
        if key[0] not in GAMES:
            raise ValueError(f"{path.name} has an unknown treatment")
        place = (_whole(_scalar(player), "a player number"), _whole(_scalar(round_), "a round"))
        if place in places.setdefault(key, {}):
            raise ValueError(f"{path.name} repeats a seat and round of a group")
        places[key][place] = position

    groups: list[Group] = []
    counts: dict[str, int] = {}
    for cell in sorted({cell for cell, _ in places}):
        game = GAMES[cell]
        sealed = held_out(cell, [group for game_cell, group in places if game_cell == cell])
        counts[cell] = len(sealed)
        wanted = [(s, r) for s in range(1, game.size + 1) for r in range(1, ROUNDS + 1)]
        for (game_cell, raw_group), place in places.items():
            if game_cell != cell or raw_group in sealed:
                continue
            if set(place) != set(wanted):
                raise RecordFault(f"a group of {cell} is not {game.size} seats by {ROUNDS} rounds")
            taken = cast(
                list[object],
                frame[FOUR_VALUE].iloc[[place[w] for w in wanted]].astype(object).tolist(),
            )
            seats = [taken[s * ROUNDS : (s + 1) * ROUNDS] for s in range(game.size)]
            groups.append(Group(cell, group_index(cell, raw_group), _by_seat(game, seats)))
    return groups, counts


def _training(raw_dir: Path) -> tuple[list[Group], dict[str, int]]:
    """Check the files, then return the training groups and the held-out counts."""
    verify(raw_dir)
    pairs, pair_counts = _training_pairs(raw_dir / PAIRS_FILE)
    fours, four_counts = _training_fours(raw_dir / FOURS_FILE)
    counts = {**pair_counts, **four_counts}
    if set(counts) != set(GAMES):
        raise ValueError(f"the files hold the games {sorted(counts)}, not {sorted(GAMES)}")
    groups = sorted([*pairs, *fours], key=lambda group: (group.cell, group.index))
    return groups, dict(sorted(counts.items()))


def load(raw_dir: Path = RAW_DIR) -> list[Group]:
    """Return the training groups of all eight games, in a fixed order."""
    return _training(raw_dir)[0]


def held_out_counts(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Return how many groups of each game are held out. Nothing else is known about them."""
    return _training(raw_dir)[1]


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
