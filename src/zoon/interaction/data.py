"""Fetch the public goods data, write the training file, and load it.

Data: Wang, Hilbe and Zhang, "Dataset and computer code for 'The dynamics of cooperation in
asymmetric public goods games'", https://zenodo.org/records/16918146, CC BY 4.0.
Paper: https://doi.org/10.1073/pnas.2525760123

Three steps, and one rule about which code may do which:

1. `fetch` downloads the two raw files into `data/groups/raw/` and checks each against
   `interaction/data-manifest.json`. The raw files are never committed.
2. `seal` is the only code that opens the raw files. It writes `data/groups/training.json`,
   which holds the training groups and, for the rest, a count per game. The hash of that
   file is pinned in the manifest too.
3. `load` reads the training file and nothing else. All analysis goes through `load`.

So the contributions of the held-out groups are absent from the only file the analysis
reads. That is the guarantee, and it is a guard against accident. It is not a defence
against the person running the code: the raw files are public, and anyone can open them with
this module's private functions or with other code. Nothing here claims otherwise.

What `seal` uses from the raw files, and what it does not:

- Only the linear game, and only its first session. In the second session the same people
  play again in new groups, and the four-player file does not say who is who across
  sessions, so a second-session group cannot be kept apart from the first-session groups
  its members came from.
- Only the first 20 rounds, the ones the authors analysed. The two-player sessions ran for
  22 rounds.
- The threshold game, the second session and the questionnaire answers are never used.
- For the two-player spreadsheet it asks only for the identifier columns and the 20 round
  cells of the two first-session rows of each training pair.
- The four-player file is a MATLAB table that its reader library decodes whole. `seal` keeps
  the first-session rows of training groups and turns nothing else into a number.

Each raw file is checked against the manifest by the function that opens it, so the split
cannot move with a changed file.
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
TRAINING = REPO / "data" / "groups" / "training.json"
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
ID_COLUMNS = (1, 2)
ROUND_COLUMNS = (3, 2 + ROUNDS)

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


def _manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {"dataset": RECORD, "licence": "CC BY 4.0", "files": {}}
    return cast(dict[str, Any], json.loads(MANIFEST.read_text()))


def _record(document: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def fetch(raw_dir: Path = RAW_DIR) -> None:
    """Download any missing raw file, then check every file against the manifest.

    A file with no manifest entry is recorded (URL, size, sha256). A file whose hash differs
    from its entry raises, so a changed upstream file cannot pass unnoticed.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    document = _manifest()
    for filename in FILES:
        path = raw_dir / filename
        if not path.exists():
            with urllib.request.urlopen(_url(filename)) as response:
                path.write_bytes(response.read())
        found = {"url": _url(filename), "size": path.stat().st_size, "sha256": _sha256(path)}
        if document["files"].setdefault(filename, found) != found:
            raise ValueError(f"{filename} does not match {MANIFEST.name}")
    _record(document)


def _check(path: Path) -> None:
    """Refuse a raw file that is missing from the manifest or differs from it."""
    entry = _manifest()["files"].get(path.name)
    if entry is None or entry["sha256"] != _sha256(path):
        raise ValueError(f"{path.name} does not match {MANIFEST.name}")


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


def _pairs(path: Path) -> tuple[list[Group], dict[str, int]]:
    """Read the first-session training pairs, and count the held-out ones."""
    _check(path)
    openpyxl: Any = importlib.import_module("openpyxl")
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    groups: list[Group] = []
    counts: dict[str, int] = {}

    def cells(sheet: Any, rows: tuple[int, int], columns: tuple[int, int]) -> list[list[object]]:
        """Ask the worksheet for one rectangle, rows and columns counted from 1."""
        found = sheet.iter_rows(
            min_row=rows[0],
            max_row=rows[1],
            min_col=columns[0],
            max_col=columns[1],
            values_only=True,
        )
        return [list(row) for row in found]

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


def _scalar(cell: object) -> object:
    """Unwrap the one-element arrays MATLAB text cells arrive in."""
    value: object = cast(Any, cell).tolist() if hasattr(cell, "tolist") else cell
    while isinstance(value, list):
        items = cast(list[object], value)
        if len(items) != 1:
            break
        value = items[0]
    return cast(object, value)


def _fours(path: Path) -> tuple[list[Group], dict[str, int]]:
    """Read the first-session training groups of four, and count the held-out ones.

    The file is a MATLAB table, decoded by the optional `mat-io` package. The decoded table
    never leaves this function: the identifier columns decide which rows are wanted, and only
    those rows' contributions are taken out of it.
    """
    _check(path)
    matio: Any = importlib.import_module("matio")
    frame = cast(pd.DataFrame, matio.load_from_mat(str(path))["all_data"])
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


def seal(raw_dir: Path = RAW_DIR, training: Path = TRAINING) -> None:
    """Write the training file from the raw files, and pin or check its hash.

    The file holds every training group and, for the held-out groups, one count per game.
    Writing it twice gives the same bytes. A file whose hash differs from the manifest's
    entry raises.
    """
    pairs, pair_counts = _pairs(raw_dir / PAIRS_FILE)
    fours, four_counts = _fours(raw_dir / FOURS_FILE)
    counts = dict(sorted({**pair_counts, **four_counts}.items()))
    if set(counts) != set(GAMES):
        raise ValueError(f"the files hold the games {sorted(counts)}, not {sorted(GAMES)}")
    groups = sorted([*pairs, *fours], key=lambda group: (group.cell, group.index))
    document = {
        "dataset": RECORD,
        "groups": [{"cell": g.cell, "index": g.index, "rounds": g.rounds} for g in groups],
        "held_out": counts,
    }
    training.parent.mkdir(parents=True, exist_ok=True)
    training.write_text(json.dumps(document, sort_keys=True) + "\n")
    manifest = _manifest()
    found = {"sha256": _sha256(training), "groups": len(groups)}
    if manifest.setdefault("training", found) != found:
        raise ValueError(f"{training.name} does not match {MANIFEST.name}")
    _record(manifest)


def _read(training: Path) -> dict[str, Any]:
    """Return the training file's contents after checking its hash against the manifest."""
    entry = _manifest().get("training")
    if entry is None or entry["sha256"] != _sha256(training):
        raise ValueError(f"{training.name} does not match {MANIFEST.name}")
    return cast(dict[str, Any], json.loads(training.read_text()))


def load(training: Path = TRAINING) -> list[Group]:
    """Return the training groups of all eight games, in a fixed order."""
    groups: list[Group] = []
    for record in _read(training)["groups"]:
        game = GAMES[record["cell"]]
        seats = [[row[seat] for row in record["rounds"]] for seat in range(game.size)]
        groups.append(Group(game.cell, str(record["index"]), _by_seat(game, seats)))
    return groups


def held_out_counts(training: Path = TRAINING) -> dict[str, int]:
    """Return how many groups of each game are held out. Nothing else is known about them."""
    return {str(cell): int(count) for cell, count in _read(training)["held_out"].items()}


def main() -> None:
    """Fetch the raw files, write the training file, and print how many groups each game
    has in each part."""
    fetch()
    seal()
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
