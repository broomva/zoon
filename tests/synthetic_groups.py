"""Synthetic raw files shaped like the published public goods data. No real record is used."""

import random
from pathlib import Path
from typing import Any, cast

import pandas as pd

from zoon.interaction import data
from zoon.interaction.games import GAMES, ROUNDS

# A value no seat may contribute. It fills every cell the loader must not read: the second
# session, the rounds after the twentieth and the questionnaire columns.
SENTINEL = 999
SESSION_ONE_ROUNDS = 22

Key = tuple[str, str]
Intended = dict[Key, tuple[tuple[int, ...], ...]]


def _rounds(rng: random.Random, cell: str) -> tuple[tuple[int, ...], ...]:
    endowments = GAMES[cell].endowments
    return tuple(tuple(rng.randint(0, e) for e in endowments) for _ in range(ROUNDS))


def write_raw(
    raw_dir: Path, groups_per_game: int, unreadable: frozenset[Key] = frozenset()
) -> Intended:
    """Write both files and return the first-session contributions they were built from.

    The contribution cells of an `unreadable` group hold text.
    """
    rng = random.Random(7)
    intended: Intended = {}
    raw_dir.mkdir(parents=True, exist_ok=True)
    columns = ["NO.", "Participant", *(f"R_{r}" for r in range(1, SESSION_ONE_ROUNDS + 1)), "CC_0"]
    writer: Any = cast(Any, pd).ExcelWriter(raw_dir / data.PAIRS_FILE)
    with writer:
        for treatment, sheet in data.PAIR_SHEETS.items():
            cell = f"2-{treatment}"
            rows: list[list[object]] = []
            for number in range(1, groups_per_game + 1):
                key = (cell, str(number))
                rounds = intended[key] = _rounds(rng, cell)
                for seat, label in enumerate((data.FOCAL_LABEL, f"Type2_{number:02d}")):
                    cells: list[object] = [
                        "unreadable" if key in unreadable else row[seat] for row in rounds
                    ]
                    extra = [SENTINEL] * (SESSION_ONE_ROUNDS - ROUNDS + 1)
                    rows.append([number if seat == 0 else None, label, *cells, *extra])
                for label in (f"Type2_{number % groups_per_game + 1:02d}", data.FOCAL_LABEL):
                    rows.append([None, label, *[SENTINEL] * (SESSION_ONE_ROUNDS + 1)])
            cast(Any, pd.DataFrame(rows, columns=columns)).to_excel(
                writer, sheet_name=sheet, index=False
            )
            decoy = pd.DataFrame(
                [[1, "me", *[SENTINEL] * (SESSION_ONE_ROUNDS + 1)]], columns=columns
            )
            cast(Any, decoy).to_excel(
                writer, sheet_name=sheet.replace("Type1", "Type2"), index=False
            )

    table: list[dict[str, object]] = []
    for cell in ("4-FE", "4-AI", "4-MI"):
        treatment = cell.removeprefix("4-")
        for number in range(1, 2 * groups_per_game + 1):
            session = 1 if number <= groups_per_game else 2
            key = (cell, str(number))
            rounds = _rounds(rng, cell)
            if session == 1:
                intended[key] = rounds
            for seat in range(4):
                for round_ in range(ROUNDS):
                    value: object = rounds[round_][seat] if session == 1 else SENTINEL
                    table.append(
                        {
                            "Treatment": treatment,
                            "Session": float(session),
                            "GroupID": float(number),
                            "PlayerID": float(seat + 1),
                            "GlobalPlayerID": float(len(table)),
                            "Round": float(round_ + 1),
                            "Contribution": "unreadable" if key in unreadable else float(value),
                        }
                    )
    rng.shuffle(table)
    pd.DataFrame(table).to_pickle(raw_dir / data.FOURS_FILE)
    return intended


def read_synthetic_fours(path: Path) -> pd.DataFrame:
    """Stand in for the MATLAB reader: the synthetic four-player file is a pickled table."""
    return pd.read_pickle(path)
