"""Synthetic raw files shaped like the published public goods data. No real record is used."""

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from zoon.interaction import data
from zoon.interaction.games import GAMES, ROUNDS

# Text where a number would be. It fills every cell the loader must not use: the second
# session, the rounds after the twentieth and the questionnaire columns. Turning any of them
# into a number raises.
SENTINEL = "not for reading"
UNREADABLE = "unreadable"
SESSION_ONE_ROUNDS = 22

Key = tuple[str, str]
Intended = dict[Key, tuple[tuple[int, ...], ...]]


def _rounds(rng: random.Random, cell: str) -> tuple[tuple[int, ...], ...]:
    endowments = GAMES[cell].endowments
    return tuple(tuple(rng.randint(0, e) for e in endowments) for _ in range(ROUNDS))


def write_raw(
    raw_dir: Path,
    groups_per_game: int | Mapping[str, int],
    unreadable: frozenset[Key] = frozenset(),
) -> Intended:
    """Write both files and return the first-session contributions they were built from.

    The contribution cells of an `unreadable` group hold text.
    """
    counts = (
        {c: groups_per_game for c in GAMES} if isinstance(groups_per_game, int) else groups_per_game
    )
    rng = random.Random(7)
    intended: Intended = {}
    raw_dir.mkdir(parents=True, exist_ok=True)
    columns = ["NO.", "Participant", *(f"R_{r}" for r in range(1, SESSION_ONE_ROUNDS + 1)), "CC_0"]
    writer: Any = cast(Any, pd).ExcelWriter(raw_dir / data.PAIRS_FILE)
    with writer:
        for treatment, sheet in data.PAIR_SHEETS.items():
            cell = f"2-{treatment}"
            count = counts[cell]
            rows: list[list[object]] = []
            for number in range(1, count + 1):
                key = (cell, str(number))
                rounds = intended[key] = _rounds(rng, cell)
                for seat, label in enumerate((data.FOCAL_LABEL, f"Type2_{number:02d}")):
                    cells: list[object] = [
                        UNREADABLE if key in unreadable else row[seat] for row in rounds
                    ]
                    extra = [SENTINEL] * (SESSION_ONE_ROUNDS - ROUNDS + 1)
                    rows.append([number if seat == 0 else None, label, *cells, *extra])
                for label in (f"Type2_{number % count + 1:02d}", data.FOCAL_LABEL):
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
        count = counts[cell]
        for number in range(1, 2 * count + 1):
            session = 1 if number <= count else 2
            key = (cell, str(number))
            rounds = _rounds(rng, cell)
            if session == 1:
                intended[key] = rounds
            for seat in range(4):
                for round_ in range(ROUNDS):
                    value: object = float(rounds[round_][seat]) if session == 1 else SENTINEL
                    table.append(
                        {
                            "Treatment": treatment,
                            "Session": float(session),
                            "GroupID": float(number),
                            "PlayerID": float(seat + 1),
                            "GlobalPlayerID": float(len(table)),
                            "Round": float(round_ + 1),
                            "Contribution": UNREADABLE if key in unreadable else value,
                        }
                    )
    rng.shuffle(table)
    pd.DataFrame(table).to_pickle(raw_dir / data.FOURS_FILE)
    return intended


def first_row(number: int) -> int:
    """The worksheet row, counted from 1, where the block of pair `number` starts."""
    return 2 + 4 * (number - 1)


class FakeMatio:
    """Stands in for the MATLAB reader library: the synthetic four-player file is a pickled
    table, returned the way the library returns the real one."""

    @staticmethod
    def load_from_mat(path: str) -> dict[str, pd.DataFrame]:
        return {"all_data": pd.read_pickle(path)}
