"""The pre-registered calibration draw and rotation-specific examples, from training only."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd

from zoon.pilot import fetch, power
from zoon.pilot.materials import Answer

FILE = fetch.REPO / "prereg" / "cohorts.json"
CALIBRATION_SIZE = 300


@dataclass(frozen=True)
class Trial:
    dilemma: str
    answer: Answer
    success: bool
    rating: int


@dataclass(frozen=True)
class Participant:
    experiment: int
    participant: str
    trials: tuple[Trial, ...]


class Identity(TypedDict):
    experiment: int
    participant: str


class Example(Identity):
    last: int


class Rotation(TypedDict):
    held_out: list[str]
    examples: list[Example]


class Cohorts(TypedDict):
    calibration: list[Identity]
    rotations: list[Rotation]


def participants(table: pd.DataFrame) -> list[Participant]:
    """Convert a training trial table to ordered records with integer, original-scale ratings."""
    result: list[Participant] = []
    for (experiment, index), rows in table.sort_values(
        ["experiment", "participant", "trial"]
    ).groupby(["experiment", "participant"], sort=True):
        if list(rows["trial"]) != list(range(1, len(fetch.DILEMMAS) + 1)):
            raise fetch.RecordFault("participant must have a complete ordered record")
        if sorted(rows["dilemma"]) != sorted(fetch.DILEMMAS):
            raise fetch.RecordFault("participant must see each dilemma once")
        trials: list[Trial] = []
        for dilemma, action, success, valence in zip(
            rows["dilemma"], rows["action_choice"], rows["success"], rows["valence"], strict=True
        ):
            try:
                rating = float(valence) * 100
            except (TypeError, ValueError):
                raise fetch.RecordFault("unreadable rating") from None
            if not math.isfinite(rating) or not -100 <= rating <= 100:
                raise fetch.RecordFault("rating must be between -100 and 100")
            rounded = round(rating)
            if not math.isclose(rating, rounded, abs_tol=1e-8):
                raise fetch.RecordFault("rating must be an integer on the original scale")
            if action not in (0, 1) or success not in (0, 1):
                raise fetch.RecordFault("answer and success must be binary")
            trials.append(Trial(str(dilemma), "Yes" if action else "No", bool(success), rounded))
        result.append(Participant(int(str(experiment)), str(index), tuple(trials)))
    return result


def load_training() -> list[Participant]:
    return participants(fetch.load())


def identity(person: Participant) -> Identity:
    return {"experiment": person.experiment, "participant": person.participant}


def generate(people: Sequence[Participant], calibration_size: int = CALIBRATION_SIZE) -> Cohorts:
    """Execute the registered PCG64 draw, with no random calls beyond its pseudocode."""
    ids = sorted(people, key=lambda person: (person.experiment, person.participant))
    if len({(p.experiment, p.participant) for p in ids}) != len(ids):
        raise ValueError("participant identities must be unique")
    if not 0 < calibration_size <= len(ids) - 2:
        raise ValueError("insufficient participants for calibration and examples")
    rng = np.random.Generator(np.random.PCG64(20261115))
    order = rng.permutation(len(ids))
    calibration = [ids[int(i)] for i in order[:calibration_size]]
    rest = [ids[int(i)] for i in order[calibration_size:]]
    rotations: list[Rotation] = []
    for pair in power.held_out_sets(True, len(fetch.DILEMMAS)):
        held_out = [fetch.DILEMMAS[d] for d in pair]
        first = [next(i for i, t in enumerate(p.trials, 1) if t.dilemma in held_out) for p in rest]
        eligible = [(p, f) for p, f in zip(rest, first, strict=True) if f >= 3]
        if len(eligible) < 2:
            raise ValueError("rotation needs two eligible example participants")
        a, b = rng.choice(len(eligible), size=2, replace=False)
        examples: list[Example] = []
        for index in (a, b):
            person, position = eligible[int(index)]
            last = int(rng.integers(2, position))
            examples.append({**identity(person), "last": last})
        rotations.append({"held_out": held_out, "examples": examples})
    return {"calibration": [identity(p) for p in calibration], "rotations": rotations}


def serialise(document: Cohorts) -> bytes:
    return (json.dumps(document, indent=2) + "\n").encode()


def load(path: Path = FILE) -> Cohorts:
    return cast(Cohorts, json.loads(path.read_text()))


if __name__ == "__main__":
    document = generate(load_training())
    FILE.write_bytes(serialise(document))
    print(f"[cohorts] calibration participants = {len(document['calibration'])}")
    print(f"[cohorts] rotations = {len(document['rotations'])}")
