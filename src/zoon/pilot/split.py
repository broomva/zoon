"""Deterministic assignment of participants to the training or the held-out set.

The assignment depends only on a committed salt, the experiment number and the participant's
raw identifier. It does not depend on the order of the rows or on who else is in the file, so
adding participants never moves an existing one.

The held-out set is sealed. Nothing in this package returns a held-out row; the loader in
`fetch` counts held-out participants and skips them.
"""

import hashlib
from typing import Literal

SALT = "zoon-pilot-h1-split-v1"
HELD_OUT_FRACTION = 0.30

Part = Literal["train", "held_out"]


def _digest(purpose: str, experiment: int, raw_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{purpose}|{experiment}|{raw_id}".encode()).hexdigest()


def assign(experiment: int, raw_id: str) -> Part:
    """Return the part a participant belongs to. About 30% of each experiment is held out."""
    fraction = int(_digest("split", experiment, raw_id)[:12], 16) / 16**12
    return "held_out" if fraction < HELD_OUT_FRACTION else "train"


def participant_index(experiment: int, raw_id: str) -> str:
    """Return the opaque index that stands for a participant outside the loader.

    The index is a salted hash. It keeps raw identifiers out of tables, logs and results. It
    is not anonymisation: the dataset is public and so is the salt.
    """
    return _digest("index", experiment, raw_id)[:16]
