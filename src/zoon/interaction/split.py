"""Deterministic assignment of whole groups to the training or the held-out set.

A group is the unit: the people in it played together, so their records are not independent
and a group is never divided.
Within each game the groups are ranked by a salted hash of their raw identifier and the first
two fifths are held out. The rank depends on which groups are in the file, and the file is
pinned by its hash in the manifest, so the assignment cannot move.

Training loaders keep the held-out groups sealed.
"""

import hashlib
from collections.abc import Iterable

SALT = "zoon-interaction-split-v1"
HELD_OUT_NUMERATOR = 2
HELD_OUT_DENOMINATOR = 5


def _digest(purpose: str, cell: str, raw_group: str) -> str:
    return hashlib.sha256(f"{SALT}|{purpose}|{cell}|{raw_group}".encode()).hexdigest()


def held_out(cell: str, raw_groups: Iterable[str]) -> frozenset[str]:
    """Return the raw identifiers of the held-out groups of one game: two fifths, rounded up."""
    groups = sorted(set(raw_groups), key=lambda group: _digest("split", cell, group))
    count = -(-len(groups) * HELD_OUT_NUMERATOR // HELD_OUT_DENOMINATOR)
    return frozenset(groups[:count])


def group_index(cell: str, raw_group: str) -> str:
    """Return the opaque index that stands for a group outside the loader.

    The index is a salted hash. It keeps raw identifiers out of tables, logs and results. It
    is not anonymisation: the dataset is public and so is the salt.
    """
    return _digest("index", cell, raw_group)[:16]
