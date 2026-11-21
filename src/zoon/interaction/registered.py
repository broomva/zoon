"""Committed primary laws and training amounts; no registered run refits them."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from zoon.interaction import data, kernel, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES

FITTED = data.REPO / "interaction" / "fitted.json"
AMOUNTS = data.REPO / "interaction" / "amounts.json"
FORMS = {"zoon": person.Candidate("B"), "alone": person.Candidate("B", twin=True)}
PROBABILITY_TOLERANCE = 1e-3


def encoded(document: Mapping[str, Any]) -> bytes:
    """Full Python float precision, sorted keys, portable bytes and final newline."""
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sealed(document: dict[str, Any]) -> bytes:
    return encoded({**document, "sha256": hashlib.sha256(encoded(document)).hexdigest()})


def training_digest(manifest: Path = data.MANIFEST) -> str:
    document = json.loads(manifest.read_bytes())
    return str(document["training"]["sha256"])


def read_document(path: Path, manifest: Path = data.MANIFEST) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        document = cast(dict[str, Any], json.loads(raw))
        digest = document.pop("sha256")
        if digest != hashlib.sha256(encoded(document)).hexdigest() or raw != sealed(document):
            raise ValueError("material checksum differs from file bytes")
        if document["training_sha256"] != training_digest(manifest):
            raise ValueError("material training digest differs from manifest")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid material document") from error
    return document


def require_keys(document: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict) or set(cast(dict[str, Any], document)) != keys:
        raise ValueError("invalid material keys")
    return cast(dict[str, Any], document)


def array(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("invalid material array")
    return cast(list[Any], value)


def number(value: Any) -> bool:
    return type(value) in (int, float) and np.isfinite(value)


def fit(groups: Sequence[Group]) -> dict[str, person.Person]:
    """Both default-start fits, on exactly the supplied order of training groups."""
    return {arm: person.fit(groups, form) for arm, form in FORMS.items()}


def fitted_bytes(laws: Mapping[str, person.Person], digest: str) -> bytes:
    return sealed(
        {
            "training_sha256": digest,
            "arms": {
                arm: {
                    "letter": law.candidate.letter,
                    "twin": law.candidate.twin,
                    "count": law.candidate.count,
                    "numbers": law.numbers.tolist(),
                }
                for arm, law in laws.items()
            },
        }
    )


def amounts_bytes(groups: Sequence[Group], digest: str) -> bytes:
    return sealed(
        {
            "training_sha256": digest,
            "rows": [
                [e, b, values] for (e, b), values in kernel.Amounts.build(groups).rows.items()
            ],
        }
    )


def read_fitted(path: Path = FITTED, manifest: Path = data.MANIFEST) -> dict[str, person.Person]:
    document = read_document(path, manifest)
    require_keys(document, {"training_sha256", "arms"})
    arms = document["arms"]
    if not isinstance(arms, dict) or set(cast(dict[str, Any], arms)) != set(FORMS):
        raise ValueError("wrong primary arms")
    arms = cast(dict[str, Any], arms)
    result: dict[str, person.Person] = {}
    for arm, form in FORMS.items():
        row = require_keys(arms[arm], {"letter", "twin", "count", "numbers"})
        if (
            (row["letter"], row["twin"], row["count"]) != (form.letter, form.twin, form.count)
            or type(row["twin"]) is not bool
            or type(row["count"]) is not int
        ):
            raise ValueError("wrong primary form or count")
        numbers = array(row["numbers"])
        if len(numbers) != form.count or not all(map(number, numbers)):
            raise ValueError("invalid material fitted numbers")
        if any(not 0 <= v <= 1 for v in numbers[-len(form.traces) :]):
            raise ValueError("invalid material update fraction")
        result[arm] = person.Person(form, np.array(numbers, dtype=float))
    return result


def read_amounts(path: Path = AMOUNTS, manifest: Path = data.MANIFEST) -> kernel.Amounts:
    document = read_document(path, manifest)
    require_keys(document, {"training_sha256", "rows"})
    rows = array(document["rows"])
    expected = sorted(
        {(e, b) for game in GAMES.values() for e in game.endowments for b in (1, 2, 3)}
    )
    keys: list[tuple[int, int]] = []
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    for raw_row in rows:
        row = array(raw_row)
        if len(row) != 3:
            raise ValueError("invalid material amount row")
        e, b, raw_values = row
        values = array(raw_values)
        if (
            type(e) is not int
            or type(b) is not int
            or not values
            or any(type(v) is not int for v in values)
            or values != sorted(values)
        ):
            raise ValueError("invalid material amount row")
        keys.append((e, b))
        result[e, b] = tuple(values)
    if keys != expected:
        raise ValueError("material amount rows must be complete, unique and sorted")
    return kernel.Amounts(result)


def write_materials(groups: Sequence[Group], digest: str) -> None:
    laws = fit(groups)
    FITTED.write_bytes(fitted_bytes(laws, digest))
    AMOUNTS.write_bytes(amounts_bytes(groups, digest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materials",))
    parser.parse_args()
    groups = data.load()
    digest = hashlib.sha256(data.TRAINING.read_bytes()).hexdigest()
    if digest != training_digest():
        raise ValueError("training file differs from manifest")
    write_materials(groups, digest)
    print("wrote committed primary fits and training amounts")


if __name__ == "__main__":
    main()
