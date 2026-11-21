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
    document = cast(dict[str, Any], json.loads(path.read_bytes()))
    digest = document.pop("sha256")
    if digest != hashlib.sha256(encoded(document)).hexdigest():
        raise ValueError("material checksum differs")
    if document["training_sha256"] != training_digest(manifest):
        raise ValueError("material training digest differs from manifest")
    return document


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
    arms = read_document(path, manifest)["arms"]
    if set(arms) != set(FORMS):
        raise ValueError("wrong primary arms")
    result: dict[str, person.Person] = {}
    for arm, form in FORMS.items():
        row = arms[arm]
        if (row["letter"], row["twin"], row["count"]) != (form.letter, form.twin, form.count):
            raise ValueError("wrong primary form or count")
        result[arm] = person.Person(form, np.array(row["numbers"], dtype=float))
    return result


def read_amounts(path: Path = AMOUNTS, manifest: Path = data.MANIFEST) -> kernel.Amounts:
    rows = read_document(path, manifest)["rows"]
    return kernel.Amounts({(e, b): tuple(values) for e, b, values in rows})


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
