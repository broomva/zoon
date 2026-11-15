"""Published human judgments and feature codes for the dilemmas this task was adapted from.

Source: Körner and Deutsch, "Deontology and Utilitarianism in Real Life: A Set of Moral
Dilemmas Based on Historic Events", Personality and Social Psychology Bulletin 49(10),
https://doi.org/10.1177/01461672221103058, CC BY 4.0. Data at https://osf.io/cg5tq/.

Two things come from that source and they are kept apart:

- `korner_deutsch_2022.csv`, committed, is transcribed from the paper's tables: the feature
  codes of its 38 factual dilemmas and the mean appropriateness judgment of every dilemma in
  both studies. `task_dilemmas.csv`, committed, maps this task's 13 dilemmas to their
  originals and holds the task authors' own codes.
- The participant-level files are never committed. The OSF project states no licence, so
  `fetch` downloads them into `data/outside/` and checks each against
  `data-manifest-outside.json`. `ratings` reduces them to a count and a mean per dilemma.

Those files hold a subject number, date, time, age, gender, native language and free-text
comments. `ratings` reads a cell only if its column holds the appropriateness judgment of a
dilemma named in the committed table, and refuses any other column.

Nothing here touches the task's own participants.
"""

import csv
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zoon.pilot.fetch import CBR_IS_ACTION, DILEMMAS, REPO
from zoon.pilot.models import Floats

RAW_DIR = REPO / "data" / "outside"
MANIFEST = REPO / "data-manifest-outside.json"
TABLE = Path(__file__).with_name("korner_deutsch_2022.csv")
TASK_TABLE = Path(__file__).with_name("task_dilemmas.csv")

SCALE = (1, 7)  # appropriateness of the suggested action, from not at all to completely
MIDPOINT = 4.0
# The paper prints means to two decimals, so a mean computed from the raw files may differ
# from the printed one by half a unit in the second decimal and no more.
ROUNDING = 0.005

# The only columns that may be read from a participant-level file: one dilemma's judgment.
JUDGMENT_COLUMN = re.compile(r"(Classic|New)_[A-Za-z0-9]+\.dilemma(Classic|New)[123]?")


@dataclass(frozen=True)
class Study:
    """One of the two samples that rated the original dilemmas."""

    label: str
    filename: str
    url: str
    delimiter: str


STUDIES: dict[int, Study] = {
    1: Study("German", "kd_study1_dilAll.csv", "https://osf.io/download/72jta/", ";"),
    2: Study("US", "kd_study2_dilEng.csv", "https://osf.io/download/cye64/", ";"),
}

# How a published answer becomes a number. Hedged answers sit between yes and no; an answer
# that is neither ("?", "Unclear", a description) sits in the middle.
GRADES: dict[str, float] = {
    "yes": 1.0,
    "rather yes": 0.75,
    "maybe": 0.5,
    "somewhat": 0.5,
    "unclear": 0.5,
    "?": 0.5,
    "state and its citizens": 0.5,
    "probably no": 0.25,
    "not likely": 0.25,
    "not really": 0.25,
    "no": 0.0,
}
# The original paper files a dilemma in which animals die (Lion Hunt) under "other norms", so
# the task authors' "Animals" is coded as not involving killing.
KILLING: dict[str, float] = {"yes": 1.0, "no": 0.0, "animals": 0.0}
CONTEXTS: tuple[str, ...] = (
    "Medicine",
    "Accident",
    "War",
    "Terrorism",
    "Crime",
    "Humanitarian Aid",
    "State authority",
    "Justice",
    "Animal rights",
)
FEATURES: tuple[str, ...] = (
    "involves killing",
    "action is the utilitarian option",
    "victims a subset of the benefited",
    "self in the benefited group",
    "opposite framing reasonable",
    *(f"context: {context}" for context in CONTEXTS),
)


@dataclass(frozen=True)
class Codes:
    """The published description of one dilemma, as the answers printed in the source."""

    context: str
    involves_killing: str
    action_is_utilitarian: str
    subset_of_benefited: str
    self_in_benefited: str
    opposite_framing: str


@dataclass(frozen=True)
class Dilemma:
    """One row of the committed table."""

    name: str
    kind: str  # "factual" or "trolley"
    codes: Codes | None  # published for the factual dilemmas only
    published: dict[int, float]  # study -> the mean judgment printed in the paper
    columns: dict[int, tuple[str, ...]]  # study -> where the judgment sits in the raw file


@dataclass(frozen=True)
class Rating:
    """The judgments of one dilemma in one study, reduced to what the analysis needs."""

    count: int
    mean: float


Rated = dict[int, dict[str, Rating]]  # study -> dilemma -> judgments


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def published(table: Path = TABLE) -> dict[str, Dilemma]:
    """Return the committed table of the original dilemmas, by name."""
    dilemmas: dict[str, Dilemma] = {}
    for row in _rows(table):
        section = row["section"]
        codes = None
        if row["kind"] == "factual":
            codes = Codes(
                context=row["context"],
                # Tables 1 and 2 split the dilemmas by these two properties and print them as
                # section headings.
                involves_killing="Yes" if section in ("killing", "saving") else "No",
                action_is_utilitarian=(
                    "Yes" if section in ("killing", "other proscriptive") else "No"
                ),
                subset_of_benefited=row["subset_of_benefited"],
                self_in_benefited=row["self_in_benefited"],
                opposite_framing=row["opposite_framing"],
            )
        studies = [study for study in STUDIES if row[f"study{study}_columns"]]
        dilemmas[row["name"]] = Dilemma(
            name=row["name"],
            kind=row["kind"],
            codes=codes,
            published={study: float(row[f"study{study}_judgment"]) for study in studies},
            columns={study: tuple(row[f"study{study}_columns"].split("|")) for study in studies},
        )
    return dilemmas


def originals(table: Path = TASK_TABLE) -> dict[str, str]:
    """Return, for each of the task's dilemmas, the name of the dilemma it was adapted from."""
    mapping = {row["dilemma"]: row["original"] for row in _rows(table)}
    if tuple(mapping) != DILEMMAS:
        raise ValueError(f"{table.name} must list the task's dilemmas in order: {DILEMMAS}")
    return mapping


def task_codes(table: Path = TASK_TABLE, source: Path = TABLE) -> dict[str, Codes]:
    """Return the task authors' own codes of their 13 adapted texts.

    Their table has no column for whether the self is in the benefited group, so that one
    answer is the original paper's.
    """
    original = published(source)
    codes: dict[str, Codes] = {}
    for row in _rows(table):
        theirs = original[row["original"]].codes
        assert theirs is not None
        codes[row["dilemma"]] = Codes(
            context=row["context"],
            involves_killing=row["involves_killing"],
            action_is_utilitarian=row["action_is_utilitarian"],
            subset_of_benefited=row["subset_of_benefited"],
            self_in_benefited=theirs.self_in_benefited,
            opposite_framing=row["opposite_framing"],
        )
    return codes


def original_codes(table: Path = TASK_TABLE, source: Path = TABLE) -> dict[str, Codes]:
    """Return the original paper's codes of the dilemma each of the task's 13 was adapted
    from, with one answer replaced: whether the suggested action is the utilitarian option
    is the task's, because the task may ask about the opposite action and that answer has to
    describe the action being predicted."""
    original = published(source)
    codes: dict[str, Codes] = {}
    for dilemma, name in originals(table).items():
        theirs = original[name].codes
        assert theirs is not None
        codes[dilemma] = Codes(
            context=theirs.context,
            involves_killing=theirs.involves_killing,
            action_is_utilitarian="Yes" if CBR_IS_ACTION[dilemma] else "No",
            subset_of_benefited=theirs.subset_of_benefited,
            self_in_benefited=theirs.self_in_benefited,
            opposite_framing=theirs.opposite_framing,
        )
    return codes


def action_reversed(table: Path = TASK_TABLE, source: Path = TABLE) -> dict[str, bool]:
    """Return, for each of the task's dilemmas, whether the adaptation asks about the opposite
    action: the original's suggested action was one kind of option and the task's is the
    other. The task's side is checked against the labels the analysis uses."""
    original = published(source)
    reversed_: dict[str, bool] = {}
    for dilemma, codes in task_codes(table, source).items():
        theirs = original[originals(table)[dilemma]].codes
        assert theirs is not None
        if (codes.action_is_utilitarian == "Yes") != CBR_IS_ACTION[dilemma]:
            raise ValueError(f"{dilemma}: the task table and CBR_IS_ACTION disagree")
        reversed_[dilemma] = codes.action_is_utilitarian != theirs.action_is_utilitarian
    return reversed_


def grade(answer: str) -> float:
    """Turn a published answer into a number from 0 (no) to 1 (yes).

    A qualification after a comma or "but" is dropped: "No, self in victim-group" is no.
    """
    head = re.split(r",| but ", answer.strip().lower())[0].strip()
    if head not in GRADES:
        raise ValueError(f"no grade for the published answer {answer!r}")
    return GRADES[head]


def encode(codes: Codes) -> Floats:
    """Return one dilemma's codes as numbers, in the order of FEATURES."""
    killing = codes.involves_killing.strip().lower()
    if killing not in KILLING:
        raise ValueError(f"no code for 'involves killing' = {codes.involves_killing!r}")
    contexts = {part.strip() for part in codes.context.split("/")}
    if not contexts <= set(CONTEXTS):
        raise ValueError(f"unknown context {codes.context!r}")
    return np.array(
        [
            KILLING[killing],
            grade(codes.action_is_utilitarian),
            grade(codes.subset_of_benefited),
            grade(codes.self_in_benefited),
            grade(codes.opposite_framing),
            *(float(context in contexts) for context in CONTEXTS),
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(raw_dir: Path = RAW_DIR, manifest: Path = MANIFEST) -> None:
    """Download any missing participant-level file, then check every file against the
    manifest.

    A file with no manifest entry is recorded (URL, size, sha256). A file whose hash differs
    from its entry raises, so a changed upstream file cannot pass unnoticed.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, dict[str, str | int]] = (
        json.loads(manifest.read_text())["files"] if manifest.exists() else {}
    )
    for study in STUDIES.values():
        path = raw_dir / study.filename
        if not path.exists():
            with urllib.request.urlopen(study.url) as response:
                path.write_bytes(response.read())
        found = {"url": study.url, "size": path.stat().st_size, "sha256": _sha256(path)}
        expected = recorded.setdefault(study.filename, found)
        if expected != found:
            raise ValueError(f"{study.filename} does not match {manifest.name}: {found}")
    document = {
        "dataset": "https://osf.io/cg5tq/",
        "licence": "none stated on the OSF project; the paper is CC BY 4.0",
        "files": recorded,
    }
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def allowed_columns(study: int, dilemmas: dict[str, Dilemma]) -> dict[str, tuple[str, ...]]:
    """Return the only columns `ratings` may read from a study's file, by dilemma, or raise
    if one of them is not a judgment column."""
    allowed = {d.name: d.columns[study] for d in dilemmas.values() if study in d.columns}
    refused = sorted(
        column
        for columns in allowed.values()
        for column in columns
        if not JUDGMENT_COLUMN.fullmatch(column)
    )
    if refused:
        raise ValueError(f"only judgment columns may be read: {refused}")
    return allowed


def ratings(
    raw_dir: Path = RAW_DIR, dilemmas: dict[str, Dilemma] | None = None
) -> dict[int, dict[str, Rating]]:
    """Return, for each study and dilemma, how many participants judged it and the mean.

    Each record adds to a count and a sum and is then dropped. No record, and no cell outside
    the judgment columns, is kept or returned.
    """
    table = published() if dilemmas is None else dilemmas
    low, high = SCALE
    valid = {str(point): point for point in range(low, high + 1)}
    result: dict[int, dict[str, Rating]] = {}
    for number, study in STUDIES.items():
        allowed = allowed_columns(number, table)
        wanted = {column for columns in allowed.values() for column in columns}
        count = dict.fromkeys(allowed, 0)
        total = dict.fromkeys(allowed, 0)
        with (raw_dir / study.filename).open(encoding="latin-1", newline="") as handle:
            reader = csv.reader(handle, delimiter=study.delimiter)
            position = {name: i for i, name in enumerate(next(reader)) if name in wanted}
            if set(position) != wanted:
                raise ValueError(f"{study.filename} lacks {sorted(wanted - set(position))}")
            for record in reader:
                for name, columns in allowed.items():
                    for column in columns:
                        cell = record[position[column]].strip()
                        if not cell:
                            continue  # this participant was not shown this dilemma
                        if cell not in valid:
                            raise ValueError(f"{study.filename}: {column} holds {cell!r}")
                        count[name] += 1
                        total[name] += valid[cell]
        empty = sorted(name for name in allowed if count[name] == 0)
        if empty:
            raise ValueError(f"{study.filename}: no judgment of {empty}")
        result[number] = {name: Rating(count[name], total[name] / count[name]) for name in allowed}
    return result


def discrepancy(
    rated: dict[int, dict[str, Rating]], dilemmas: dict[str, Dilemma]
) -> dict[int, tuple[float, str]]:
    """Return, for each study, the largest difference between a mean computed from the raw
    file and the mean printed in the paper, with the dilemma it occurs in."""
    worst: dict[int, tuple[float, str]] = {}
    for study, by_name in rated.items():
        gaps = {
            name: abs(rating.mean - dilemmas[name].published[study])
            for name, rating in by_name.items()
        }
        name = max(gaps, key=lambda key: gaps[key])
        worst[study] = (gaps[name], name)
    return worst


def check_against_paper(rated: dict[int, dict[str, Rating]], dilemmas: dict[str, Dilemma]) -> None:
    """Raise unless every computed mean equals the paper's printed mean up to rounding."""
    for study, (gap, name) in discrepancy(rated, dilemmas).items():
        if gap > ROUNDING + 1e-9:
            raise ValueError(
                f"study {study}, {name}: the mean from the raw file differs from the paper's "
                f"table by {gap:.4f}, more than rounding allows"
            )
