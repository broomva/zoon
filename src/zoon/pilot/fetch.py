"""Fetch the moral-learning data from OSF and load the training participants.

Data: Maier, Cheung and Lieder, "Metacognitive learning from consequences of past choices
shapes moral decision-making", https://osf.io/4up5z/, CC-BY 4.0.
Paper: https://doi.org/10.1038/s41562-025-02271-w

The raw files are never committed. `fetch` downloads them into `data/raw/` and checks each
against `data-manifest.json`. `load` returns one row per training participant and trial.

Two rules shape `load`:

- Only the columns named in this module are ever copied out of a raw record. The raw files
  also hold platform identifiers, network addresses, coordinates, demographics and free text;
  none of those columns is on the list, and a guard refuses to run if one is added.
- A held-out participant contributes one to a count and nothing else. The only cells read
  from a held-out record are the identifier and the fields of the published exclusion
  rules.
"""

import csv
import hashlib
import json
import math
import re
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zoon.pilot.split import assign, participant_index

REPO = Path(__file__).resolve().parents[3]
RAW_DIR = REPO / "data" / "raw"
MANIFEST = REPO / "data-manifest.json"

# The 13 dilemmas, in the order of the authors' analysis scripts. True where the option
# recommended by cost-benefit reasoning is the action the vignette asks about ("Yes"), False
# where it is the omission ("No"). Eight are framed one way and five the other.
CBR_IS_ACTION: dict[str, bool] = {
    "Ladder": True,
    "Tyran": True,
    "Lifeboat": True,
    "Suicide": True,
    "Medicine": False,
    "RAF": False,
    "Rwanda": False,
    "Ransom": False,
    "Tax": True,
    "Operations": True,
    "Vet": False,
    "Endow": True,
    "RobinHood": True,
}
DILEMMAS: tuple[str, ...] = tuple(CBR_IS_ACTION)
CONDITIONS: tuple[str, str] = ("cbr_success", "rule_success")

FORBIDDEN_COLUMN = re.compile(
    r"prolific|ipaddress|location|recipient|externalreference|participant\.id|submission\.id"
    r"|email|name|age|sex|ethnic|country|national|language|open_|_open|responseid",
    re.IGNORECASE,
)

# The raw participant identifier is the row label in the first, unnamed column of each file.
# The authors blanked the platform identifiers before publishing, so the row label is the only
# identifier all four files have. It is stable because the manifest pins each file's hash.
ROW_LABEL = ""

_COMMON_CHECKS = ("Attn_Q", "Attn_M_1")
# Experiment 2 has a second attention check inside a later questionnaire. Each participant
# answered it for one of three vignettes, so the answer sits in one of three columns.
_RULE_CHECKS = ("Organ_MC_CL1_16", "Sweatshop_MC_CL1_16", "Prisoner_MC_CL1_16")
_CBR_CHECKS = ("Torture_MC_CL1_16", "Ethics_MC_CL1_16", "WW2_MC_CL1_16")


@dataclass(frozen=True)
class Source:
    """Where one experiment's raw file lives and how to read it."""

    filename: str
    url: str
    delimiter: str
    # Qualtrics display-order columns holding each participant's dilemma order, one per
    # condition, in the order of CONDITIONS.
    order_columns: tuple[str, str]
    exclusion_columns: tuple[str, ...]


SOURCES: dict[int, Source] = {
    1: Source(
        "exp1_data_overall.csv",
        "https://osf.io/download/mq38j/",
        ";",
        ("FL_30_DO", "FL_31_DO"),
        _COMMON_CHECKS,
    ),
    2: Source(
        "exp2_dataExp2.csv",
        "https://osf.io/download/vknx5/",
        ",",
        ("FL_92_DO", "FL_77_DO"),
        (*_COMMON_CHECKS, *_RULE_CHECKS, *_CBR_CHECKS),
    ),
    3: Source(
        "exp3_dataExp3.csv",
        "https://osf.io/download/c789b/",
        ",",
        ("FL_36_DO", "FL_21_DO"),
        _COMMON_CHECKS,
    ),
    4: Source(
        "exp4_dataExp4.csv",
        "https://osf.io/download/fp7ed/",
        ",",
        ("FL_38_DO", "FL_23_DO"),
        (*_COMMON_CHECKS, "Finished.x", "Finished.y", "mc_1"),
    ),
}


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
    for source in SOURCES.values():
        path = raw_dir / source.filename
        if not path.exists():
            with urllib.request.urlopen(source.url) as response:
                path.write_bytes(response.read())
        found = {"url": source.url, "size": path.stat().st_size, "sha256": _sha256(path)}
        expected = recorded.setdefault(source.filename, found)
        if expected != found:
            raise ValueError(f"{source.filename} does not match {manifest.name}: {found}")
    document = {
        "dataset": "https://osf.io/4up5z/",
        "licence": "CC-BY 4.0",
        "files": recorded,
    }
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _response_columns(condition: str) -> dict[str, tuple[str, str]]:
    """Map each dilemma to its choice and valence columns for one condition.

    The suffix says which answer was followed by the good outcome: YS (yes) or NS (no).
    """
    columns: dict[str, tuple[str, str]] = {}
    for dilemma, cbr_is_action in CBR_IS_ACTION.items():
        yes_succeeds = cbr_is_action == (condition == "cbr_success")
        suffix = "YS" if yes_succeeds else "NS"
        columns[dilemma] = (f"{dilemma}_Q_{suffix}", f"{dilemma}_V_{suffix}_1")
    return columns


def allowed_columns(source: Source) -> set[str]:
    """Return the only columns `load` may read from a raw file, or raise if one is personal."""
    allowed = {ROW_LABEL, *source.order_columns, *source.exclusion_columns}
    for condition in CONDITIONS:
        for choice, valence in _response_columns(condition).values():
            allowed |= {choice, valence}
    leaked = sorted(column for column in allowed if FORBIDDEN_COLUMN.search(column))
    if leaked:
        raise ValueError(f"personal-data columns may not be read: {leaked}")
    return allowed


class RecordFault(ValueError):
    """A participant record cannot be read as valid trials."""


def _number(cell: str) -> float | None:
    try:
        return float(cell.replace(",", "."))
    except ValueError:
        return None


def _first_filled(cells: dict[str, str], columns: tuple[str, ...]) -> float | None:
    return next((n for c in columns if (n := _number(cells[c])) is not None), None)


def _included(experiment: int, cells: dict[str, str]) -> bool:
    """Apply the exclusion rules of the authors' analysis scripts for one experiment.

    Experiment 1 also had a rule against very fast completion. It is not applied here: its
    threshold is computed from every participant's duration, and the paper reports that it
    excluded nobody.
    """
    passed = cells["Attn_Q"] == "Yes" and _number(cells["Attn_M_1"]) == 100
    if experiment == 2:
        passed = passed and _first_filled(cells, _RULE_CHECKS) == 100
        passed = passed and _first_filled(cells, _CBR_CHECKS) == 100
    if experiment == 4:
        completed = cells["Finished.x"].lower() == "true" and cells["Finished.y"].lower() == "true"
        passed = passed and completed and cells["mc_1"] != "Yes"
    return passed


def _trials(source: Source, cells: dict[str, str]) -> Iterator[dict[str, str | int | float]]:
    """Turn one training participant's record into 13 trial rows, in the order they saw."""
    condition = CONDITIONS[0] if cells["Ladder_Q_YS"] in ("Yes", "No") else CONDITIONS[1]
    responses = _response_columns(condition)
    shown = cells[source.order_columns[CONDITIONS.index(condition)]].split("|")
    order = [block.partition("_")[2] for block in shown if block != "AttnCheck"]
    if sorted(order) != sorted(DILEMMAS):
        raise RecordFault("unexpected dilemma order")
    for trial, dilemma in enumerate(order, start=1):
        choice_column, valence_column = responses[dilemma]
        answer, valence = cells[choice_column], _number(cells[valence_column])
        if answer not in ("Yes", "No") or valence is None or not math.isfinite(valence):
            raise RecordFault("missing or invalid response")
        action = answer == "Yes"
        cbr = action == CBR_IS_ACTION[dilemma]
        yield {
            "condition": condition,
            "dilemma": dilemma,
            "trial": trial,
            "cbr_choice": int(cbr),
            "action_choice": int(action),
            "success": int(cbr == (condition == "cbr_success")),
            "valence": valence / 100,
        }


def _scan(
    raw_dir: Path, *, count_sealed: bool = False
) -> tuple[list[dict[str, str | int | float]], dict[int, int]]:
    """Read every raw file once. Return training trial rows and held-out counts."""
    rows: list[dict[str, str | int | float]] = []
    held_out: dict[int, int] = {}
    for experiment, source in SOURCES.items():
        allowed = allowed_columns(source)
        gate = (ROW_LABEL, *source.exclusion_columns)
        held_out[experiment] = 0
        seen: set[str] = set()
        with (raw_dir / source.filename).open(encoding="latin-1", newline="") as handle:
            reader = csv.reader(handle, delimiter=source.delimiter)
            position = {name: i for i, name in enumerate(next(reader)) if name in allowed}
            if set(position) != allowed:
                raise ValueError(f"{source.filename} lacks {sorted(allowed - set(position))}")
            for record in reader:
                cells = {name: record[position[name]] for name in gate}
                raw_id = cells[ROW_LABEL]
                if not raw_id or raw_id in seen:
                    raise ValueError(f"{source.filename}: empty or repeated identifier")
                seen.add(raw_id)
                if not _included(experiment, cells):
                    continue
                if assign(experiment, raw_id) == "held_out":
                    if count_sealed:
                        held_out[experiment] += 1
                    continue
                cells = {name: record[position[name]] for name in allowed}
                index = participant_index(experiment, raw_id)
                for trial in _trials(source, cells):
                    rows.append({"experiment": experiment, "participant": index, **trial})
    return rows, held_out


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return the trial-level table of the training participants.

    Columns: experiment (1 to 4), participant (opaque index), condition (which kind of option
    was followed by good outcomes), dilemma, trial (1 to 13, the order the participant saw),
    cbr_choice (1 = the option recommended by cost-benefit reasoning, 0 = the option
    recommended by a moral rule), action_choice (1 = "Yes", the action the vignette asks
    about), success (1 = the chosen option was followed by the good outcome) and valence (the
    participant's rating of how good or bad the consequences were, rescaled from -100..100
    to -1..1).
    """
    rows, _ = _scan(raw_dir)
    return pd.DataFrame(rows)


def held_out_counts(raw_dir: Path = RAW_DIR) -> dict[int, int]:
    """Return how many participants who pass the exclusion rules are held out in each
    experiment, and nothing else."""
    _, held_out = _scan(raw_dir, count_sealed=True)
    return held_out


def load_registered_held_out(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load held-out trials only for the registered fifth step."""
    rows: list[dict[str, str | int | float]] = []
    for experiment, source in SOURCES.items():
        allowed = allowed_columns(source)
        gate = (ROW_LABEL, *source.exclusion_columns)
        seen: set[str] = set()
        with (raw_dir / source.filename).open(encoding="latin-1", newline="") as handle:
            reader = csv.reader(handle, delimiter=source.delimiter)
            position = {name: i for i, name in enumerate(next(reader)) if name in allowed}
            if set(position) != allowed:
                raise RecordFault("raw file lacks required columns")
            for record in reader:
                cells = {name: record[position[name]] for name in gate}
                raw_id = cells[ROW_LABEL]
                if not raw_id or raw_id in seen:
                    raise RecordFault("empty or repeated identifier")
                seen.add(raw_id)
                if not _included(experiment, cells) or assign(experiment, raw_id) != "held_out":
                    continue
                if len(record) <= max(position.values()):
                    raise RecordFault("unreadable record")
                cells = {name: record[position[name]] for name in allowed}
                index = participant_index(experiment, raw_id)
                for trial in _trials(source, cells):
                    rows.append({"experiment": experiment, "participant": index, **trial})
    return pd.DataFrame(rows)
