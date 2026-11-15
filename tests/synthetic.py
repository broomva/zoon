"""Synthetic data for the tests: simulated choices, and raw files in the published layout.

Nothing here reads the real dataset. The simulator re-implements the learning rule on its
own, so that fitting it back is a check on `zoon.pilot.models` and not a tautology.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.special import expit

from zoon.pilot import fetch
from zoon.pilot.models import Floats, Sequences

DILEMMAS = len(fetch.DILEMMAS)
SECRET = "SECRET-PERSONAL-VALUE"
PERSONAL_COLUMNS = (
    "ProlificID",
    "IPAddress.x",
    "LocationLatitude.x",
    "LocationLongitude.x",
    "IPAddress.y",
    "LocationLatitude.y",
    "LocationLongitude.y",
    "RecipientEmail.x",
    "RecipientLastName",
    "ExternalReference.y",
    "ResponseId.x",
    "strategy_open",
)


def simulate(
    people: int,
    learning_rate: float,
    temperature: float,
    seed: int,
    intercept: Floats | None = None,
) -> Sequences:
    """Simulate choices from the learning model, each person seeing the dilemmas in their own
    random order. For half the people the cost-benefit option is followed by good outcomes
    (feedback near +0.6, else near -0.6); for the other half the rule option is."""
    rng = np.random.default_rng(seed)
    lean = np.zeros(DILEMMAS) if intercept is None else intercept
    dilemma = np.array([rng.permutation(DILEMMAS) for _ in range(people)])
    cbr_succeeds = rng.random(people) < 0.5
    choice = np.zeros((people, DILEMMAS), dtype=np.int64)
    feedback = np.zeros((people, DILEMMAS))
    value = np.zeros((people, 2))
    who = np.arange(people)
    for t in range(DILEMMAS):
        logits = lean[dilemma[:, t]] + temperature * (value[:, 1] - value[:, 0])
        chosen = (rng.random(people) < expit(logits)).astype(np.int64)
        good = (chosen == 1) == cbr_succeeds
        outcome = np.clip(np.where(good, 0.6, -0.6) + rng.normal(0, 0.2, people), -1, 1)
        value[who, chosen] = value[who, chosen] + learning_rate * (outcome - value[who, chosen])
        choice[:, t], feedback[:, t] = chosen, outcome
    return Sequences(choice, feedback, dilemma)


def write_raw(
    raw_dir: Path,
    people: int,
    unreadable: frozenset[tuple[int, str]] = frozenset(),
) -> dict[tuple[int, str], dict[str, int]]:
    """Write one raw file per experiment in the published layout, personal columns included.

    Each row depends only on its experiment and row label, so writing more people leaves the
    earlier rows unchanged. Rows listed in `unreadable` get no dilemma order and no
    responses: reading them as a participant's trials would raise.

    Return, for every participant who passes the exclusion rules, the choices they were given:
    {(experiment, row label): {dilemma: 1 if the cost-benefit option was chosen}}.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    intended: dict[tuple[int, str], dict[str, int]] = {}
    for experiment, source in fetch.SOURCES.items():
        responses = [
            f"{dilemma}_Q_{suffix}" if kind == "Q" else f"{dilemma}_V_{suffix}_1"
            for suffix in ("YS", "NS")
            for dilemma in fetch.DILEMMAS
            for kind in ("Q", "V")
        ]
        header = ["", *PERSONAL_COLUMNS, *source.exclusion_columns, *source.order_columns]
        header += responses
        with (raw_dir / source.filename).open("w", newline="", encoding="latin-1") as handle:
            writer = csv.DictWriter(handle, header, delimiter=source.delimiter, restval="")
            writer.writeheader()
            for label in map(str, range(1, people + 1)):
                rng = np.random.default_rng([experiment, int(label)])
                attentive = bool(rng.random() > 0.1)
                checks = {
                    "Attn_Q": "Yes" if attentive else "No",
                    "Attn_M_1": "100",
                    "Organ_MC_CL1_16": "100",
                    "WW2_MC_CL1_16": "100",
                    "Finished.x": "True",
                    "Finished.y": "True",
                    "mc_1": "No",
                }
                row = {"": label} | dict.fromkeys(PERSONAL_COLUMNS, f"{SECRET}-{label}")
                row |= {name: cell for name, cell in checks.items() if name in header}
                cbr_succeeds = bool(rng.random() < 0.5)
                chosen: dict[str, int] = {}
                blocks: list[str] = []
                for position in rng.permutation(DILEMMAS):
                    dilemma = fetch.DILEMMAS[int(position)]
                    yes_succeeds = fetch.CBR_IS_ACTION[dilemma] == cbr_succeeds
                    suffix = "YS" if yes_succeeds else "NS"
                    chosen[dilemma] = int(rng.random() < 0.5)
                    yes = bool(chosen[dilemma]) == fetch.CBR_IS_ACTION[dilemma]
                    row[f"{dilemma}_Q_{suffix}"] = "Yes" if yes else "No"
                    row[f"{dilemma}_V_{suffix}_1"] = str(int(rng.integers(-100, 101)))
                    blocks.append(f"{suffix}_{dilemma}")
                blocks.insert(int(rng.integers(0, len(blocks))), "AttnCheck")
                row[source.order_columns[0 if cbr_succeeds else 1]] = "|".join(blocks)
                if (experiment, label) in unreadable:
                    row = {name: cell for name, cell in row.items() if name not in responses}
                    row |= dict.fromkeys(source.order_columns, "")
                writer.writerow(row)
                if attentive:
                    intended[(experiment, label)] = chosen
    return intended
