"""Training-only rotation fits and trialwise probabilities for the coded arms."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from zoon.pilot import cohorts, fetch, judge, lean, models, outside, power, score
from zoon.pilot.models import Floats, Parameters, Sequences


@dataclass(frozen=True)
class Fit:
    parameters: Parameters
    count: int
    alpha: float | None = None
    beta: float | None = None


@dataclass(frozen=True)
class Rotation:
    held_out: tuple[int, ...]
    arms: dict[str, Fit]


def sequences(people: Sequence[cohorts.Participant]) -> Sequences:
    ordered = sorted(people, key=lambda p: (p.experiment, p.participant))
    if not ordered or len({(p.experiment, p.participant) for p in ordered}) != len(ordered):
        raise ValueError("participants must be nonempty and unique")
    data = Sequences(
        np.array(
            [
                [int((t.answer == "Yes") == fetch.CBR_IS_ACTION[t.dilemma]) for t in p.trials]
                for p in ordered
            ],
            dtype=np.int64,
        ),
        np.array([[t.rating / 100 for t in p.trials] for p in ordered]),
        np.array(
            [[fetch.DILEMMAS.index(t.dilemma) for t in p.trials] for p in ordered], dtype=np.int64
        ),
    )
    score.complete(data)
    return data


def reference_ratings() -> dict[str, Floats]:
    published = outside.published()
    rated = {name: outside.Rating(0, d.published[2]) for name, d in published.items()}
    fitted = lean.fit_map(published, outside.originals(), rated)
    coded = lean.coded_lean(fitted, outside.original_codes()).rating
    return {"sign": np.zeros(len(fetch.DILEMMAS)), "coded": coded, "norm": judge.norm_ratings()}


def fit_rotations(training: Sequences, ratings: Mapping[str, Floats]) -> list[Rotation]:
    score.complete(training)
    count = len(fetch.DILEMMAS)
    result: list[Rotation] = []
    for pair in power.held_out_sets(True, count):
        targets = ~np.isin(training.dilemma, pair)
        everything = np.ones(training.choice.shape, dtype=np.bool_)
        fitted: dict[str, Fit] = {
            "pooled": Fit(models.fit_constant(training, targets, count, True), count - len(pair)),
            "laws-only": Fit(
                models.fit_learning(training, targets, count, True), count - len(pair) + 2
            ),
            "known share": Fit(models.fit_constant(training, everything, count, True), count),
            "known share" + lean.LEARNING: Fit(
                models.fit_learning(training, everything, count, True), count + 2
            ),
        }
        for name, raw in ratings.items():
            measure = judge.standardise(raw, pair)
            is_judge = "/zoon-" in name
            for learning in (False, True):
                arm = (
                    name
                    if is_judge and learning
                    else name.replace("/zoon-", "/no-state-")
                    if is_judge
                    else name + (lean.LEARNING if learning else "")
                )
                fit = models.fit_lean(training, targets, lean.SIGN, measure, learning)
                parameters = 1 + int(bool(measure.any())) + (2 if learning else 0)
                fitted[arm] = Fit(fit.parameters, parameters, fit.alpha, fit.beta)
        result.append(Rotation(tuple(pair), fitted))
    return result


def predict(rotations: Sequence[Rotation], data: Sequences) -> dict[str, list[Floats]]:
    score.complete(data)
    result: dict[str, list[Floats]] = {}
    for rotation in rotations:
        for name, fit in rotation.arms.items():
            observed = models.probabilities(fit.parameters, data)
            result.setdefault(name, []).append(np.where(data.choice == 1, observed, 1 - observed))
    return result
