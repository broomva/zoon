"""Registered judge transforms and calibration, with no participant data."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from zoon.pilot import clients, fetch, lean, outside
from zoon.pilot.models import Floats


@dataclass(frozen=True)
class Calibration:
    correlation: float | None
    passed: bool


def evaluative(response: clients.Response) -> float:
    return sum(i * response.probabilities[str(i)] for i in range(1, 8))


def features(responses: Sequence[clients.Response], action_is_utilitarian: bool) -> Floats:
    if len(responses) != 5:
        raise ValueError("strict judge needs five responses")
    yes = [r.probabilities["Yes"] for r in responses[:4]]
    return np.array(
        [
            yes[0],
            float(action_is_utilitarian),
            *yes[1:],
            *(responses[4].probabilities[c] for c in outside.CONTEXTS),
        ]
    )


def correlation(first: Floats, second: Floats) -> float | None:
    if first.shape != second.shape or first.size < 2:
        return None
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    if first.std() == 0 or second.std() == 0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else None


def calibration(ratings: Floats, human: Floats) -> Calibration:
    value = correlation(ratings, human)
    return Calibration(value, value is not None and value >= 0.50)


def standardise(ratings: Floats, held_out: Sequence[int]) -> Floats:
    rest = ratings[~np.isin(np.arange(len(ratings)), held_out)]
    if not np.isfinite(ratings).all() or rest.size == 0:
        raise clients.StepFailure("invalid judge ratings")
    spread = float(rest.std(ddof=0))
    return np.zeros_like(ratings) if spread < 1e-9 else (ratings - rest.mean()) / spread


@dataclass(frozen=True)
class StrictMap:
    fitted: lean.Map
    finite: lean.Map
    finite_error: float
    finite_correlation: float | None


def fit_map(names: tuple[str, ...], descriptive: Floats, human: Floats) -> StrictMap:
    penalty = lean.choose_penalty(descriptive, human)
    offset, weights = lean.ridge(descriptive, human, penalty)
    finite = lean.choose_penalty(descriptive, human, lean.FINITE_PENALTIES)
    finite_offset, finite_weights = lean.ridge(descriptive, human, finite)
    predicted = lean.left_out_predictions(descriptive, human, (finite,))
    return StrictMap(
        lean.Map(names, penalty, offset, weights),
        lean.Map(names, finite, finite_offset, finite_weights),
        float(np.sqrt(np.mean((predicted - human) ** 2))),
        correlation(predicted, human),
    )


def agreement(predicted: Floats, published: Floats) -> dict[str, float]:
    if predicted.shape != published.shape or predicted.shape[1] != len(outside.FEATURES):
        raise ValueError("strict features must match the published feature matrix")
    return {
        name: float(np.abs(predicted[:, i] - published[:, i]).mean())
        for i, name in enumerate(outside.FEATURES)
        if i != 1
    }


def human_ratings() -> dict[str, float]:
    return {name: dilemma.published[2] for name, dilemma in outside.published().items()}


def norm_ratings(human: Mapping[str, float] | None = None) -> Floats:
    human = human_ratings() if human is None else human
    ratings = np.array([human[outside.originals()[name]] for name in fetch.DILEMMAS])
    return lean.suggested_action_rating(
        ratings, np.array([outside.action_reversed()[name] for name in fetch.DILEMMAS])
    )
