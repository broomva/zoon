"""The registered two-number recalibration of a prompted Yes probability."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from zoon.pilot.models import PROBABILITY_FLOOR, Bools, Floats, Ints


@dataclass(frozen=True)
class Fit:
    a: float
    c: float
    success: bool


def input_logit(yes: Floats, action_is_utilitarian: Bools) -> Floats:
    p1 = np.clip(yes, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    p2 = np.where(action_is_utilitarian, p1, 1 - p1)
    return logit(p2)


def probabilities(yes: Floats, action_is_utilitarian: Bools, fit: Fit) -> Floats:
    return np.clip(
        expit(fit.a * input_logit(yes, action_is_utilitarian) + fit.c),
        PROBABILITY_FLOOR,
        1 - PROBABILITY_FLOOR,
    )


def fit(yes: Floats, action_is_utilitarian: Bools, choice: Ints) -> Fit:
    if not (yes.shape == action_is_utilitarian.shape == choice.shape) or not yes.size:
        raise ValueError("calibration arrays must be nonempty and have the same shape")
    if not np.isfinite(yes).all() or not np.isin(choice, [0, 1]).all():
        raise ValueError("invalid calibration probabilities or choices")
    logits = input_logit(yes, action_is_utilitarian)

    def loss(x: Floats) -> tuple[float, Floats]:
        raw: Floats = expit(float(x[0]) * logits + float(x[1]))
        q = np.clip(raw, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        chosen = np.where(choice == 1, q, 1 - q)
        residual = np.where(
            (raw > PROBABILITY_FLOOR) & (raw < 1 - PROBABILITY_FLOOR), raw - choice, 0
        )
        return -float(np.log(chosen).sum()), np.array([(residual * logits).sum(), residual.sum()])

    result = minimize(
        loss, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B", bounds=[(-10, 10), (-10, 10)]
    )
    if not result.success:
        return Fit(1.0, 0.0, False)
    return Fit(float(result.x[0]), float(result.x[1]), True)
