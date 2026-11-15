"""A learning model and a no-learning baseline for choices between two kinds of option.

Both follow the Stan programs published with the data (CompModels.zip at
https://osf.io/4up5z/). The learning model is `model_free.stan`:

    initV = rep_array(0, 2);
    target += categorical_logit_lpmf(util_choice[i]|tau * to_vector(v));
    pe = value[i] - v[util_choice[i]];
    v[util_choice[i]] = v[util_choice[i]] + lr * pe;

with `lr` in [0, 1] and `tau` in [0, 20]. The baseline is `constant.stan`:

    target+=bernoulli_lpmf(util_choice[n]|theta);

The two kinds of option are the one recommended by cost-benefit reasoning (index 1) and the
one recommended by a moral rule (index 0). With two options the categorical logit reduces to

    P(cost-benefit option) = sigmoid(tau * (v[1] - v[0])).

Each model also has a variant with one fitted intercept per dilemma, added to the logit. All
four are one likelihood,

    logit P(cost-benefit option on trial t) = intercept[dilemma_t] + tau * (v[1] - v[0]),

with different parts fixed:

    constant                      one shared intercept; tau = 0
    constant, dilemma intercepts  one intercept per dilemma; tau = 0
    learning                      intercept = 0; lr and tau fitted
    learning, dilemma intercepts  one intercept per dilemma; lr and tau fitted

A dilemma with no trial among the fitting targets takes the share of choices over all
fitting targets, as a logit; the learning model adds the average shift of its fitted intercepts.

Differences from the paper. The authors fit each participant separately, with priors, and
compare models by marginal likelihood. Here each model has one set of parameters shared by
all participants, fitted by maximum likelihood with the paper's prior on the temperature
kept as a penalty, because the quantity of interest is the
prediction of people the model has not seen. Pooling is a simplification: it ignores
differences between people in learning rate and temperature.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, log_expit

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]

PROBABILITY_FLOOR = 1e-3
LEARNING_RATE_BOUNDS = (0.0, 1.0)
TEMPERATURE_BOUNDS = (1e-6, 20.0)
# The paper's prior on the temperature: `lognormal_lpdf(tau | 0, 1.4)` in model_free.stan.
TEMPERATURE_PRIOR_SCALE = 1.4


@dataclass(frozen=True)
class Sequences:
    """Trials of several participants. Every array is participants x trials, in trial order."""

    choice: Ints  # 1 = cost-benefit option, 0 = rule option
    feedback: Floats  # the participant's rating of the outcome, in [-1, 1]
    dilemma: Ints  # index of the dilemma shown on that trial

    def take(self, rows: Bools | Ints) -> "Sequences":
        return Sequences(self.choice[rows], self.feedback[rows], self.dilemma[rows])


@dataclass(frozen=True)
class Parameters:
    intercept: Floats  # one per dilemma
    learning_rate: float
    temperature: float


def _run(
    data: Sequences, intercept: Floats, learning_rate: float, temperature: float
) -> tuple[Floats, Floats, Floats]:
    """Run the model over every participant's trials in order.

    Return the logit of choosing the cost-benefit option on each trial, the value difference
    v[1] - v[0] before each trial, and the derivative of that difference with respect to the
    learning rate. Values are updated after every trial, whether or not the trial is a
    fitting or scoring target.
    """
    people, trials = len(data.choice), len(data.choice[0])
    who = np.arange(people)
    value = np.zeros((people, 2))
    slope = np.zeros((people, 2))  # d value / d learning_rate
    gap = np.empty((people, trials))
    gap_slope = np.empty((people, trials))
    for t in range(trials):
        gap[:, t] = value[:, 1] - value[:, 0]
        gap_slope[:, t] = slope[:, 1] - slope[:, 0]
        chosen = data.choice[:, t]
        error = data.feedback[:, t] - value[who, chosen]
        slope[who, chosen] += error - learning_rate * slope[who, chosen]
        value[who, chosen] += learning_rate * error
    return intercept[data.dilemma] + temperature * gap, gap, gap_slope


def probabilities(parameters: Parameters, data: Sequences) -> Floats:
    """Return the model's probability of the choice each participant made on each trial."""
    logits, _, _ = _run(
        data, parameters.intercept, parameters.learning_rate, parameters.temperature
    )
    signed: Floats = np.where(data.choice == 1, logits, -logits)
    return expit(signed)


def log_likelihood(parameters: Parameters, data: Sequences) -> Floats:
    """Return the log-probability of each observed choice, with the probability clamped."""
    clamped = np.clip(probabilities(parameters, data), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return np.log(clamped)


def _base_rate_logits(data: Sequences, targets: Bools, dilemmas: int, shared: bool) -> Floats:
    """Maximum-likelihood intercepts of the constant model, as logits."""

    def rate(mask: Bools) -> float:
        share = float(data.choice[mask].mean())
        share = min(max(share, PROBABILITY_FLOOR), 1 - PROBABILITY_FLOOR)
        return math.log(share / (1 - share))

    if shared:
        return np.full(dilemmas, rate(targets))
    seen = [bool((targets & (data.dilemma == d)).any()) for d in range(dilemmas)]
    # An unseen dilemma takes the share over all fitting targets. Averaging the fitted logits
    # instead would overshoot, because the logit of a mean is not the mean of the logits.
    pooled = rate(targets)
    fitted = [rate(targets & (data.dilemma == d)) if seen[d] else pooled for d in range(dilemmas)]
    return np.array(fitted)


def fit_constant(
    data: Sequences, targets: Bools, dilemmas: int, dilemma_intercepts: bool
) -> Parameters:
    """Fit the no-learning model to the trials marked in `targets`."""
    intercept = _base_rate_logits(data, targets, dilemmas, shared=not dilemma_intercepts)
    return Parameters(intercept, 0.0, 0.0)


def fit_learning(
    data: Sequences,
    targets: Bools,
    dilemmas: int,
    dilemma_intercepts: bool,
    start: tuple[float, float] | None = None,
) -> Parameters:
    """Fit the learning model to the trials marked in `targets`.

    The fit maximises the likelihood times the paper's prior on the temperature. Over 13
    trials the values move so little that learning rate and temperature trade off along a
    flat ridge, and without the prior an optimiser stops at a different point on it on each
    machine. `start` sets the starting learning rate and temperature; it exists so that a
    test can check the result does not depend on it.

    Trials outside `targets` add nothing to the likelihood, and the model still learns from
    their feedback, because the participant saw it.
    """
    seen = np.array([bool((targets & (data.dilemma == d)).any()) for d in range(dilemmas)])
    free = np.flatnonzero(seen) if dilemma_intercepts else np.array([], dtype=np.int64)
    sign = np.where(data.choice == 1, 1.0, -1.0)

    constant_intercepts = _base_rate_logits(data, targets, dilemmas, shared=False)
    start_intercepts = constant_intercepts[free]
    count = int(targets.sum())

    def unpack(x: Floats) -> tuple[Floats, float, float]:
        intercept = np.zeros(dilemmas)
        if dilemma_intercepts:
            intercept[free] = x[2:]
            # An unseen dilemma starts from the constant model's value for it and moves by the
            # average amount that fitting moved the seen dilemmas.
            shift = float((x[2:] - start_intercepts).mean()) if len(free) else 0.0
            intercept[~seen] = constant_intercepts[~seen] + shift
        return intercept, float(x[0]), float(x[1])

    def loss(x: Floats) -> tuple[float, Floats]:
        intercept, learning_rate, temperature = unpack(x)
        logits, gap, gap_slope = _run(data, intercept, learning_rate, temperature)
        # d(log-likelihood) / d(logit) is the observed choice minus its probability.
        residual = np.where(targets, data.choice - expit(logits), 0.0)
        gradient = np.empty_like(x)
        gradient[0] = -(residual * temperature * gap_slope).sum()
        gradient[1] = -(residual * gap).sum()
        for position, d in enumerate(free):
            gradient[2 + position] = -residual[data.dilemma == d].sum()
        # Minus the log of the lognormal prior density of the temperature, up to a constant.
        log_temperature = math.log(temperature)
        penalty = log_temperature + log_temperature**2 / (2 * TEMPERATURE_PRIOR_SCALE**2)
        gradient[1] += (1 + log_temperature / TEMPERATURE_PRIOR_SCALE**2) / temperature
        negative_log_likelihood = -float(log_expit(sign * logits)[targets].sum())
        # Per choice, so that the stopping tolerances mean the same thing at any sample size.
        return (negative_log_likelihood + penalty) / count, gradient / count

    starts = [
        np.concatenate([[learning_rate, temperature], start_intercepts])
        for learning_rate in (0.05, 0.2, 0.5, 0.9)
        for temperature in (0.5, 2.0, 6.0)
    ]
    if start is not None:
        starts = [np.concatenate([list(start), start_intercepts])]
    first = min(starts, key=lambda x: loss(x)[0])
    bounds = [LEARNING_RATE_BOUNDS, TEMPERATURE_BOUNDS] + [(None, None)] * len(free)
    best = minimize(
        loss,
        first,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
    )
    intercept, learning_rate, temperature = unpack(best.x)
    return Parameters(intercept, learning_rate, temperature)
