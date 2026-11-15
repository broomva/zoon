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

A dilemma with no trial among the fitting targets takes the mean of the fitted intercepts.

Differences from the paper. The authors fit each participant separately, with priors, and
compare models by marginal likelihood. Here each model has one set of parameters shared by
all participants, fitted by maximum likelihood, because the quantity of interest is the
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
TEMPERATURE_BOUNDS = (0.0, 20.0)


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
    fitted = [rate(targets & (data.dilemma == d)) if seen[d] else np.nan for d in range(dilemmas)]
    return np.where(seen, fitted, np.nanmean(fitted))


def fit_constant(
    data: Sequences, targets: Bools, dilemmas: int, dilemma_intercepts: bool
) -> Parameters:
    """Fit the no-learning model to the trials marked in `targets`."""
    intercept = _base_rate_logits(data, targets, dilemmas, shared=not dilemma_intercepts)
    return Parameters(intercept, 0.0, 0.0)


def fit_learning(
    data: Sequences, targets: Bools, dilemmas: int, dilemma_intercepts: bool
) -> Parameters:
    """Fit the learning model to the trials marked in `targets` by maximum likelihood.

    Trials outside `targets` add nothing to the likelihood, and the model still learns from
    their feedback, because the participant saw it.
    """
    seen = np.array([bool((targets & (data.dilemma == d)).any()) for d in range(dilemmas)])
    free = np.flatnonzero(seen) if dilemma_intercepts else np.array([], dtype=np.int64)
    sign = np.where(data.choice == 1, 1.0, -1.0)

    def unpack(x: Floats) -> tuple[Floats, float, float]:
        intercept = np.zeros(dilemmas)
        if dilemma_intercepts:
            intercept[free] = x[2:]
            intercept[~seen] = x[2:].mean()
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
        return -float(log_expit(sign * logits)[targets].sum()), gradient

    start_intercepts = _base_rate_logits(data, targets, dilemmas, shared=False)[free]
    starts = [
        np.concatenate([[learning_rate, temperature], start_intercepts])
        for learning_rate in (0.05, 0.2, 0.5, 0.9)
        for temperature in (0.5, 2.0, 6.0)
    ]
    start = min(starts, key=lambda x: loss(x)[0])
    bounds = [LEARNING_RATE_BOUNDS, TEMPERATURE_BOUNDS] + [(None, None)] * len(free)
    best = minimize(loss, start, jac=True, method="L-BFGS-B", bounds=bounds)
    intercept, learning_rate, temperature = unpack(best.x)
    return Parameters(intercept, learning_rate, temperature)
