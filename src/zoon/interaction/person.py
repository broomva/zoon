"""Six ordered-logit persons and their twins, fitted to whole training groups."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from zoon.interaction import measure
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats, Ints

LETTERS = tuple("ABCDEF")
FLOOR = 1e-9
RIDGE = 0.01
# Numerical bounds on log gaps; cut points remain strictly ordered.
LOG_GAP_LIMIT = 8.0
MAX_ITERATIONS = 2000
TOLERANCE = 1e-7
FUNCTION_TOLERANCE = 1e-10
REPRODUCIBILITY_TOLERANCE = 0.02
START_NUDGE = 1e-6


@dataclass(frozen=True)
class Candidate:
    letter: str
    twin: bool = False

    def __post_init__(self) -> None:
        if self.letter not in LETTERS:
            raise ValueError("unknown candidate")

    @property
    def name(self) -> str:
        return self.letter + (" twin" if self.twin else "")

    @property
    def traces(self) -> tuple[str, ...]:
        own = () if self.letter == "F" else ("own",)
        others = () if self.twin else (("same", "other") if self.letter == "C" else ("others",))
        return own + others

    @property
    def features(self) -> tuple[str, ...]:
        return self.traces + (("anchor",) if self.letter == "E" else ())

    @property
    def constants(self) -> int:
        return 5 if self.letter == "B" else len(measure.STRATA)

    @property
    def phase_size(self) -> int:
        return self.constants + 3

    @property
    def count(self) -> int:
        return 2 * self.phase_size + len(self.features) + len(self.traces)

    def start(self) -> Floats:
        phase = np.zeros(self.phase_size)
        return np.r_[phase, phase, np.zeros(len(self.features)), np.full(len(self.traces), 0.5)]

    def base(self, game: Game, seat: int) -> Floats:
        if self.letter == "B":
            unit = sum(game.endowments) / game.size / 24
            canonical = replace(game, endowments=tuple(round(e / unit) for e in game.endowments))
            return np.array([1.0, *measure.traits(canonical, seat)])
        row = np.zeros(self.constants)
        row[measure.STRATA[game.cell, measure.seat_type(game, seat)]] = 1
        return row

    def trace_sources(self, game: Game, seat: int) -> tuple[int, ...]:
        if self.letter == "C" and not self.twin:
            kind = measure.seat_type(game, seat)
            kinds = {measure.seat_type(game, s) == kind for s in range(game.size) if s != seat}
            if True not in kinds:
                return (0, 2, 2)
            if False not in kinds:
                return (0, 1, 1)
        return tuple(range(len(self.traces)))

    def observations(self, game: Game, seat: int, contributions: tuple[int, ...]) -> Floats:
        """Return the exact-share observations which can enter this person's traces."""
        own = contributions[seat] / game.endowments[seat]
        others = [s for s in range(game.size) if s != seat]
        values: dict[str, float] = {"own": own}
        if not self.twin:
            if self.letter == "D":
                values["others"] = min(
                    1.0, sum(contributions[s] for s in others) / len(others) / game.endowments[seat]
                )
            elif self.letter == "C":
                kind = measure.seat_type(game, seat)
                for label, same in (("same", True), ("other", False)):
                    seats = [s for s in others if (measure.seat_type(game, s) == kind) == same]
                    seats = seats or others
                    values[label] = sum(contributions[s] / game.endowments[s] for s in seats) / len(
                        seats
                    )
            else:
                values["others"] = sum(contributions[s] / game.endowments[s] for s in others) / len(
                    others
                )
        return np.array([values[name] for name in self.traces], dtype=float)


CANDIDATES = tuple(Candidate(letter, twin) for twin in (False, True) for letter in LETTERS)


@dataclass(frozen=True)
class State:
    traces: tuple[float, ...] = ()
    anchor: float = 0.0
    seen: bool = False


@dataclass(frozen=True)
class Person:
    candidate: Candidate
    numbers: Floats

    def __post_init__(self) -> None:
        if self.numbers.shape != (self.candidate.count,) or not np.isfinite(self.numbers).all():
            raise ValueError("wrong fitted numbers")

    @property
    def fractions(self) -> Floats:
        n = len(self.candidate.traces)
        return self.numbers[-n:] if n else np.empty(0)

    def update(self, state: State, game: Game, seat: int, contributions: tuple[int, ...]) -> State:
        observed = self.candidate.observations(game, seat, contributions)
        values = (
            np.array(state.traces) + self.fractions * (observed - state.traces)
            if state.seen
            else observed
        )
        anchor = 0.0
        if self.candidate.letter == "E":
            anchor = state.anchor if state.seen else contributions[seat] / game.endowments[seat]
        values = values[list(self.candidate.trace_sources(game, seat))]
        return State(tuple(float(v) for v in values), anchor, True)

    def probabilities(self, game: Game, seat: int, state: State = State()) -> Floats:
        c = self.candidate
        start = c.phase_size if state.seen else 0
        phase = self.numbers[start : start + c.phase_size]
        score = float(c.base(game, seat) @ phase[:-3])
        if state.seen:
            features = [*state.traces, *([state.anchor] if c.letter == "E" else [])]
            weights = self.numbers[2 * c.phase_size : 2 * c.phase_size + len(c.features)]
            score += float((np.array(features) - 0.5) @ weights)
        cuts: Floats = np.asarray(np.r_[0.0, np.cumsum(np.exp(phase[-3:]))], dtype=float)
        below = expit(cuts - score)
        p = np.maximum(np.diff(np.r_[0.0, below, 1.0]), FLOOR)
        return p / p.sum()


def ordered(score: Floats, cuts: Floats) -> tuple[Floats, Floats, Floats]:
    """Probabilities and their derivatives by score and three log-gap parameters."""
    gaps = np.exp(cuts)
    thresholds: Floats = np.asarray(np.r_[0.0, np.cumsum(gaps)], dtype=float)
    below = expit(thresholds[None, :] - score[:, None])
    density = below * (1 - below)
    p = np.diff(np.c_[np.zeros(len(score)), below, np.ones(len(score))], axis=1)
    ds = np.diff(np.c_[np.zeros(len(score)), -density, np.zeros(len(score))], axis=1)
    dc = np.zeros((len(score), 5, 3))
    for j in range(3):
        derivative = density * (np.arange(4) > j) * gaps[j]
        dc[:, :, j] = np.diff(np.c_[np.zeros(len(score)), derivative, np.zeros(len(score))], axis=1)
    active = p > FLOOR
    ds *= active
    dc *= active[:, :, None]
    p = np.maximum(p, FLOOR)
    total = p.sum(axis=1)
    ds = (ds - p * (ds.sum(axis=1) / total)[:, None]) / total[:, None]
    dc = (dc - p[:, :, None] * (dc.sum(axis=1) / total[:, None])[:, None, :]) / total[:, None, None]
    return p / total[:, None], ds, dc


@dataclass(frozen=True)
class Records:
    """One sequence per person, ordered as groups, seats, then rounds."""

    base: Floats
    observations: Floats
    anchor: Floats
    choice: Ints
    sources: Ints


def records(groups: Sequence[Group], candidate: Candidate) -> Records:
    sources: list[tuple[int, ...]] = []
    base: list[Floats] = []
    observed: list[Floats] = []
    anchor: list[float] = []
    choice: list[list[int]] = []
    for group in groups:
        game = GAMES[group.cell]
        for seat in range(game.size):
            base.append(candidate.base(game, seat))
            sources.append(candidate.trace_sources(game, seat))
            observed.append(
                np.array([candidate.observations(game, seat, row) for row in group.rounds])
            )
            anchor.append(group.rounds[0][seat] / game.endowments[seat])
            choice.append([measure.band(row[seat], game.endowments[seat]) for row in group.rounds])
    if not base:
        raise ValueError("a fit needs training groups")
    return Records(
        np.array(base),
        np.array(observed),
        np.array(anchor),
        np.array(choice, dtype=np.int64),
        np.array(sources, dtype=np.int64),
    )


def states(record: Records, c: Candidate, fractions: Floats) -> tuple[Floats, Floats]:
    """Trace values before each later decision and derivatives by update fraction."""
    n, _, k = record.observations.shape
    values, derivative = np.zeros((n, ROUNDS - 1, k)), np.zeros((n, ROUNDS - 1, k))
    if k:
        values[:, 0] = record.observations[:, 0]
        for r in range(1, ROUNDS - 1):
            delta = record.observations[:, r] - values[:, r - 1]
            values[:, r] = values[:, r - 1] + fractions * delta
            derivative[:, r] = (1 - fractions) * derivative[:, r - 1] + delta
    values = np.take_along_axis(values, record.sources[:, None, :], axis=2)
    if c.letter == "E":
        values = np.concatenate(
            [values, np.broadcast_to(record.anchor[:, None, None], (n, ROUNDS - 1, 1))], axis=2
        )
    return values - 0.5, derivative


def likelihood(numbers: Floats, record: Records, c: Candidate) -> tuple[float, Floats, Floats]:
    """Negative penalised likelihood, analytic gradient, and log scores by person and round."""
    m, q, k = c.constants, c.phase_size, len(c.traces)
    weights = numbers[2 * q : 2 * q + len(c.features)]
    features, derivatives = states(record, c, numbers[-k:] if k else np.empty(0))
    gradient = np.zeros(c.count)
    logs = np.empty(record.choice.shape)
    for first in (True, False):
        offset = 0 if first else q
        phase = numbers[offset : offset + q]
        repeats = 1 if first else ROUNDS - 1
        x = np.repeat(record.base, repeats, axis=0)
        f = features.reshape(-1, len(c.features)) if c.features else np.empty((len(x), 0))
        score = np.einsum("nm,m->n", x, phase[:m])
        if not first:
            score += np.einsum("nk,k->n", f, weights)
        choices = record.choice[:, :1] if first else record.choice[:, 1:]
        choice = choices.ravel()
        p, ds, dc = ordered(score, phase[-3:])
        rows = np.arange(len(choice))
        logp = np.log(p[rows, choice])
        g = -ds[rows, choice] / p[rows, choice]
        gradient[offset : offset + m] = np.einsum("nm,n->m", x, g)
        gradient[offset + m : offset + q] = (-dc[rows, choice] / p[rows, choice, None]).sum(axis=0)
        if first:
            logs[:, 0] = logp
        else:
            logs[:, 1:] = logp.reshape(-1, ROUNDS - 1)
            gradient[2 * q : 2 * q + len(c.features)] = np.einsum("nk,n->k", f, g)
            if k:
                effective = np.zeros((len(record.base), k))
                np.add.at(
                    effective, (np.arange(len(record.base))[:, None], record.sources), weights[:k]
                )
                gradient[-k:] = np.einsum(
                    "nr,nrk,nk->k", g.reshape(-1, ROUNDS - 1), derivatives, effective
                )
    loss = float(-logs.sum() + RIDGE * (numbers**2).sum())
    return loss, gradient + 2 * RIDGE * numbers, logs


def fit(groups: Sequence[Group], candidate: Candidate) -> Person:
    record = records(groups, candidate)
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * candidate.count
    for offset in (0, candidate.phase_size):
        for j in range(offset + candidate.constants, offset + candidate.phase_size):
            bounds[j] = (-LOG_GAP_LIMIT, LOG_GAP_LIMIT)
    for j in range(candidate.count - len(candidate.traces), candidate.count):
        bounds[j] = (0.0, 1.0)

    def objective(numbers: Floats) -> tuple[float, Floats]:
        loss, gradient, _ = likelihood(numbers, record, candidate)
        return loss, gradient

    result = minimize(
        objective,
        candidate.start(),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": FUNCTION_TOLERANCE,
            "gtol": TOLERANCE,
            "maxls": 50,
        },
    )
    if not result.success:
        raise RuntimeError(f"a person did not converge: {result.message}")
    return Person(candidate, np.asarray(result.x, dtype=np.float64))


def log_likelihoods(person: Person, groups: Sequence[Group]) -> Floats:
    logs = likelihood(person.numbers, records(groups, person.candidate), person.candidate)[2]
    # Public scores use the measuring stick's groups, rounds, seats order.
    result: list[float] = []
    row = 0
    for group in groups:
        size = GAMES[group.cell].size
        result.extend(logs[row : row + size].T.ravel().tolist())
        row += size
    return np.array(result)


def out_of_sample(
    groups: Sequence[Group], candidate: Candidate, leave_out_game: bool = False
) -> Floats:
    labels = [g.cell if leave_out_game else measure.fold_of(g) for g in groups]
    decision_group = np.repeat(
        np.arange(len(groups)), [ROUNDS * GAMES[g.cell].size for g in groups]
    )
    scores = np.empty(len(decision_group))
    for part in sorted(set(labels)):
        train = [g for g, label in zip(groups, labels, strict=True) if label != part]
        test = [g for g, label in zip(groups, labels, strict=True) if label == part]
        mask = np.array([labels[int(n)] == part for n in decision_group])
        scores[mask] = log_likelihoods(fit(train, candidate), test)
    return scores
