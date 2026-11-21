"""The registered scores and intervals. All data and laws are explicit arguments."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from zoon.interaction import compare, kernel, measure, person, prompted
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS
from zoon.interaction.measure import Floats, Ints

SEED = 20261120
RESAMPLES = 10_000
SIMULATED_GROUPS = 100
HALF_ENSEMBLE = 50
HUMAN_DEFICIT = 1
PERCENTILES = (2.5, 97.5)
H2_THRESHOLD = 0.0052
H3_THRESHOLD = 0.0031
SCORING_FLOOR = prompted.PROMPTED_FLOOR
RESCORING_FLOOR = person.FLOOR
SIMULATION_PREFIX = "registered"
UNDEFINED_GAMES = frozenset(("2-FE", "4-FE"))
Verdict = Literal["passed", "failed", "no verdict"]


def finite(values: Floats) -> None:
    if not np.isfinite(values).all():
        raise ValueError("non-finite probability, score or defined summary")


def log_scores(probabilities: Floats, chosen: Ints, floor: float = SCORING_FLOOR) -> Floats:
    finite(probabilities)
    if (
        probabilities.shape != (len(chosen), measure.BANDS)
        or (probabilities < 0).any()
        or (probabilities.sum(axis=1) <= 0).any()
        or not np.isfinite(floor)
        or floor <= 0
        or ((chosen < 0) | (chosen >= measure.BANDS)).any()
    ):
        raise ValueError("invalid band distribution, choice or floor")
    raised = np.maximum(probabilities, floor)
    totals = raised.sum(axis=1)
    finite(totals)
    result = np.log(raised[np.arange(len(chosen)), chosen] / totals)
    finite(result)
    return result


def below_floor(probabilities: Floats, chosen: Ints) -> int:
    log_scores(probabilities, chosen)
    return int(np.count_nonzero(probabilities[np.arange(len(chosen)), chosen] < SCORING_FLOOR))


@dataclass(frozen=True)
class Decisions:
    group: Ints
    chosen: Ints
    probabilities: Floats

    def logs(self, floor: float = SCORING_FLOOR) -> Floats:
        return log_scores(self.probabilities, self.chosen, floor)

    @property
    def raised_choices(self) -> int:
        return below_floor(self.probabilities, self.chosen)


def coded(law: person.Person, groups: Sequence[Group]) -> Decisions:
    """Group, round, seat order; every update sees only real earlier rounds."""
    identities: list[int] = []
    choices: list[int] = []
    probabilities: list[Floats] = []
    for i, group in enumerate(groups):
        game = GAMES[group.cell]
        states = [person.State() for _ in range(game.size)]
        for row in group.rounds:
            for seat, state in enumerate(states):
                probabilities.append(law.probabilities(game, seat, state))
                choices.append(measure.band(row[seat], game.endowments[seat]))
                identities.append(i)
            states = [law.update(state, game, seat, row) for seat, state in enumerate(states)]
    result = Decisions(np.array(identities), np.array(choices), np.array(probabilities))
    result.logs()
    return result


def group_means(values: Floats, identities: Ints, groups: Sequence[Group]) -> Floats:
    finite(values)
    if len(values) != len(identities) or values.ndim != 2:
        raise ValueError("one row of comparisons per decision is needed")
    expected = np.array([ROUNDS * GAMES[g.cell].size for g in groups])
    counts = np.bincount(identities, minlength=len(groups))
    if not np.array_equal(counts, expected):
        raise ValueError("every decision of every group is required")
    return np.array([values[identities == i].mean(axis=0) for i in range(len(groups))])


def by_game(groups: Sequence[Group]) -> dict[str, Ints]:
    if not groups or any(g.cell not in GAMES for g in groups):
        raise ValueError("known games and nonempty groups are required")
    result = {
        cell: np.array(sorted(mine, key=lambda i: groups[i].index), dtype=np.int64)
        for cell in sorted(GAMES)
        if (mine := [i for i, g in enumerate(groups) if g.cell == cell])
    }
    return result


def h2_weights(groups: Sequence[Group]) -> dict[str, float]:
    cells = by_game(groups)
    counts = {cell: len(mine) * ROUNDS * GAMES[cell].size for cell, mine in cells.items()}
    return {cell: count / sum(counts.values()) for cell, count in counts.items()}


@dataclass(frozen=True)
class Interval:
    estimate: Floats
    values: Floats

    def bounds(self) -> Floats:
        finite(self.estimate)
        finite(self.values)
        return np.asarray(np.percentile(self.values, PERCENTILES, axis=0), dtype=float)

    def table(self) -> Floats:
        return np.stack((self.estimate, *self.bounds()), axis=-1)


def h2_interval(
    differences: Floats, groups: Sequence[Group], resamples: int = RESAMPLES
) -> Interval:
    """Rows are group mean differences, columns are paired comparisons."""
    finite(differences)
    if differences.ndim != 2 or len(differences) != len(groups) or resamples <= 0:
        raise ValueError("one row per group and positive replicates are required")
    rng = np.random.default_rng(SEED)
    weights = h2_weights(groups)
    values = np.zeros((resamples, differences.shape[1]))
    estimate = np.zeros(differences.shape[1])
    for cell, mine in by_game(groups).items():
        n_g = len(mine)
        if n_g <= HUMAN_DEFICIT:
            raise ValueError("each resampled game needs at least two groups")
        idx = rng.integers(0, n_g, size=(resamples, n_g - HUMAN_DEFICIT))
        values += weights[cell] * differences[mine][idx].mean(axis=1)
        estimate += weights[cell] * differences[mine].mean(axis=0)
    finite(values)
    return Interval(estimate, values)


def comparison_verdict(estimate: float, lower: float, threshold: float) -> Verdict:
    finite(np.array([estimate, lower, threshold]))
    return "passed" if estimate >= threshold and lower > 0 else "failed"


def hypothesis_verdict(first: Verdict | None, second: Verdict | None) -> Verdict:
    if first in (None, "no verdict") or second in (None, "no verdict"):
        return "no verdict"
    return "passed" if first == second == "passed" else "failed"


@dataclass(frozen=True)
class Simulation:
    logs: Mapping[str, tuple[kernel.Run, ...]]
    summaries: Mapping[str, Floats]


def simulate(
    agent: kernel.Agent, amounts: kernel.Amounts, *, open_channel: bool = True
) -> Simulation:
    logs: dict[str, tuple[kernel.Run, ...]] = {}
    values: dict[str, Floats] = {}
    for cell in sorted(GAMES):
        game = GAMES[cell]
        runs = tuple(
            kernel.play(
                game,
                [agent] * game.size,
                amounts,
                SEED,
                f"{SIMULATION_PREFIX}|{cell}|{n}",
                open_channel=open_channel,
            )
            for n in range(SIMULATED_GROUPS)
        )
        logs[cell] = runs
        values[cell] = np.array([compare.summaries(run.rounds, game) for run in runs])
        validate_summaries(values[cell], cell)
    return Simulation(logs, values)


def validate_summaries(values: Floats, cell: str) -> None:
    if values.ndim != 2 or values.shape[1] != len(compare.SUMMARIES):
        raise ValueError("five summaries are required")
    if cell in UNDEFINED_GAMES:
        finite(values[:, :-1])
        if not np.isnan(values[:, -1]).all():
            raise ValueError("type gap must be undefined in equal-seat games")
    else:
        finite(values)


def observed(groups: Sequence[Group]) -> dict[str, Floats]:
    return {
        cell: np.array([compare.summaries(groups[int(i)].rounds, GAMES[cell]) for i in mine])
        for cell, mine in by_game(groups).items()
    }


def h3_weights(groups: Sequence[Group]) -> dict[str, Floats]:
    cells = by_game(groups)
    defined = {
        cell: np.array([True] * (len(compare.SUMMARIES) - 1) + [cell not in UNDEFINED_GAMES])
        for cell in cells
    }
    totals = sum(len(cells[cell]) * mask for cell, mask in defined.items())
    if np.any(totals == 0):
        raise ValueError("each summary needs defined groups")
    return {cell: len(cells[cell]) * mask / totals for cell, mask in defined.items()}


@dataclass(frozen=True)
class Distances:
    human: Floats  # observed group, simulated group, defined summary
    simulated: Floats  # simulated group, simulated group, defined summary

    def full(self) -> Floats:
        n = self.simulated.shape[0]
        result = self.human.mean(axis=1) - self.simulated.sum(axis=(0, 1)) / (2 * n * (n - 1))
        finite(result)
        return result

    def draws(self, human: Ints, half: Ints) -> Floats:
        """Exact subset sums; no per-replicate sorting or score approximation."""
        count = len(human)
        humans = np.zeros((count, self.human.shape[0]))
        np.add.at(humans, (np.arange(count)[:, None], human), 1)
        halves = np.zeros((count, self.simulated.shape[0]))
        halves[np.arange(count)[:, None], half] = 1
        k = half.shape[1]
        # Contiguous summary slices let matrix multiplication use its fast path.
        # These are the same subset sums as the literal pairwise definition.
        cross = np.column_stack(
            [
                ((humans @ np.ascontiguousarray(self.human[:, :, s])) * halves).sum(axis=1)
                for s in range(self.human.shape[-1])
            ]
        )
        pairs = np.column_stack(
            [
                ((halves @ np.ascontiguousarray(self.simulated[:, :, s])) * halves).sum(axis=1)
                for s in range(self.simulated.shape[-1])
            ]
        )
        result = cross / (human.shape[1] * k) - pairs / (2 * k * (k - 1))
        finite(result)
        return result


def distances(human: Floats, simulated: Floats, cell: str) -> Distances:
    validate_summaries(human, cell)
    validate_summaries(simulated, cell)
    if len(simulated) < 2:
        raise ValueError("fair scores need two simulated values")
    if cell in UNDEFINED_GAMES:
        human, simulated = human[:, :-1], simulated[:, :-1]
    return Distances(
        np.abs(human[:, None, :] - simulated[None, :, :]),
        np.abs(simulated[:, None, :] - simulated[None, :, :]),
    )


def fair_scores(groups: Sequence[Group], simulated: Mapping[str, Floats]) -> Floats:
    values = np.full((len(groups), len(compare.SUMMARIES)), np.nan)
    for cell, mine in by_game(groups).items():
        human = np.array([compare.summaries(groups[int(i)].rounds, GAMES[cell]) for i in mine])
        result = distances(human, simulated[cell], cell).full()
        values[mine, : result.shape[1]] = result
    return values


def h3_interval(
    groups: Sequence[Group], arms: Mapping[str, Mapping[str, Floats]], resamples: int = RESAMPLES
) -> dict[str, Interval]:
    """Arm scores, composite then summaries; subtract arms to form paired differences."""
    if not arms or resamples <= 0:
        raise ValueError("arms and positive replicates are required")
    humans = observed(groups)
    weights = h3_weights(groups)
    rng = np.random.default_rng(SEED)
    values = {arm: np.zeros((resamples, len(compare.SUMMARIES))) for arm in arms}
    points = {arm: np.zeros(len(compare.SUMMARIES)) for arm in arms}
    for cell, human_values in humans.items():
        n_g = len(human_values)
        if n_g <= HUMAN_DEFICIT:
            raise ValueError("each resampled game needs at least two groups")
        human = rng.integers(0, n_g, size=(resamples, n_g - HUMAN_DEFICIT))
        half = np.array(
            [rng.permutation(SIMULATED_GROUPS)[:HALF_ENSEMBLE] for _ in range(resamples)]
        )
        for arm, simulations in arms.items():
            if simulations[cell].shape != (SIMULATED_GROUPS, len(compare.SUMMARIES)):
                raise ValueError("the registered ensemble size is required")
            matrices = distances(human_values, simulations[cell], cell)
            width = matrices.human.shape[-1]
            values[arm][:, :width] += weights[cell][:width] * matrices.draws(human, half)
            points[arm][:width] += weights[cell][:width] * matrices.full().mean(axis=0)
    result: dict[str, Interval] = {}
    for arm in arms:
        finite(points[arm])
        finite(values[arm])
        result[arm] = Interval(
            np.r_[points[arm].mean(), points[arm]],
            np.column_stack((values[arm].mean(axis=1), values[arm])),
        )
    return result


def difference(other: Interval, zoon: Interval) -> Interval:
    result = Interval(other.estimate - zoon.estimate, other.values - zoon.values)
    finite(result.estimate)
    finite(result.values)
    return result
