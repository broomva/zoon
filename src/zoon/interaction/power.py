"""Training-only power and coverage under the registered comparison rules."""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from zoon.interaction import compare, data, games, kernel, measure, person, registered, score
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES
from zoon.interaction.measure import Floats

TRIALS = 200
TRIAL_RESAMPLES = score.RESAMPLES
POOL_GROUPS = 2000
SEED = 20261120
HALF_EFFECT = 0.5
NULL_EFFECT = 0.0
TRAINING_EFFECT = 1.0
POOL_PREFIX = "power-pool"
POOL_REPLACEMENT = False
EXPECTED_H2 = ("0.0209", "0.0133", "0.0281")
EXPECTED_RESCORE = "0.0253"
TITLE = "zoon interaction: registered-rule power on training groups"
TRAINING_H2 = "[training, H2, out of fold] zoon minus alone"
TRAINING_RESCORE = "[training, H2, out of fold, person floor] zoon minus alone"
TRAINING_H3 = "[training, H3, in-sample laws] alone minus zoon"
RATE_LABELS = (
    "coverage at training effect",
    "pass at zero",
    "pass at half effect",
    "pass at training effect",
)

# Resolve attributes at rendering time, so no pasted setting can silently become stale.
SETTINGS: tuple[tuple[str, ModuleType, str], ...] = (
    ("seed", sys.modules[__name__], "SEED"),
    ("registered interval replicates", score, "RESAMPLES"),
    ("power trials", sys.modules[__name__], "TRIALS"),
    ("power interval replicates", sys.modules[__name__], "TRIAL_RESAMPLES"),
    ("pool groups per game and arm", sys.modules[__name__], "POOL_GROUPS"),
    ("pool prefix", sys.modules[__name__], "POOL_PREFIX"),
    ("pool draws with replacement", sys.modules[__name__], "POOL_REPLACEMENT"),
    ("simulated groups per game", score, "SIMULATED_GROUPS"),
    ("half ensemble", score, "HALF_ENSEMBLE"),
    ("human groups omitted per game", score, "HUMAN_DEFICIT"),
    ("interval percentiles", score, "PERCENTILES"),
    ("H2 threshold", score, "H2_THRESHOLD"),
    ("H3 threshold", score, "H3_THRESHOLD"),
    ("scoring floor", score, "SCORING_FLOOR"),
    ("rescoring floor", score, "RESCORING_FLOOR"),
    ("folds", measure, "FOLDS"),
    ("rounds", games, "ROUNDS"),
    ("null effect multiplier", sys.modules[__name__], "NULL_EFFECT"),
    ("half effect multiplier", sys.modules[__name__], "HALF_EFFECT"),
    ("training effect multiplier", sys.modules[__name__], "TRAINING_EFFECT"),
    ("registered simulation prefix", score, "SIMULATION_PREFIX"),
)


def settings() -> list[str]:
    return [f"[settings] {label} = {getattr(source, name)}" for label, source, name in SETTINGS]


def out_of_fold(groups: Sequence[Group], candidate: person.Candidate) -> score.Decisions:
    """The same folds and fit order as person.out_of_sample, retaining all bands."""
    folds = np.array([measure.fold_of(g) for g in groups])
    decisions = measure.decisions(groups)
    probabilities = np.empty((len(decisions.choice), measure.BANDS))
    for fold in sorted(set(folds.tolist())):
        train = [g for i, g in enumerate(groups) if folds[i] != fold]
        test = [g for i, g in enumerate(groups) if folds[i] == fold]
        probabilities[folds[decisions.group] == fold] = score.coded(
            person.fit(train, candidate), test
        ).probabilities
    return score.Decisions(decisions.group, decisions.choice, probabilities)


def h2_training(groups: Sequence[Group]) -> tuple[Floats, score.Interval, score.Interval]:
    zoon = out_of_fold(groups, registered.FORMS["zoon"])
    alone = out_of_fold(groups, registered.FORMS["alone"])
    common = score.group_means((zoon.logs() - alone.logs())[:, None], zoon.group, groups)
    original = score.group_means(
        (zoon.logs(person.FLOOR) - alone.logs(person.FLOOR))[:, None], zoon.group, groups
    )
    return common, score.h2_interval(common, groups), score.h2_interval(original, groups)


def checked_training_h2(result: score.Interval, rescore: score.Interval) -> None:
    quoted = tuple(f"{v:.4f}" for v in result.table()[0])
    if quoted != EXPECTED_H2 or f"{rescore.estimate[0]:.4f}" != EXPECTED_RESCORE:
        raise RuntimeError(f"registered training H2 check failed: {quoted}")


def sample_groups(
    groups: Sequence[Group], counts: Mapping[str, int], rng: np.random.Generator
) -> tuple[list[Group], measure.Ints]:
    cells = score.by_game(groups)
    if set(counts) != set(cells) or any(n <= score.HUMAN_DEFICIT for n in counts.values()):
        raise ValueError("held-out counts must cover the scored games with at least two each")
    indices = np.concatenate(
        [rng.choice(cells[cell], counts[cell], replace=True) for cell in cells]
    ).astype(np.int64)
    # Each resampled occurrence is a distinct group with a stable within-game index.
    trial = [
        Group(groups[int(i)].cell, str(n).zfill(len(str(len(indices)))), groups[int(i)].rounds)
        for n, i in enumerate(indices)
    ]
    return trial, indices


@dataclass(frozen=True)
class Rates:
    coverage: float
    null: float
    half: float
    full: float

    def values(self) -> tuple[float, ...]:
        return self.coverage, self.null, self.half, self.full


def trial_rates(
    estimates: Floats,
    lows: Floats,
    highs: Floats,
    truth: float,
    threshold: float,
    training_effect: float | None = None,
) -> Rates:
    """Location shifts preserve all game's residuals, interval widths and pairing."""
    if not len(estimates) or estimates.shape != lows.shape or lows.shape != highs.shape:
        raise ValueError("nonempty matched trial estimates and bounds are required")
    score.finite(estimates)
    score.finite(lows)
    score.finite(highs)
    score.finite(np.array([truth, threshold]))
    target = truth if training_effect is None else training_effect
    score.finite(np.array([target]))
    coverage = float(np.mean((lows <= truth) & (truth <= highs)))
    passed: list[float] = []
    for multiplier in (NULL_EFFECT, HALF_EFFECT, TRAINING_EFFECT):
        shift = multiplier * target - truth
        passed.append(float(np.mean((estimates + shift >= threshold) & (lows + shift > 0))))
    return Rates(coverage, *passed)


def h2_rates(
    groups: Sequence[Group],
    differences: Floats,
    counts: Mapping[str, int],
    *,
    trials: int = TRIALS,
    resamples: int = TRIAL_RESAMPLES,
) -> Rates:
    rng = np.random.default_rng(SEED)
    means = {cell: differences[mine].mean(axis=0) for cell, mine in score.by_game(groups).items()}
    residual = differences.copy()
    for cell, mine in score.by_game(groups).items():
        residual[mine] -= means[cell]
    estimates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    truth = None
    for _ in range(trials):
        trial, indices = sample_groups(groups, counts, rng)
        weights = score.h2_weights(trial)
        true = float(sum(weights[cell] * means[cell][0] for cell in weights))
        truth = true
        # Centre within game, then restore its own training mean. Other effect sizes
        # centre those means within game, then shift to the printed training effect.
        restored = residual[indices] + np.array([means[g.cell] for g in trial])
        interval = score.h2_interval(restored, trial, resamples)
        low, high = interval.bounds()[:, 0]
        estimates.append(interval.estimate[0])
        lows.append(low)
        highs.append(high)
    if truth is None:
        raise ValueError("at least one power trial is required")
    return trial_rates(
        np.array(estimates),
        np.array(lows),
        np.array(highs),
        truth,
        score.H2_THRESHOLD,
        training_effect=float(
            sum(score.h2_weights(groups)[cell] * mean[0] for cell, mean in means.items())
        ),
    )


def pool(
    laws: Mapping[str, person.Person], amounts: kernel.Amounts, count: int = POOL_GROUPS
) -> dict[str, dict[str, Floats]]:
    """Simulate once; distinct pool strings, shared group numbers across arms."""
    result: dict[str, dict[str, Floats]] = {}
    for arm, law in laws.items():
        result[arm] = {}
        for cell in sorted(GAMES):
            game = GAMES[cell]
            agents = [kernel.Coded(law) for _ in range(game.size)]
            values = np.array(
                [
                    compare.summaries(
                        kernel.play(
                            game,
                            agents,
                            amounts,
                            score.SEED,
                            f"{POOL_PREFIX}|{cell}|{n}",
                            open_channel=not law.candidate.twin,
                        ).rounds,
                        game,
                    )
                    for n in range(count)
                ]
            )
            score.validate_summaries(values, cell)
            result[arm][cell] = values
    return result


def pool_truth(
    groups: Sequence[Group],
    simulations: Mapping[str, Mapping[str, Floats]],
    counts: Mapping[str, int],
) -> float:
    # Fair CRPS on the whole finite pool is the expectation of fair CRPS for a
    # subset drawn without replacement. Human expectations use each game's training
    # distribution; weights use the registered held-out counts.
    trial, _ = sample_groups(groups, counts, np.random.default_rng(SEED))
    weights = score.h3_weights(trial)
    humans = score.observed(groups)
    scores: dict[str, Floats] = {}
    for arm, simulated in simulations.items():
        total = np.zeros(len(compare.SUMMARIES))
        for cell, human in humans.items():
            width = len(compare.SUMMARIES) - int(cell in score.UNDEFINED_GAMES)
            for s in range(width):
                mean = np.mean([compare.crps(simulated[cell][:, s], float(y)) for y in human[:, s]])
                total[s] += weights[cell][s] * mean
        scores[arm] = total
    return float((scores["alone"] - scores["zoon"]).mean())


def h3_rates(
    groups: Sequence[Group],
    simulations: Mapping[str, Mapping[str, Floats]],
    counts: Mapping[str, int],
    *,
    training_effect: float,
    trials: int = TRIALS,
    resamples: int = TRIAL_RESAMPLES,
) -> Rates:
    rng = np.random.default_rng(SEED)
    truth = pool_truth(groups, simulations, counts)
    estimates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(trials):
        trial, _ = sample_groups(groups, counts, rng)
        draws: dict[str, dict[str, Floats]] = {arm: {} for arm in simulations}
        for cell in score.by_game(groups):
            sizes = {len(simulated[cell]) for simulated in simulations.values()}
            if len(sizes) != 1:
                raise ValueError("paired arms require equal pool sizes")
            indices = rng.choice(sizes.pop(), score.SIMULATED_GROUPS, replace=POOL_REPLACEMENT)
            for arm in simulations:
                draws[arm][cell] = simulations[arm][cell][indices]
        intervals = score.h3_interval(trial, draws, resamples)
        result = score.difference(intervals["alone"], intervals["zoon"])
        low, high = result.bounds()[:, 0]
        estimates.append(result.estimate[0])
        lows.append(low)
        highs.append(high)
    return trial_rates(
        np.array(estimates),
        np.array(lows),
        np.array(highs),
        truth,
        score.H3_THRESHOLD,
        training_effect=training_effect,
    )


def formatted(label: str, interval: score.Interval) -> str:
    point, low, high = interval.table()[0]
    return f"{label} = {point:.4f} [{low:.4f}, {high:.4f}]"


def labels() -> list[str]:
    return [
        TITLE,
        *[line.split(" = ")[0] for line in settings()],
        "[data] training groups",
        "[data] held-out groups",
        TRAINING_H2,
        TRAINING_RESCORE,
        TRAINING_H3,
        *[f"[{hypothesis}] {label}" for hypothesis in ("H2", "H3") for label in RATE_LABELS],
    ]


def main() -> None:
    groups, counts = data.load(), data.held_out_counts()
    print(TITLE, flush=True)
    for line in settings():
        print(line, flush=True)
    print(f"[data] training groups = {len(groups)}", flush=True)
    print(f"[data] held-out groups = {sum(counts.values())}", flush=True)
    differences, h2, original = h2_training(groups)
    checked_training_h2(h2, original)
    print(formatted(TRAINING_H2, h2), flush=True)
    print(formatted(TRAINING_RESCORE, original), flush=True)
    laws, amounts = registered.read_fitted(), registered.read_amounts()
    simulations = {
        arm: score.simulate(
            kernel.Coded(law), amounts, open_channel=not law.candidate.twin
        ).summaries
        for arm, law in laws.items()
    }
    intervals = score.h3_interval(groups, simulations)
    h3 = score.difference(intervals["alone"], intervals["zoon"])
    # The printed row is composite only; each summary remains available to callers.
    print(formatted(TRAINING_H3, score.Interval(h3.estimate[:1], h3.values[:, :1])), flush=True)
    h2_power = h2_rates(groups, differences, counts)
    simulations_pool = pool(laws, amounts)
    h3_power = h3_rates(groups, simulations_pool, counts, training_effect=float(h3.estimate[0]))
    for hypothesis, rates in (("H2", h2_power), ("H3", h3_power)):
        for label, value in zip(RATE_LABELS, rates.values(), strict=True):
            print(f"[{hypothesis}] {label} = {value:.4f}", flush=True)


if __name__ == "__main__":
    main()
