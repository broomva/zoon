"""Compare coded persons on training groups, before registration."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from zoon.interaction import data, kernel, measure, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats, Ints

# Proposed registered number, fixed before the training comparison.
SIMULATED_GROUPS = 100
SUMMARIES = ("level", "members apart", "trend", "movement", "type gap")
METRICS = ("composite", *SUMMARIES)


@dataclass(frozen=True)
class Selection:
    reference: str
    tied: tuple[str, ...]  # includes the reference
    winner: str


def select(
    scores: Mapping[str, Floats],
    counts: Mapping[str, int],
    group: Ints,
    groups: int,
) -> Selection:
    """Apply the three steps to decision scores; exact score ties use letter order."""
    means = {name: float(score.mean()) for name, score in scores.items()}
    reference = min(scores, key=lambda name: (-means[name], name))
    tied: list[str] = [reference]
    for name in sorted(scores):
        if name == reference:
            continue
        _, low, high = measure.paired(scores[name] - scores[reference], group, groups)
        if low <= 0 <= high:
            tied.append(name)
    winner = min(tied, key=lambda name: (counts[name], -means[name], name))
    return Selection(reference, tuple(sorted(tied)), winner)


def summaries(rounds: Sequence[Sequence[int]], game: Game) -> Floats:
    """Five summaries from exact contribution shares; equal-seat games have no type gap."""
    shares = np.array(rounds, dtype=float) / np.array(game.endowments)
    level = shares.mean()
    apart = np.mean(
        [
            np.abs(shares[:, a] - shares[:, b]).mean()
            for a in range(game.size)
            for b in range(a + 1, game.size)
        ]
    )
    trend = shares[-5:].mean() - shares[:5].mean()
    movement = np.abs(np.diff(shares, axis=0)).mean()
    types = np.array([measure.seat_type(game, s) for s in range(game.size)])
    gap = shares[:, types == 0].mean() - shares[:, types == 1].mean() if types.any() else np.nan
    return np.array([level, apart, trend, movement, gap])


def crps(simulated: Floats, observed: float) -> float:
    """Fair CRPS: use distinct sample pairs, so its expectation is independent of count."""
    n = len(simulated)
    if n < 2 or not np.isfinite(simulated).all() or not np.isfinite(observed):
        raise ValueError("fair CRPS needs two finite draws and a finite observation")
    ordered = np.sort(simulated)
    pair_sum = float(((2 * np.arange(1, n + 1) - n - 1) * ordered).sum())
    return float(np.abs(ordered - observed).mean()) - pair_sum / (n * (n - 1))


def combine(values: Floats) -> Floats:
    """Mean per defined summary first, then equal weight for the five summaries."""
    means = np.nanmean(values, axis=0)
    return np.r_[means.mean(), means]


def interval(values: Floats, groups: Sequence[Group], resamples: int = measure.RESAMPLES) -> Floats:
    """Point estimates and group intervals, preserving each game's number of groups."""
    rng = np.random.default_rng(measure.SEED)
    indices = np.concatenate(
        [
            rng.choice(mine, (resamples, len(mine)))
            for cell in sorted({g.cell for g in groups})
            if len(mine := np.array([n for n, g in enumerate(groups) if g.cell == cell]))
        ],
        axis=1,
    )
    means = np.nanmean(values[indices], axis=1)
    draws = np.column_stack([means.mean(axis=1), means])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return np.column_stack([combine(values), low, high])


@dataclass(frozen=True)
class Evaluation:
    open: Floats
    closed: Floats
    covered: Floats
    fitted: person.Person


def evaluate(
    groups: Sequence[Group],
    candidate: person.Candidate,
    simulations: int = SIMULATED_GROUPS,
) -> Evaluation:
    """Fit and build the amount table outside each fold, in both loops."""
    if simulations < 2:
        raise ValueError("at least two simulated groups are needed")
    folds = np.array([measure.fold_of(g) for g in groups])
    decision_groups = np.repeat(
        np.arange(len(groups)), [ROUNDS * GAMES[g.cell].size for g in groups]
    )
    opened = np.empty(len(decision_groups))
    closed, covered = np.full((len(groups), 5), np.nan), np.full((len(groups), 5), np.nan)
    for fold in sorted(set(folds.tolist())):
        train = [g for n, g in enumerate(groups) if folds[n] != fold]
        test = [g for n, g in enumerate(groups) if folds[n] == fold]
        law = person.fit(train, candidate)
        opened[folds[decision_groups] == fold] = person.log_likelihoods(law, test)
        amounts = kernel.Amounts.build(train)
        for cell in sorted({g.cell for g in test}):
            game = GAMES[cell]
            simulated = np.array(
                [
                    summaries(
                        kernel.play(
                            game,
                            [kernel.Coded(law) for _ in range(game.size)],
                            amounts,
                            measure.SEED,
                            f"fold-{fold}|{cell}|{n}",
                            open_channel=not candidate.twin,
                        ).rounds,
                        game,
                    )
                    for n in range(simulations)
                ]
            )
            for n, group in enumerate(groups):
                if folds[n] != fold or group.cell != cell:
                    continue
                human = summaries(group.rounds, game)
                for j, observed in enumerate(human):
                    if not np.isfinite(observed):
                        continue
                    closed[n, j] = crps(simulated[:, j], float(observed))
                    low, high = np.percentile(simulated[:, j], [5, 95])
                    covered[n, j] = float(low <= observed <= high)
    return Evaluation(opened, closed, covered, person.fit(groups, candidate))


def fitted_lines(law: person.Person) -> list[str]:
    """Print every fitted number, with the equation's parameter names."""
    c = law.candidate
    base = (
        ("constant", "endowment", "return", "product", "four seats")
        if c.letter == "B"
        else tuple(f"constant {cell} type {kind}" for cell, kind in measure.STRATA)
    )
    names = [
        f"{phase} {name}"
        for phase in ("first", "later")
        for name in (*base, "cut origin", "log gap 1", "log gap 2", "log gap 3")
    ]
    names += [f"weight {name}" for name in c.features]
    names += [f"trace fraction {name}" for name in c.traces]
    return [
        f"[numbers, {c.name}] {name} = {number:+.6f}"
        for name, number in zip(names, law.numbers, strict=True)
    ]


def main(*, simulations: int = SIMULATED_GROUPS, resamples: int = measure.RESAMPLES) -> None:
    groups = data.load()
    held = data.held_out_counts()
    decisions = measure.decisions(groups)
    print("zoon interaction: candidates on training groups")
    print(f"[data] training groups = {len(groups)}")
    print(f"[data] decisions = {len(decisions.choice)}")
    print(f"[data] held-out groups = {sum(held.values())}")
    print(f"[simulation] groups per game and fold = {simulations}")
    print(f"[fit] ridge = {person.RIDGE}")
    print(f"[fit] probability floor = {person.FLOOR:.10f}")
    print(f"[fit] log gap limit = {person.LOG_GAP_LIMIT}")
    print(f"[fit] iteration limit = {person.MAX_ITERATIONS}")
    print(f"[fit] gradient tolerance = {person.TOLERANCE:.8f}")
    print("[fit] function tolerance = 0.0000000001")

    def found(label: str, result: Sequence[float]) -> None:
        mean, low, high = result
        print(f"{label} = {mean:+.4f} [{low:+.4f}, {high:+.4f}]")

    for (endowment, band), values in kernel.Amounts.build(groups).rows.items():
        counts = ", ".join(f"{points}:{count}" for points, count in sorted(Counter(values).items()))
        print(f"[amounts, endowment {endowment}, band {band}] points:count = {counts}")

    results: dict[str, Evaluation] = {}
    for c in person.CANDIDATES:
        # These twins have exactly the same equation, starting point and penalty.
        if c.twin and c.letter in ("C", "D"):
            a = results["A twin"]
            results[c.name] = Evaluation(
                a.open, a.closed, a.covered, person.Person(c, a.fitted.numbers)
            )
        else:
            results[c.name] = evaluate(groups, c, simulations)
        result = results[c.name]
        print(f"[count] {c.name} = {c.count}")
        print("\n".join(fitted_lines(result.fitted)))
        found(f"[open loop] {c.name}", measure.paired(result.open, decisions.group, len(groups)))
        for metric, value in zip(METRICS, interval(result.closed, groups, resamples), strict=True):
            found(f"[closed loop, {metric}] {c.name}", value)
        for j, summary in enumerate(SUMMARIES):
            print(f"[coverage, {summary}] {c.name} = {np.nanmean(result.covered[:, j]):.4f}")

    chosen: list[Selection] = []
    for twin in (False, True):
        candidates = [c for c in person.CANDIDATES if c.twin == twin]
        selection = select(
            {c.name: results[c.name].open for c in candidates},
            {c.name: c.count for c in candidates},
            decisions.group,
            len(groups),
        )
        chosen.append(selection)
        label = "twins" if twin else "candidates"
        print(f"[selection, {label}] highest = {selection.reference}")
        print(f"[selection, {label}] tied at step 2 = {', '.join(selection.tied)}")
        print(f"[selection, {label}] winner = {selection.winner}")
    winner, alone = chosen[0].winner, chosen[1].winner
    own = winner + " twin"
    for comparison, opponent in (("selected twin", alone), ("own twin", own)):
        a, b = results[winner], results[opponent]
        h2 = measure.paired(a.open - b.open, decisions.group, len(groups))
        found(f"[gain, {comparison}, open loop] candidate minus twin", h2)
        h3 = interval(b.closed - a.closed, groups, resamples)
        for metric, value in zip(METRICS, h3, strict=True):
            found(f"[gain, {comparison}, closed loop, {metric}] twin minus candidate", value)
        for metric, value in (("H2", h2), *zip(METRICS, h3, strict=True)):
            defined = {
                g.cell
                for g in groups
                if metric != "type gap"
                or any(measure.seat_type(GAMES[g.cell], s) == 1 for s in range(GAMES[g.cell].size))
            }
            training_count = sum(g.cell in defined for g in groups)
            sealed_count = sum(n for cell, n in held.items() if cell in defined)
            if value[2] == value[1]:
                full = half = float("nan")
            else:
                full, half = measure.power(
                    (float(value[0]), float(value[1]), float(value[2])),
                    training_count,
                    sealed_count,
                )
            print(f"[power, {comparison}, {metric}] training effect = {full:.3f}")
            print(f"[power, {comparison}, {metric}] half effect = {half:.3f}")
    for arm, name in (("candidate", winner), ("twin", alone)):
        c = next(c for c in person.CANDIDATES if c.name == name)
        left = person.out_of_sample(groups, c, leave_out_game=True)
        found(f"[game left out, {arm}] score", measure.paired(left, decisions.group, len(groups)))
        found(
            f"[game left out, {arm}] loss",
            measure.paired(results[name].open - left, decisions.group, len(groups)),
        )
        for cell in sorted(GAMES):
            mask = np.array([groups[int(g)].cell == cell for g in decisions.group])
            print(f"[game left out, {arm}, {cell}] score = {left[mask].mean():+.4f}")


if __name__ == "__main__":
    main()
