"""What the others' history adds, measured on training groups only.

Four questions are answered before anything is designed or registered:

1. Open loop: how much better is a person's next contribution predicted when the model sees
   what the others in the group did last round?
2. A game left out: how much is lost when no group of the game being played was in the fit,
   and how much do the game's four numbers recover?
3. Closed loop: do simulated groups whose members respond to each other look more like human
   groups than simulated groups whose members do not?
4. Without any model: are the members of a human group closer to each other than the same
   people re-formed into groups at random, and were they already closer in the first round,
   before anyone had seen anyone else?

A gain in prediction is not by itself evidence that people influence each other. Members who
never looked at each other but shared a tendency from the start would produce one too. The
fourth question is the check on that: a shared tendency shows in the first round, influence
does not.

Every model here is an ordered logit over five bands of the share contributed. The models are
measuring sticks, not the person model that will be registered. Held-out groups are never
loaded: `data.load` returns training groups only.
"""

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from zoon.interaction import data
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

# Nothing, up to a third, up to two thirds, more but not all, everything.
BANDS = 5
# The share that stands for a band wherever a model looks back at an earlier contribution.
BAND_SHARE: tuple[float, ...] = (0.0, 1 / 6, 1 / 2, 5 / 6, 1.0)
FOLDS = 5
SEED = 20261120
RESAMPLES = 2000
SIMULATED_GROUPS = 400
REGROUPINGS = 500
# First round, early, middle and the last five rounds, as slices of the 20.
BLOCKS: tuple[tuple[str, slice], ...] = (
    ("round 1", slice(0, 1)),
    ("rounds 2 to 5", slice(1, 5)),
    ("rounds 6 to 15", slice(5, 15)),
    ("rounds 16 to 20", slice(15, 20)),
)


def band(points: int, endowment: int) -> int:
    """Return the band of one contribution."""
    if points == 0:
        return 0
    if points == endowment:
        return BANDS - 1
    share = Fraction(points, endowment)
    return 1 if share <= Fraction(1, 3) else 2 if share <= Fraction(2, 3) else 3


@dataclass(frozen=True)
class Decisions:
    """One row per training decision."""

    group: Ints  # position of the group in the list the rows were built from
    stratum: Ints  # the game and the type of seat, numbered over all games
    first: Floats  # 1.0 in the first round
    own: Floats  # the share standing for the person's own previous band, centred
    others: Floats  # the mean of the shares standing for the others' previous bands, centred
    traits: Floats  # the game's numbers as the person meets them, one row each
    choice: Ints  # the band chosen


def seat_type(game: Game, seat: int) -> int:
    """Return 0 for a seat of the first type and 1 for the second. Full equality has one."""
    first = (game.endowments[0], game.productivities[0])
    return 0 if (game.endowments[seat], game.productivities[seat]) == first else 1


STRATA: dict[tuple[str, int], int] = {
    key: number
    for number, key in enumerate(
        sorted({(g.cell, seat_type(g, s)) for g in GAMES.values() for s in range(g.size)})
    )
}


def traits(game: Game, seat: int) -> tuple[float, ...]:
    """The game's numbers from one seat: endowment and return relative to the equal game,
    whether the seat is the richer and the less productive one, and the group size."""
    endowment = game.endowments[seat] / 24 - 1
    returned = float(game.productivities[seat] / game.size) / 0.8 - 1
    return (endowment, returned, endowment * returned, float(game.size == 4))


def bands_of(group: Group) -> list[tuple[int, ...]]:
    game = GAMES[group.cell]
    return [
        tuple(band(c, e) for c, e in zip(row, game.endowments, strict=True)) for row in group.rounds
    ]


def decisions(groups: Sequence[Group]) -> Decisions:
    """Build one row per person and round from the groups' recorded contributions."""
    rows: list[tuple[int, int, float, float, float, tuple[float, ...], int]] = []
    for number, group in enumerate(groups):
        game = GAMES[group.cell]
        history = bands_of(group)
        for round_, chosen in enumerate(history):
            for seat in range(game.size):
                if round_ == 0:
                    own = others = 0.0
                else:
                    before = [BAND_SHARE[b] for b in history[round_ - 1]]
                    own = before[seat] - 0.5
                    others = (sum(before) - before[seat]) / (game.size - 1) - 0.5
                rows.append(
                    (
                        number,
                        STRATA[group.cell, seat_type(game, seat)],
                        float(round_ == 0),
                        own,
                        others,
                        traits(game, seat),
                        chosen[seat],
                    )
                )
    return Decisions(
        np.array([r[0] for r in rows], dtype=np.int64),
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([r[2] for r in rows]),
        np.array([r[3] for r in rows]),
        np.array([r[4] for r in rows]),
        np.array([r[5] for r in rows]),
        np.array([r[6] for r in rows], dtype=np.int64),
    )


Design = Callable[[Decisions], Floats]


def _strata(d: Decisions) -> Floats:
    return np.eye(len(STRATA))[d.stratum]


def _one(d: Decisions) -> Floats:
    return np.ones((len(d.choice), 1))


def _own(d: Decisions) -> Floats:
    return np.column_stack([d.first, d.own])


def _social(d: Decisions) -> Floats:
    return np.column_stack([d.first, d.own, d.others])


# What each model may look at. `per game` models have a constant for each game and seat type;
# `pooled` has one constant; `game numbers` has one constant and the game's four numbers.
DESIGNS: dict[str, Design] = {
    "per game, no history": _strata,
    "per game, own history": lambda d: np.column_stack([_strata(d), _own(d)]),
    "per game, own and others": lambda d: np.column_stack([_strata(d), _social(d)]),
    "pooled, own and others": lambda d: np.column_stack([_one(d), _social(d)]),
    "game numbers, own and others": lambda d: np.column_stack([_one(d), d.traits, _social(d)]),
}


def probabilities(weights: Floats, x: Floats) -> Floats:
    """Return each row's probability of each band under an ordered logit.

    The last four weights place the cut points: the first, then the logs of three gaps.
    """
    slopes, cuts = weights[:-4], np.cumsum(np.concatenate([weights[-4:-3], np.exp(weights[-3:])]))
    below = expit(cuts[None, :] - (x @ slopes)[:, None])
    edges = np.column_stack([np.zeros(len(x)), below, np.ones(len(x))])
    return np.clip(np.diff(edges, axis=1), 1e-9, 1.0)


def log_likelihoods(weights: Floats, x: Floats, choice: Ints) -> Floats:
    return np.log(probabilities(weights, x)[np.arange(len(choice)), choice])


def fit(x: Floats, choice: Ints) -> Floats:
    """Fit an ordered logit by maximum likelihood, with a small penalty to keep it finite."""
    start = np.concatenate([np.zeros(x.shape[1]), [-1.5], np.zeros(3)])

    def loss(weights: Floats) -> float:
        return float(-log_likelihoods(weights, x, choice).sum() + 0.01 * (weights**2).sum())

    return np.asarray(minimize(loss, start, method="L-BFGS-B").x, dtype=np.float64)


def fold_of(group: Group) -> int:
    """Assign a training group to one of five folds by a hash of its index."""
    return int(hashlib.sha256(f"{SEED}|{group.index}".encode()).hexdigest()[:8], 16) % FOLDS


def _take(d: Decisions, mask: NDArray[np.bool_]) -> Decisions:
    return Decisions(*(getattr(d, name)[mask] for name in d.__dataclass_fields__))


def out_of_sample(groups: Sequence[Group], name: str, leave_out_game: bool) -> Floats:
    """Return each decision's log-likelihood from a model fitted without its group.

    With `leave_out_game` the model is fitted on the other games only; otherwise on the
    other four folds of all games.
    """
    d, design = decisions(groups), DESIGNS[name]
    if leave_out_game:
        label = np.array([sorted(GAMES).index(groups[int(g)].cell) for g in d.group])
    else:
        label = np.array([fold_of(groups[int(g)]) for g in d.group])
    scores = np.empty(len(d.choice))
    for part in np.unique(label):
        test = label == part
        weights = fit(design(_take(d, ~test)), d.choice[~test])
        scores[test] = log_likelihoods(weights, design(_take(d, test)), d.choice[test])
    return scores


def per_group(values: Floats, group: Ints, groups: int) -> tuple[Floats, Floats]:
    """Return each group's sum of `values` and its number of decisions."""
    return np.bincount(group, values, groups), np.bincount(group, minlength=groups).astype(float)


def paired(difference: Floats, group: Ints, groups: int) -> tuple[float, float, float]:
    """Return the mean difference per decision and a 95% interval, resampling groups."""
    sums, counts = per_group(difference, group, groups)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, groups, size=(RESAMPLES, groups))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def simulate(weights: Floats, game: Game, social: bool, count: int, seed: int) -> Ints:
    """Play `count` groups of one game forward from the first round with a fitted model.

    Returns bands, groups by rounds by seats. With `social` false each person looks back at
    their own last contribution only, so the members of a group never affect one another.
    """
    rng = np.random.default_rng(seed)
    shares = np.array(BAND_SHARE)
    stratum = np.eye(len(STRATA))[[STRATA[game.cell, seat_type(game, s)] for s in range(game.size)]]
    played = np.empty((count, ROUNDS, game.size), dtype=np.int64)
    for round_ in range(ROUNDS):
        rows: list[Floats] = []
        for seat in range(game.size):
            if round_ == 0:
                state = np.tile([1.0, 0.0, 0.0], (count, 1))
            else:
                before = shares[played[:, round_ - 1, :]]
                own = before[:, seat] - 0.5
                others = (before.sum(axis=1) - before[:, seat]) / (game.size - 1) - 0.5
                state = np.column_stack([np.zeros(count), own, others])
            x = np.column_stack(
                [np.tile(stratum[seat], (count, 1)), state[:, : 3 if social else 2]]
            )
            rows.append(probabilities(weights, x))
        for seat, p in enumerate(rows):
            draw = rng.random(count)[:, None]
            played[:, round_, seat] = (
                (draw > np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1))
                .sum(axis=1)
                .clip(0, BANDS - 1)
            )
    return played


def summaries(played: Ints) -> Floats:
    """Describe each group by three numbers, all in shares of the endowment.

    The group's mean share over the session; how far apart its members are, as the mean
    absolute difference between two members' shares within a round; and how far the last
    five rounds are above the first five.
    """
    shares = np.array(BAND_SHARE)[played]
    level = shares.mean(axis=(1, 2))
    size = shares.shape[2]
    gaps = [
        np.abs(shares[:, :, a] - shares[:, :, b]).mean(axis=1)
        for a in range(size)
        for b in range(a + 1, size)
    ]
    apart = np.mean(gaps, axis=0)
    trend = shares[:, -5:, :].mean(axis=(1, 2)) - shares[:, :5, :].mean(axis=(1, 2))
    return np.column_stack([level, apart, trend])


SUMMARIES: tuple[str, ...] = ("level", "members apart", "trend")
HISTORY_MODELS: tuple[tuple[str, bool], ...] = (
    ("per game, own history", False),
    ("per game, own and others", True),
)


def crps(simulated: Floats, observed: float) -> float:
    """Score one observed number against a simulated sample; lower is closer.

    The continuous ranked probability score: the mean distance from the sample to the
    observation, less half the mean distance between two draws of the sample. It is smallest
    on average when the sample is drawn from the distribution the observation came from.
    """
    ordered = np.sort(simulated)
    n = len(ordered)
    spread = 2.0 * float(((2 * np.arange(1, n + 1) - n - 1) * ordered).sum()) / n**2
    return float(np.abs(ordered - observed).mean()) - 0.5 * spread


def closed_loop(groups: Sequence[Group]) -> tuple[Floats, Floats]:
    """Score every training group against groups simulated without it.

    Returns one array per history model, groups by summaries. Each human group's three
    summaries are scored against the summaries of groups simulated for its game by a model
    fitted on the other four folds.
    """
    d = decisions(groups)
    fold = np.array([fold_of(group) for group in groups])
    scores = [np.empty((len(groups), len(SUMMARIES))) for _ in HISTORY_MODELS]
    for part in range(FOLDS):
        train = np.isin(d.group, np.flatnonzero(fold != part))
        for which, (name, social) in enumerate(HISTORY_MODELS):
            weights = fit(DESIGNS[name](_take(d, train)), d.choice[train])
            for cell in sorted(GAMES):
                mine = [n for n, g in enumerate(groups) if fold[n] == part and g.cell == cell]
                if not mine:
                    continue
                played = simulate(weights, GAMES[cell], social, SIMULATED_GROUPS, SEED + part)
                simulated = summaries(played)
                human = summaries(np.array([bands_of(groups[n]) for n in mine]))
                for row, n in enumerate(mine):
                    scores[which][n] = [
                        crps(simulated[:, column], float(human[row, column]))
                        for column in range(len(SUMMARIES))
                    ]
    return scores[0], scores[1]


def interval(values: Floats) -> tuple[float, float, float]:
    """Return the mean over groups and a 95% interval, resampling groups."""
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def power(found: tuple[float, float, float], training: int, sealed: int) -> tuple[float, float]:
    """Return the chance that the held-out groups detect an effect, at its training size and
    at half of it.

    A normal approximation with three assumptions: the held-out groups are as variable as
    the training groups, so the standard error read off the training interval only needs
    scaling by the two counts; the scoring models are treated as fixed; and the true effect
    is the size given. Detecting means a two-sided 95% interval that excludes zero, on either
    side.
    """
    mean, low, high = found
    error = (high - low) / (2 * 1.96) * (training / sealed) ** 0.5

    def below(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def detected(effect: float) -> float:
        return below(effect / error - 1.96) + below(-effect / error - 1.96)

    return detected(mean), detected(mean / 2)


def apart(shares: Floats) -> Floats:
    """Return, for each block of rounds, how far apart the members of a group are.

    `shares` is groups by rounds by seats. The result is the mean over groups, rounds and
    pairs of members of the absolute difference between two members' shares. It is a mean
    per round, so a block of one round and a block of ten estimate the same kind of quantity;
    the one-round block is the noisier estimate.
    """
    size = shares.shape[2]
    gaps = np.mean(
        [
            np.abs(shares[:, :, a] - shares[:, :, b])
            for a in range(size)
            for b in range(a + 1, size)
        ],
        axis=0,
    )
    return np.array([gaps[:, rounds].mean() for _, rounds in BLOCKS])


def against_chance(groups: Sequence[Group]) -> tuple[Floats, Floats]:
    """Compare human groups with the same people re-formed into groups at random.

    Returns how far apart members are in each block of rounds, for the real groups and for
    each of `REGROUPINGS` re-formings. A re-forming keeps every person's whole session and
    seat, and deals the people of one game and seat to groups at random. Games are weighted
    by their number of groups. Shares are the contributions as recorded, divided by the
    endowment; no bands are involved.

    What it can and cannot show. If the people of a game and seat are exchangeable when the
    groups are formed, a re-forming is a draw from what the groups would look like had their
    members never met. A tendency shared from the start makes real members closer by the
    same expected amount in every round. Members who respond to each other start no closer
    than chance and become closer. So the comparison is between the first round's excess and
    the later ones. It cannot rule out a shared tendency smaller than the first round's range
    of chance, and it does not say what kind of response draws members together.
    """
    rng = np.random.default_rng(SEED)
    real = np.zeros(len(BLOCKS))
    chance = np.zeros((REGROUPINGS, len(BLOCKS)))
    for cell in sorted(GAMES):
        mine = [group for group in groups if group.cell == cell]
        endowments = np.array(GAMES[cell].endowments, dtype=np.float64)
        shares = np.array([group.rounds for group in mine], dtype=np.float64) / endowments
        weight = len(mine) / len(groups)
        real += weight * apart(shares)
        for draw in range(REGROUPINGS):
            dealt = np.stack(
                [shares[rng.permutation(len(mine)), :, seat] for seat in range(shares.shape[2])],
                axis=2,
            )
            chance[draw] += weight * apart(dealt)
    return real, chance


def closer_than_chance(real: Floats, chance: Floats) -> tuple[Floats, Floats, Floats]:
    """Return, for each block, how much closer real members are than re-formed ones: the
    mean over re-formings and the ends of its 95% range."""
    closer = chance - real
    low, high = np.percentile(closer, [2.5, 97.5], axis=0)
    return closer.mean(axis=0), low, high


def main() -> None:
    groups = data.load()
    d = decisions(groups)
    count = len(groups)
    print("zoon interaction: what the others' history adds, on training groups")
    print(f"[data] training groups = {count}; decisions = {len(d.choice)}")
    shares = np.bincount(d.choice, minlength=BANDS) / len(d.choice)
    print("[bands] share of decisions = " + ", ".join(f"{s:.3f}" for s in shares))

    def gain(label: str, difference: Floats, group: Ints) -> tuple[float, float, float]:
        found = paired(difference, group, count)
        print(f"{label} = {found[0]:+.4f} [{found[1]:+.4f}, {found[2]:+.4f}]")
        return found

    left = {name: out_of_sample(groups, name, leave_out_game=False) for name in DESIGNS}
    for name, scores in left.items():
        print(f"[open loop, groups left out] {name} = {scores.mean():+.4f} nats per decision")
    own, both = left["per game, own history"], left["per game, own and others"]
    gain("[open loop] own history adds", own - left["per game, no history"], d.group)
    others_add = gain("[open loop] others' history adds", both - own, d.group)
    later = d.first == 0
    gain("[open loop, rounds 2 to 20] others' history adds", (both - own)[later], d.group[later])

    pooled, numbers = "pooled, own and others", "game numbers, own and others"
    out = {name: out_of_sample(groups, name, leave_out_game=True) for name in (pooled, numbers)}
    for name, scores in out.items():
        print(f"[open loop, game left out] {name} = {scores.mean():+.4f} nats per decision")
    gain("[game left out] a constant per game adds", both - left[pooled], d.group)
    gain("[game left out] having the game in the fit adds", left[pooled] - out[pooled], d.group)
    gain("[game left out] both together", both - out[pooled], d.group)
    gain("[game left out] the game's numbers recover", out[numbers] - out[pooled], d.group)
    first = d.first == 1
    gain("[game left out, round 1 only] both together", (both - out[pooled])[first], d.group[first])

    alone, together = closed_loop(groups)
    for column, summary in enumerate(SUMMARIES):
        for name, scores in (("own history", alone), ("own and others", together)):
            print(f"[closed loop, {summary}] {name} = {scores[:, column].mean():.4f}")
        mean, low, high = interval(alone[:, column] - together[:, column])
        found = f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"
        print(f"[closed loop, {summary}] others' history gains = {found}")
    loop_gain = interval((alone - together).mean(axis=1))
    mean, low, high = loop_gain
    print(f"[closed loop, all three] others' history gains = {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
    fours = np.array([GAMES[group.cell].size == 4 for group in groups])
    for label, mask in (("pairs", ~fours), ("groups of four", fours)):
        mean, low, high = interval((alone - together).mean(axis=1)[mask])
        print(
            f"[closed loop, {label}] others' history gains = {mean:+.4f} [{low:+.4f}, {high:+.4f}]"
        )

    real, chance = against_chance(groups)
    closer, closer_low, closer_high = closer_than_chance(real, chance)
    for column, (label, _) in enumerate(BLOCKS):
        low, high = np.percentile(chance[:, column], [2.5, 97.5])
        dealt = f"{chance[:, column].mean():.4f} [{low:.4f}, {high:.4f}]"
        print(f"[members apart, {label}] human groups = {real[column]:.4f}")
        print(f"[members apart, {label}] re-formed at random = {dealt}")
        excess = f"{closer[column]:+.4f} [{closer_low[column]:+.4f}, {closer_high[column]:+.4f}]"
        print(f"[members apart, {label}] closer than chance by = {excess}")

    sealed = sum(data.held_out_counts().values())
    print(f"[power] held-out groups = {sealed}")
    for label, found in (("open loop", others_add), ("closed loop", loop_gain)):
        full, half = power(found, count, sealed)
        print(f"[power] {label}: at the training effect = {full:.3f}; at half of it = {half:.3f}")


if __name__ == "__main__":
    main()
