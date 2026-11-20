"""The training-group measurement: its pieces, a negative control and a positive control."""

import numpy as np
import pytest

from zoon.interaction import measure
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS

# "Within noise of zero" for the negative controls: nats per decision in the open loop and
# shares of the endowment in the closed loop. The positive controls must exceed several
# times these.
NOISE = 0.004


def _points(band: int, endowment: int) -> int:
    """A contribution that falls in the given band."""
    return (0, endowment // 6, endowment // 2, endowment * 5 // 6, endowment)[band]


def _groups(others: float, count: int, seed: int) -> list[Group]:
    """Synthetic groups of every game, played forward by a model whose weight on the others'
    last contributions is `others`. With zero the members never affect one another."""
    strata = np.linspace(-0.6, 0.6, len(measure.STRATA))
    weights = np.concatenate([strata, [0.3, 2.5, others], [-1.5, 0.0, 0.0, 0.0]])
    groups: list[Group] = []
    for number, cell in enumerate(sorted(GAMES)):
        game = GAMES[cell]
        played = measure.simulate(weights, game, True, count, seed + number)
        for g in range(count):
            rounds = tuple(
                tuple(_points(int(b), e) for b, e in zip(row, game.endowments, strict=True))
                for row in played[g]
            )
            groups.append(Group(cell, f"{cell}-{g}", rounds))
    return groups


def _gain(groups: list[Group]) -> tuple[float, float, float]:
    d = measure.decisions(groups)
    seen = {
        name: measure.out_of_sample(groups, name, leave_out_game=False)
        for name in ("seen, own history", "seen, own and others")
    }
    difference = seen["seen, own and others"] - seen["seen, own history"]
    return measure.paired(difference, d.group, len(groups))


def test_bands_cover_nothing_to_everything() -> None:
    assert [measure.band(c, 12) for c in (0, 1, 4, 5, 8, 9, 11, 12)] == [0, 1, 1, 2, 2, 3, 3, 4]
    assert [measure.band(c, 36) for c in (0, 12, 13, 24, 25, 35, 36)] == [0, 1, 2, 2, 3, 3, 4]
    for endowment in (12, 24, 36):
        assert [measure.band(_points(b, endowment), endowment) for b in range(5)] == [0, 1, 2, 3, 4]


def test_a_row_looks_back_one_round_and_never_at_itself() -> None:
    """In the first round a person has no history. Afterwards `own` is their own last band
    and `others` the mean of everyone else's, both as shares centred on one half."""
    game = GAMES["4-FE"]
    first, second = (24, 0, 12, 4), (0, 0, 0, 0)
    group = Group("4-FE", "g", (first, second, *[second] * (ROUNDS - 2)))
    d = measure.decisions([group])
    assert d.first[: game.size].tolist() == [1.0] * 4
    assert d.own[: game.size].tolist() == [0.0] * 4 and d.others[: game.size].tolist() == [0.0] * 4
    shares = [1.0, 0.0, 0.5, 1 / 6]
    assert d.own[game.size : 2 * game.size].tolist() == pytest.approx([s - 0.5 for s in shares])
    expected = [(sum(shares) - s) / 3 - 0.5 for s in shares]
    assert d.others[game.size : 2 * game.size].tolist() == pytest.approx(expected)
    assert d.choice[: game.size].tolist() == [4, 0, 2, 1]


def test_probabilities_are_a_distribution_that_moves_with_the_score() -> None:
    weights = np.array([1.0, -1.0, 0.2, -0.3, 0.1])
    p = measure.probabilities(weights, np.array([[-2.0], [0.0], [2.0]]))
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    mean_band = p @ np.arange(measure.BANDS)
    assert mean_band[0] < mean_band[1] < mean_band[2]


def test_the_score_rewards_a_sample_from_the_right_distribution() -> None:
    rng = np.random.default_rng(3)
    observed = rng.normal(0.0, 1.0, 400)
    right, shifted = rng.normal(0.0, 1.0, 2000), rng.normal(0.8, 1.0, 2000)
    narrow = rng.normal(0.0, 0.2, 2000)
    score = {
        name: np.mean([measure.crps(sample, float(y)) for y in observed])
        for name, sample in (("right", right), ("shifted", shifted), ("narrow", narrow))
    }
    assert score["right"] < score["shifted"] and score["right"] < score["narrow"]
    assert measure.crps(np.full(50, 0.3), 0.3) == pytest.approx(0.0)


def test_with_no_weight_on_others_the_group_does_not_interact() -> None:
    """NULL INTERACTION. A model that may look at the others but gives them no weight plays
    exactly the same games as the model that cannot look at them."""
    weights = np.concatenate([np.zeros(len(measure.STRATA)), [0.4, 2.0], [-1.5, 0.0, 0.0, 0.0]])
    social = np.insert(weights, len(measure.STRATA) + 2, 0.0)
    for cell in ("2-MI", "4-AI"):
        alone = measure.simulate(weights, GAMES[cell], False, 60, 5)
        together = measure.simulate(social, GAMES[cell], True, 60, 5)
        assert (alone == together).all()
        assert alone.shape == (60, ROUNDS, GAMES[cell].size)
        assert alone.min() >= 0 and alone.max() <= measure.BANDS - 1


def test_no_influence_in_the_data_gives_no_advantage() -> None:
    """NEGATIVE CONTROL. Groups simulated with members who ignore each other: seeing the
    others' history is worth nothing out of sample, in the open loop and the closed loop."""
    groups = _groups(others=0.0, count=40, seed=21)
    mean, _, _ = _gain(groups)
    assert abs(mean) < NOISE
    alone, together = measure.closed_loop(groups)
    assert abs(float((alone - together).mean())) < NOISE


def test_influence_in_the_data_is_found_in_both_loops() -> None:
    """POSITIVE CONTROL. Groups simulated with members who follow each other: the advantage
    is clearly positive in the open loop, and in the closed loop it shows where it should,
    in how far apart the members of a group are."""
    groups = _groups(others=2.0, count=40, seed=22)
    _, low, _ = _gain(groups)
    assert low > 5 * NOISE
    alone, together = measure.closed_loop(groups)
    apart = measure.SUMMARIES.index("members apart")
    _, low, _ = measure.interval(alone[:, apart] - together[:, apart])
    assert low > 2 * NOISE


def test_a_group_is_scored_by_a_model_that_never_saw_it() -> None:
    """Changing one group's own contributions does not change the model that scores it: the
    scores of every other fold's groups move, the group's own fold-mates' model does not."""
    groups = _groups(others=1.0, count=6, seed=23)
    folds = {measure.fold_of(group) for group in groups}
    assert folds == set(range(measure.FOLDS))
    target = 0
    changed = list(groups)
    flipped = tuple(tuple(0 for _ in row) for row in groups[target].rounds)
    changed[target] = Group(groups[target].cell, groups[target].index, flipped)
    name = "seen, own and others"
    before = measure.out_of_sample(groups, name, leave_out_game=False)
    after = measure.out_of_sample(changed, name, leave_out_game=False)
    d = measure.decisions(groups)
    same_fold = np.array(
        [measure.fold_of(groups[int(g)]) == measure.fold_of(groups[target]) for g in d.group]
    )
    others_in_fold = same_fold & (d.group != target)
    assert before[others_in_fold] == pytest.approx(after[others_in_fold])
    assert not np.allclose(before[~same_fold], after[~same_fold])
