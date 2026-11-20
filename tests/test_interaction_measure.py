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
        for name in ("per game, own history", "per game, own and others")
    }
    difference = seen["per game, own and others"] - seen["per game, own history"]
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


def _shared_tendency(count: int, seed: int) -> list[Group]:
    """Synthetic groups whose members never look at each other but share a tendency from the
    start: every member of a group gives mostly a lot, or mostly little."""
    rng = np.random.default_rng(seed)
    groups: list[Group] = []
    for cell in sorted(GAMES):
        game = GAMES[cell]
        for g in range(count):
            generous = rng.random() < 0.5
            weights = [0.05, 0.05, 0.1, 0.2, 0.6] if generous else [0.6, 0.2, 0.1, 0.05, 0.05]
            rounds = tuple(
                tuple(_points(int(rng.choice(5, p=weights)), e) for e in game.endowments)
                for _ in range(ROUNDS)
            )
            groups.append(Group(cell, f"{cell}-{g}", rounds))
    return groups


def _static(count: int, seed: int, low: float, high: float) -> list[Group]:
    """Synthetic groups with a tendency that is shared and never changes: in every round each
    member gives everything with the group's probability, `low` or `high`, and nothing
    otherwise. Nobody looks at anybody."""
    rng = np.random.default_rng(seed)
    groups: list[Group] = []
    for cell in sorted(GAMES):
        game = GAMES[cell]
        for g in range(count):
            chance = high if rng.random() < 0.5 else low
            rounds = tuple(
                tuple(e if rng.random() < chance else 0 for e in game.endowments)
                for _ in range(ROUNDS)
            )
            groups.append(Group(cell, f"{cell}-{g}", rounds))
    return groups


def _closer(groups: list[Group]) -> tuple[list[float], list[float], list[float]]:
    mean, low, high = measure.closer_than_chance(*measure.against_chance(groups))
    return mean.tolist(), low.tolist(), high.tolist()


def _grows(groups: list[Group]) -> bool:
    """Is every later block's range of closeness above the first round's range?"""
    _, low, high = _closer(groups)
    return min(low[1:]) > high[0]


def test_influence_appears_only_after_the_first_round() -> None:
    """POSITIVE CONTROL for the check without a model. Members who follow each other are no
    closer than chance in the first round, and closer in every later block."""
    groups = _groups(others=2.0, count=40, seed=31)
    _, low, high = _closer(groups)
    assert low[0] < 0 < high[0]
    assert _grows(groups)


def test_members_who_ignore_each_other_are_never_closer_than_chance() -> None:
    """NEGATIVE CONTROL for the check without a model."""
    groups = _groups(others=0.0, count=40, seed=32)
    _, low, high = _closer(groups)
    assert all(lower < 0 < upper for lower, upper in zip(low, high, strict=True))
    assert not _grows(groups)


def test_a_strong_shared_tendency_is_caught_in_the_first_round() -> None:
    """THE CONFOUND. Members who share a tendency but never interact fool the model
    comparison: seeing the others' history looks valuable. The check without a model catches
    it, because they are already closer than chance in the first round, and no closer later."""
    groups = _shared_tendency(count=40, seed=33)
    _, gain_low, _ = _gain(groups)
    assert gain_low > 5 * NOISE, "the model comparison alone would call this influence"
    _, low, _ = _closer(groups)
    assert low[0] > 0
    assert not _grows(groups)


def test_a_weak_shared_tendency_passes_the_first_round_unnoticed() -> None:
    """A LIMIT of the check. A weak shared tendency is too small to stand out in one round:
    the first round's range includes zero, as it would with no shared tendency at all."""
    groups = _static(count=40, seed=1, low=0.35, high=0.65)
    _, low, high = _closer(groups)
    assert low[0] < 0 < high[0]
    assert low[2] > 0, "ten rounds can see what one round cannot"


def test_a_shared_target_that_shows_only_later_gives_the_same_table_as_influence() -> None:
    """THE LIMIT that matters. Every group has a target share, a third or two thirds, fixed
    from the start. In the first round each member gives everything or nothing by a coin
    flip that ignores the target; afterwards each gives exactly the target. Nobody looks at
    anybody. The table has the shape of the human one: the first round's range includes
    zero and every later block is far above it. So the check describes when closeness
    arises and cannot say why."""
    rng = np.random.default_rng(9)
    groups: list[Group] = []
    for cell in sorted(GAMES):
        game = GAMES[cell]
        for g in range(40):
            third = 1 if rng.random() < 0.5 else 2
            settled = tuple(e * third // 3 for e in game.endowments)
            first = tuple(e if rng.random() < 0.5 else 0 for e in game.endowments)
            groups.append(Group(cell, f"{cell}-{g}", (first, *[settled] * (ROUNDS - 1))))
    _, low, high = _closer(groups)
    assert low[0] < 0 < high[0]
    assert _grows(groups)


def test_the_check_uses_contributions_not_bands() -> None:
    """Members of a group all give a twelfth of their points, or all give a third. Both fall
    in the same band, so a check on bands would see nothing. On the recorded contributions
    the members are closer than chance in every block."""
    rng = np.random.default_rng(5)
    groups: list[Group] = []
    for cell in sorted(GAMES):
        game = GAMES[cell]
        for g in range(30):
            part = 12 if rng.random() < 0.5 else 3
            row = tuple(e // part for e in game.endowments)
            assert len({measure.band(c, e) for c, e in zip(row, game.endowments, strict=True)}) == 1
            groups.append(Group(cell, f"{cell}-{g}", (row,) * ROUNDS))
    _, low, _ = _closer(groups)
    assert min(low) > 0


def test_power_counts_both_tails() -> None:
    """A strong effect is detected whichever way it points, and no effect is detected at
    the rate of the interval's error."""
    strong, _ = measure.power((0.05, 0.04, 0.06), 200, 200)
    reverse, _ = measure.power((-0.05, -0.06, -0.04), 200, 200)
    none, _ = measure.power((0.0, -0.01, 0.01), 200, 200)
    assert strong == pytest.approx(1.0) and reverse == pytest.approx(1.0)
    assert none == pytest.approx(0.05, abs=0.001)


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
    name = "per game, own and others"
    before = measure.out_of_sample(groups, name, leave_out_game=False)
    after = measure.out_of_sample(changed, name, leave_out_game=False)
    d = measure.decisions(groups)
    same_fold = np.array(
        [measure.fold_of(groups[int(g)]) == measure.fold_of(groups[target]) for g in d.group]
    )
    others_in_fold = same_fold & (d.group != target)
    assert before[others_in_fold] == pytest.approx(after[others_in_fold])
    assert not np.allclose(before[~same_fold], after[~same_fold])


def _zeroed(groups: list[Group], target: int) -> list[Group]:
    changed = list(groups)
    nothing = tuple(tuple(0 for _ in row) for row in groups[target].rounds)
    changed[target] = Group(groups[target].cell, groups[target].index, nothing)
    return changed


def test_a_left_out_game_is_scored_by_a_model_that_saw_none_of_it() -> None:
    """Changing one group changes nothing for the other groups of its game when the game is
    left out of the fit, and changes the scores of the other games, whose fits include it."""
    groups = _groups(others=1.0, count=6, seed=24)
    name = "pooled, own and others"
    before = measure.out_of_sample(groups, name, leave_out_game=True)
    after = measure.out_of_sample(_zeroed(groups, 0), name, leave_out_game=True)
    d = measure.decisions(groups)
    same_game = np.array([groups[int(g)].cell == groups[0].cell for g in d.group])
    assert before[same_game & (d.group != 0)] == pytest.approx(after[same_game & (d.group != 0)])
    assert not np.allclose(before[~same_game], after[~same_game])
    left_in = measure.out_of_sample(groups, name, leave_out_game=False)
    assert not np.allclose(before, left_in), "leaving a game out is a different fit"


def test_a_group_is_compared_with_simulations_from_a_model_that_never_saw_it() -> None:
    """In the closed loop, changing one group changes nothing for the other groups of its
    fold, and changes the scores of groups in the other folds, whose models were fitted on
    it."""
    groups = _groups(others=1.0, count=6, seed=25)
    before = np.hstack(measure.closed_loop(groups))
    after = np.hstack(measure.closed_loop(_zeroed(groups, 0)))
    fold = np.array([measure.fold_of(group) for group in groups])
    same = fold == fold[0]
    same[0] = False
    assert before[same] == pytest.approx(after[same])
    assert not np.allclose(before[fold != fold[0]], after[fold != fold[0]])
