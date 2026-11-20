"""Exact ledgers, restricted channels and deterministic streams."""

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.test_interaction_person import law
from zoon.interaction import kernel, measure, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats


def training() -> list[Group]:
    return [
        Group(
            cell,
            cell,
            tuple(tuple((r % 5) * e // 4 for e in game.endowments) for r in range(ROUNDS)),
        )
        for cell, game in GAMES.items()
    ]


def fixed(view: kernel.View) -> Floats:
    return np.array([0.1, 0.2, 0.4, 0.2, 0.1])


def nothing(view: kernel.View) -> Floats:
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0])


def everything(view: kernel.View) -> Floats:
    return np.array([0.0, 0.0, 0.0, 0.0, 1.0])


def test_amount_table_uses_only_supplied_training_amounts_and_keeps_their_frequency() -> None:
    groups = training()
    table = kernel.Amounts.build(groups)
    for (endowment, b), values in table.rows.items():
        expected = sorted(
            row[s]
            for group in groups
            for row in group.rounds
            for s, e in enumerate(GAMES[group.cell].endowments)
            if e == endowment and measure.band(row[s], e) == b
        )
        assert list(values) == expected
        assert len(values) > 1
    for e in (12, 24, 36):
        for b in range(5):
            for u in (0.0, 0.31, 0.99):
                v = table.draw(e, b, u)
                assert 0 <= v <= e and measure.band(v, e) == b
                if b == 0:
                    assert v == 0
                if b == 4:
                    assert v == e
    changed = [replace(g, rounds=tuple(tuple(0 for _ in row) for row in g.rounds)) for g in groups]
    assert kernel.Amounts.build(changed).rows == {}
    with pytest.raises(ValueError, match="no training amounts"):
        kernel.Amounts.build(changed).draw(24, 2, 0.4)


@pytest.mark.parametrize("cell", GAMES)
def test_every_round_balances_and_log_replays_byte_for_byte(cell: str) -> None:
    game, table = GAMES[cell], kernel.Amounts.build(training())
    run = kernel.play(game, [fixed] * game.size, table, 9, "one")
    for row in run.rounds:
        pool = sum((p * c for p, c in zip(game.productivities, row, strict=True)), Fraction())
        assert sum(game.payoffs(row), Fraction()) == sum(game.endowments) - sum(row) + pool
    records = [json.loads(line) for line in run.log]
    assert len(records) == 1 + ROUNDS * game.size * 3
    for r, row in enumerate(run.rounds):
        transfers = [v for v in records if v["kind"] == "transfer" and v["round"] == r]
        assert tuple(Fraction(v["payoff"]) for v in transfers) == game.payoffs(row)
    assert kernel.replay(run.bytes).bytes == run.bytes
    assert kernel.play(game, [fixed] * game.size, table, 10, "one").rounds != run.rounds
    # Another group cannot consume this group's draws.
    kernel.play(game, [fixed] * game.size, table, 9, "another")
    assert kernel.play(game, [fixed] * game.size, table, 9, "one").bytes == run.bytes
    damaged = run.bytes.replace(b'"payoff":"', b'"payoff":"1', 1)
    with pytest.raises(ValueError, match="does not replay"):
        kernel.replay(damaged)


def test_balance_check_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    game = GAMES["2-FE"]

    def broken(self: Game, row: tuple[int, ...]) -> tuple[Fraction, ...]:
        return (Fraction(),) * self.size

    monkeypatch.setattr(type(game), "payoffs", broken)
    with pytest.raises(ArithmeticError, match="does not balance"):
        kernel.play(game, [fixed] * 2, kernel.Amounts.build(training()), 1, "one")


@pytest.mark.parametrize("letter", person.LETTERS)
@pytest.mark.parametrize("cell", ("2-MI", "4-AI"))
def test_zero_other_weights_play_identically_with_the_channel_closed(
    letter: str, cell: str
) -> None:
    p, game = law(letter), GAMES[cell]
    weights = p.numbers.copy()
    for j, name in enumerate(p.candidate.features):
        if name in ("others", "same", "other"):
            weights[2 * p.candidate.phase_size + j] = 0
    p = replace(p, numbers=weights)
    table = kernel.Amounts.build(training())
    opened = kernel.play(game, [kernel.Coded(p) for _ in range(game.size)], table, 8, "one")
    closed = kernel.play(
        game, [kernel.Coded(p) for _ in range(game.size)], table, 8, "one", open_channel=False
    )
    assert opened.rounds == closed.rounds
    opened_intents = [v for v in opened.log if '"kind":"intent"' in v]
    closed_intents = [v for v in closed.log if '"kind":"intent"' in v]
    assert opened_intents == closed_intents


@pytest.mark.parametrize("cell", ("2-MI", "4-AI"))
def test_closed_channel_reveals_no_other_contribution_or_payoff(cell: str) -> None:
    game, table = GAMES[cell], kernel.Amounts.build(training())
    laws: list[list[Floats]] = [[], []]
    histories: list[list[kernel.View]] = [[], []]
    for which, other in enumerate((nothing, everything)):

        def spy(view: kernel.View, which: int = which) -> Floats:
            histories[which].append(view)
            p = kernel.Coded(law("A"))(view)
            laws[which].append(p)
            return p

        kernel.play(game, [spy, *([other] * (game.size - 1))], table, 7, "one", open_channel=False)
    assert np.array(laws[0]) == pytest.approx(np.array(laws[1]))
    assert histories[0] == histories[1]
    assert histories[0][-1].history
    for observation in histories[0][-1].history:
        assert len(observation.contributions) == 1
        assert observation.contributions[0][0] == 0
        assert observation.payoffs is None
    if game.size == 2:
        # Giving the focal seat its payoff would reveal the partner exactly.
        a, b = game.payoffs((0, 0))[0], game.payoffs((0, game.endowments[1]))[0]
        assert a != b


def test_rounds_are_batches_and_open_channel_shows_all_outcomes() -> None:
    table = kernel.Amounts.build(training())
    seen: list[kernel.View] = []

    def observer(view: kernel.View) -> Floats:
        seen.append(view)
        return fixed(view)

    game = GAMES["4-MI"]
    run = kernel.play(game, [observer] * game.size, table, 3, "one")
    assert [len(v.history) for v in seen] == [r for r in range(ROUNDS) for _ in range(game.size)]
    for r in range(1, ROUNDS):
        for seat in range(game.size):
            observation = seen[r * game.size + seat].history[-1]
            assert observation.contributions == tuple(enumerate(run.rounds[r - 1]))
            assert observation.payoffs == game.payoffs(run.rounds[r - 1])


def test_bad_distributions_are_refused() -> None:
    for values in ([1.0, 2.0], [0.0] * 5, [-1.0, 1.0, 1.0, 1.0, 1.0], [float("nan")] * 5):
        with pytest.raises(ValueError, match="five finite"):
            kernel.distribution(np.array(values))


def test_coded_prefix_cache_cannot_change_a_prediction() -> None:
    p = law("E")
    game = GAMES["2-AI"]
    observations = tuple(kernel.Observation(((0, r), (1, r // 3)), None) for r in range(12))
    coded = kernel.Coded(p)
    views = [kernel.View(game, 0, observations[:r]) for r in range(12)]
    views += [views[3], views[10], kernel.View(game, 1, observations), views[0]]
    for view in views:
        assert coded(view) == pytest.approx(kernel.Coded(p)(view))


def test_every_stream_key_part_changes_the_draw() -> None:
    keys = [
        (1, "2-AI", "one", 0, 0, "band"),
        (2, "2-AI", "one", 0, 0, "band"),
        (1, "2-MI", "one", 0, 0, "band"),
        (1, "2-AI", "two", 0, 0, "band"),
        (1, "2-AI", "one", 1, 0, "band"),
        (1, "2-AI", "one", 0, 1, "band"),
        (1, "2-AI", "one", 0, 0, "amount"),
    ]
    draws = [kernel.uniform(*key) for key in keys]
    assert all(0 <= u < 1 for u in draws) and len(set(draws)) == len(keys)
    assert [kernel.uniform(*key) for key in keys] == draws
