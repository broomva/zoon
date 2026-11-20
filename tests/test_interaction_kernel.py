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
            tuple(
                tuple(
                    0
                    if r % 5 == 0
                    else e
                    if r % 5 == 4
                    else (r % 5 - 1) * e // 3 + 1 + (0, 0, 1, 2)[r // 5]
                    for e in game.endowments
                )
                for r in range(ROUNDS)
            ),
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
        assert len(set(values)) >= 3
        for i in range(len(values)):
            assert table.draw(endowment, b, (i + 0.5) / len(values)) == values[i]
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
        pool = sum((p * c for p, c in zip(game.productivities, row, strict=True)), Fraction())
        transfers = [v for v in records if v["kind"] == "transfer" and v["round"] == r]
        assert tuple(Fraction(v["payoff"]) for v in transfers) == game.payoffs(row)
        for seat, transfer in enumerate(transfers):
            kept = Fraction(game.endowments[seat] - row[seat])
            assert Fraction(transfer["kept"]) == kept
            assert Fraction(transfer["pool"]) == pool
            assert Fraction(transfer["share"]) == pool / game.size
            assert kept + Fraction(transfer["share"]) == Fraction(transfer["payoff"])
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
@pytest.mark.parametrize("twin", (False, True))
@pytest.mark.parametrize("cell", ("2-MI", "4-AI"))
def test_zero_other_weights_play_identically_with_the_channel_closed(
    letter: str, twin: bool, cell: str
) -> None:
    p, game = law(letter, twin), GAMES[cell]
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


def test_coded_prefix_cache_matches_fresh_predictions() -> None:
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


@pytest.mark.parametrize("c", person.CANDIDATES, ids=lambda c: c.name)
def test_played_band_likelihood_matches_own_rounds_and_coded_run_replays(
    c: person.Candidate,
) -> None:
    p = law(c.letter, c.twin)
    table = kernel.Amounts.build(training())
    for game in GAMES.values():
        run = kernel.play(
            game,
            [kernel.Coded(p) for _ in range(game.size)],
            table,
            11,
            "coded",
            open_channel=not c.twin,
        )
        intents = [json.loads(v) for v in run.log if '"kind":"intent"' in v]
        logged = [np.log(v["probabilities"][v["band"]]) for v in intents]
        expected = person.log_likelihoods(p, [Group(game.cell, "coded", run.rounds)])
        assert logged == pytest.approx(expected)
        assert kernel.replay(run.bytes).bytes == run.bytes


def test_play_uses_separate_band_amount_seat_and_group_streams() -> None:
    game = GAMES["4-AI"]
    table = kernel.Amounts.build(training())
    for label in ("first", "second"):
        run = kernel.play(game, [fixed] * game.size, table, 19, label)
        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for line in run.log:
            v = json.loads(line)
            if v["kind"] != "intent":
                continue
            r, s = int(v["round"]), int(v["seat"])
            u = kernel.uniform(19, game.cell, label, r, s, "band")
            b = min(int(np.searchsorted(np.cumsum(p), u, side="right")), 4)
            assert v["band"] == b
            e = game.endowments[s]
            amount_u = kernel.uniform(19, game.cell, label, r, s, "amount")
            expected = (
                0
                if b == 0
                else e
                if b == 4
                else table.rows[e, b][int(amount_u * len(table.rows[e, b]))]
            )
            assert v["points"] == expected


def test_coded_open_channel_updates_from_all_actual_seats() -> None:
    game, p = GAMES["4-AI"], law("C")
    coded = kernel.Coded(p)
    rows = [(0, 36, 2, 7), (5, 0, 12, 3)]
    state = person.State()
    history: tuple[kernel.Observation, ...] = ()
    for row in rows:
        state = p.update(state, game, 0, row)
        history += (kernel.Observation(tuple(enumerate(row)), game.payoffs(row)),)
        assert coded(kernel.View(game, 0, history)) == pytest.approx(
            p.probabilities(game, 0, state)
        )


@pytest.mark.parametrize(
    "rows",
    (
        {(24, 1): (23,)},
        {(24, 2): (25,)},
        {(24, 3): (25,)},
        {(24, 1): (-1,)},
        {(24, 2): (8.5,)},
        {(24, 4): (24,)},
        {(0, 1): (0,)},
        {(0, 1): ()},
        {(24, 4): ()},
    ),
)
def test_amounts_reject_illegal_or_wrong_band_values(
    rows: dict[tuple[int, int], tuple[int, ...]],
) -> None:
    message = (
        "amount table needs a positive endowment and a middle band"
        if any(e <= 0 or b not in (1, 2, 3) for e, b in rows)
        else "amount is illegal or outside its band"
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        kernel.Amounts(rows)


@pytest.mark.parametrize(
    "damage", ("dropped", "empty", "blank", "treatment", "endowment", "productivity", "amount")
)
def test_replay_rejects_malformed_records_and_changed_game_rules(
    damage: str,
) -> None:
    game = GAMES["2-FE"]
    run = kernel.play(game, [fixed] * game.size, kernel.Amounts.build(training()), 1, "one")
    records = [json.loads(v) for v in run.log]
    if damage == "dropped":
        del records[1]
    elif damage == "empty":
        records = []
    elif damage == "treatment":
        records[0]["treatment"] = "AI"
    elif damage == "endowment":
        records[0]["endowments"][0] *= 2
    elif damage == "productivity":
        records[0]["productivities"][0] = "2"
    elif damage == "amount":
        records[0]["amounts"][0][2][0] = 999
    damaged = "".join(kernel.encoded(v) for v in records).encode()
    if damage == "blank":
        damaged += b"\n"
    with pytest.raises((ValueError, KeyError, IndexError)):
        kernel.replay(damaged)


def test_replay_accepts_consistently_rescaled_weights_but_checks_their_normalisation() -> None:
    game = GAMES["2-FE"]

    def weights(view: kernel.View) -> Floats:
        return np.ones(5)

    run = kernel.play(game, [weights] * game.size, kernel.Amounts.build(training()), 1, "one")
    records = [json.loads(v) for v in run.log]
    records[1]["returned"] = [2.0] * 5
    changed = "".join(kernel.encoded(v) for v in records).encode()
    assert kernel.replay(changed).bytes == changed
    records[1]["probabilities"] = [0.1] * 5
    with pytest.raises(ValueError, match="does not replay"):
        kernel.replay("".join(kernel.encoded(v) for v in records).encode())


@pytest.mark.parametrize("kind", ("list", "single precision"))
def test_play_records_double_precision_weights_and_replays(kind: str) -> None:
    values = [0.1, 0.2, 0.4, 0.2, 0.1]

    def raw(view: kernel.View) -> Floats:
        return values if kind == "list" else np.asarray(values, dtype=np.float32)  # type: ignore[return-value]

    game = GAMES["2-FE"]
    try:
        run = kernel.play(game, [raw] * game.size, kernel.Amounts.build(training()), 1, "one")
    except (AttributeError, TypeError, ValueError):
        pytest.fail("play must accept returned lists and single-precision arrays")
    assert kernel.replay(run.bytes).bytes == run.bytes
    expected = np.asarray(raw(kernel.View(game, 0, ())), dtype=np.float64)
    for line in run.log:
        entry = json.loads(line)
        if entry["kind"] == "intent":
            assert entry["returned"] == expected.tolist()
            assert entry["probabilities"] == kernel.distribution(expected).tolist()


def test_balance_rejects_unequal_pool_allocation_even_when_total_is_conserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = GAMES["2-FE"]

    def unequal(self: Game, points: tuple[int, ...]) -> tuple[Fraction, ...]:
        pool = sum((p * c for p, c in zip(self.productivities, points, strict=True)), Fraction())
        return tuple(
            Fraction(e - c) + (pool if s == 0 else 0)
            for s, (e, c) in enumerate(zip(self.endowments, points, strict=True))
        )

    monkeypatch.setattr(Game, "payoffs", unequal)
    with pytest.raises(ArithmeticError, match="does not balance"):
        kernel.play(game, [fixed] * game.size, kernel.Amounts.build(training()), 1, "one")


@pytest.mark.parametrize("letter", person.LETTERS)
def test_nonzero_other_weights_distinguish_channels_and_closed_state_is_centred(
    letter: str,
) -> None:
    game, p = GAMES["4-AI"], law(letter)
    table = kernel.Amounts.build(training())
    plays: list[list[tuple[int, ...]]] = []
    for opened in (False, True):
        runs: list[tuple[int, ...]] = []
        laws: list[Floats] = []
        for other in (nothing, everything):
            coded = kernel.Coded(p)
            probabilities: list[Floats] = []

            def focal(
                view: kernel.View,
                coded: kernel.Coded = coded,
                probabilities: list[Floats] = probabilities,
                opened: bool = opened,
            ) -> Floats:
                result = coded(view)
                probabilities.append(result)
                if view.history and not opened:
                    for name, value in zip(
                        p.candidate.traces,
                        coded._state.traces,  # type: ignore[reportPrivateUsage]
                        strict=True,
                    ):
                        if name != "own":
                            assert value == 0.5
                return result

            run = kernel.play(
                game, [focal, other, other, other], table, 7, "one", open_channel=opened
            )
            runs.append(tuple(row[0] for row in run.rounds))
            laws.append(np.array(probabilities))
        if not opened:
            assert runs[0] == runs[1]
            assert np.array_equal(laws[0], laws[1])
        else:
            assert not np.array_equal(laws[0], laws[1])
        plays.append(runs)
    assert plays[0] != plays[1]


def test_amounts_accept_numpy_integers_and_reject_bool_and_float_keys() -> None:
    groups = [
        replace(g, rounds=tuple(tuple(np.int64(x) for x in row) for row in g.rounds))
        for g in training()
    ]
    direct = kernel.Amounts({(np.int64(24), np.int64(1)): (np.int64(1),)})  # type: ignore[arg-type]
    assert direct.rows == {(24, 1): (1,)}
    assert all(type(x) is int for key in direct.rows for x in key)
    assert all(type(x) is int for values in direct.rows.values() for x in values)
    direct_run = kernel.play(GAMES["2-FE"], [nothing] * 2, direct, 1, "one")
    assert kernel.replay(direct_run.bytes).bytes == direct_run.bytes
    table = kernel.Amounts.build(groups)
    assert table.rows == kernel.Amounts.build(training()).rows
    game = GAMES["2-FE"]
    run = kernel.play(game, [fixed] * 2, table, 1, "one")
    assert kernel.replay(run.bytes).bytes == run.bytes
    with pytest.raises(ValueError, match=r"^amount is illegal or outside its band$"):
        kernel.Amounts({(24, 1): (True,)})
    for key in ((24.0, 1), (24, 1.0)):
        with pytest.raises(
            ValueError, match=r"^amount table needs a positive endowment and a middle band$"
        ):
            kernel.Amounts({key: (1,)})  # type: ignore[arg-type]


def test_amounts_build_rejects_bool_before_integer_conversion() -> None:
    group = Group("2-FE", "one", ((True, True),) * ROUNDS)
    with pytest.raises(ValueError, match=r"^amount is illegal or outside its band$"):
        kernel.Amounts.build([group])
