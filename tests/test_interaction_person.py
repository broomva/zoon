"""Behavioural controls for each person, using exact made-up contributions."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative  # type: ignore[reportMissingTypeStubs]

from zoon.interaction import measure, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS


def law(letter: str, twin: bool = False) -> person.Person:
    c = person.Candidate(letter, twin)
    numbers = c.start()
    # Different constants and cuts in both phases exercise the live phase switch.
    numbers[: 2 * c.phase_size] += np.linspace(-0.4, 0.6, 2 * c.phase_size)
    numbers[2 * c.phase_size : 2 * c.phase_size + len(c.features)] = np.arange(
        1, len(c.features) + 1
    )
    return person.Person(c, numbers)


@pytest.mark.parametrize("letter", person.LETTERS)
def test_exact_shares_and_first_round_have_no_traces(letter: str) -> None:
    p, game = law(letter), GAMES["2-FE"]
    low = p.update(person.State(), game, 0, (2, 2))
    high = p.update(person.State(), game, 0, (7, 7))
    assert measure.band(2, 24) == measure.band(7, 24)
    assert not np.array_equal(p.probabilities(game, 0, low), p.probabilities(game, 0, high))
    assert p.probabilities(game, 0, person.State((99.0,) * len(p.fractions))) == pytest.approx(
        p.probabilities(game, 0)
    )
    assert p.probabilities(game, 0).sum() == pytest.approx(1.0)


def test_c_two_traces_and_missing_kind() -> None:
    p = law("C")
    for cell, row in (("2-AI", (18, 3)), ("2-FE", (12, 6)), ("4-FE", (0, 6, 12, 24))):
        game = GAMES[cell]
        state = p.update(person.State(), game, 0, row)
        expected = np.mean([row[s] / game.endowments[s] for s in range(1, game.size)])
        assert state.traces[1:] == pytest.approx((expected, expected))
    game = GAMES["4-AI"]
    state = p.update(person.State(), game, 0, (0, 36, 0, 0))
    assert state.traces == (0, 1, 0)
    switched = p.update(person.State(), game, 0, (0, 0, 12, 12))
    assert switched.traces == (0, 0, 1)
    assert not np.allclose(p.probabilities(game, 0, state), p.probabilities(game, 0, switched))


def test_d_cap_and_reference_endowment_for_both_seats() -> None:
    p, game = law("D"), GAMES["2-AI"]
    for seat, row, expected in ((0, (0, 12), 1 / 3), (1, (24, 0), 1), (1, (6, 0), 0.5)):
        state = p.update(person.State(), game, seat, row)
        assert state.traces[1] == pytest.approx(expected)
    a = p.update(person.State(), game, 1, (24, 0))
    b = p.update(person.State(), game, 1, (36, 0))
    assert p.probabilities(game, 1, a) == pytest.approx(p.probabilities(game, 1, b))


def test_e_keeps_the_exact_first_share() -> None:
    p, game = law("E"), GAMES["2-FE"]
    numbers = p.numbers.copy()
    numbers[-len(p.fractions) :] = 1
    p = replace(p, numbers=numbers)
    low, high = person.State(), person.State()
    for r in range(ROUNDS - 1):
        low = p.update(low, game, 0, (2, 12) if r == 0 else (12, 12))
        high = p.update(high, game, 0, (7, 12) if r == 0 else (12, 12))
    assert low.traces == high.traces
    assert (low.anchor, high.anchor) == (2 / 24, 7 / 24)
    assert not np.allclose(p.probabilities(game, 0, low), p.probabilities(game, 0, high))


def test_f_twin_has_no_state_after_first_round() -> None:
    p, game = law("F", True), GAMES["2-AI"]
    low = p.update(person.State(), game, 0, (0, 0))
    high = p.update(person.State(), game, 0, (36, 12))
    assert p.candidate.features == () and p.fractions.size == 0
    assert p.probabilities(game, 0, low) == pytest.approx(p.probabilities(game, 0, high))
    f = law("F")
    a, b = f.update(person.State(), game, 0, (0, 0)), f.update(person.State(), game, 0, (36, 0))
    assert f.probabilities(game, 0, a) == pytest.approx(f.probabilities(game, 0, b))


@pytest.mark.parametrize("c", person.CANDIDATES, ids=lambda c: c.name)
def test_disguise_scales_every_point_without_changing_predictions(c: person.Candidate) -> None:
    p = law(c.letter, c.twin)
    for game in GAMES.values():
        # B uses the rules in their declared units, so canonicalise their common point unit.
        scaled = replace(game, endowments=tuple(10 * e for e in game.endowments))
        for seat in range(game.size):
            a, b = person.State(), person.State()
            for r in range(ROUNDS):
                row = tuple((r * 7 + s * 3) % (e + 1) for s, e in enumerate(game.endowments))
                assert p.probabilities(game, seat, a) == pytest.approx(
                    p.probabilities(scaled, seat, b)
                )
                a = p.update(a, game, seat, row)
                b = p.update(b, scaled, seat, tuple(10 * v for v in row))


def tiny() -> list[Group]:
    rng = np.random.default_rng(71)
    return [
        Group(
            cell,
            f"{cell}-{g}",
            tuple(
                tuple(
                    (0, e // 6, e // 2, 5 * e // 6, e)[int(rng.integers(5))]
                    for e in game.endowments
                )
                for _ in range(ROUNDS)
            ),
        )
        for cell, game in GAMES.items()
        for g in range(2)
    ]


@pytest.mark.parametrize("c", person.CANDIDATES, ids=lambda c: c.name)
def test_likelihood_gradient_and_live_state_agree(c: person.Candidate) -> None:
    groups = tiny()[::3]
    record = person.records(groups, c)
    numbers = c.start()
    # Different constants and cuts in both phases exercise the live phase switch.
    numbers[: 2 * c.phase_size] += np.linspace(-0.4, 0.6, 2 * c.phase_size)
    numbers[2 * c.phase_size : 2 * c.phase_size + len(c.features)] = 0.7
    loss, gradient, _ = person.likelihood(numbers, record, c)
    numeric = np.asarray(approx_derivative(lambda x: person.likelihood(x, record, c)[0], numbers))
    assert np.isfinite(loss)
    assert gradient == pytest.approx(numeric.ravel(), abs=2e-5)
    fitted = person.Person(c, numbers)
    live: list[float] = []
    for group in groups:
        game = GAMES[group.cell]
        states = [person.State() for _ in range(game.size)]
        for row in group.rounds:
            for seat, state in enumerate(states):
                live.append(
                    float(
                        np.log(
                            fitted.probabilities(game, seat, state)[
                                measure.band(row[seat], game.endowments[seat])
                            ]
                        )
                    )
                )
            states = [fitted.update(state, game, seat, row) for seat, state in enumerate(states)]
    assert person.log_likelihoods(fitted, groups) == pytest.approx(live)
    assert len(numbers) == c.count


def test_b_predicts_an_absent_game_and_a_missing_constant_stays_zero() -> None:
    groups = [g for g in tiny() if g.cell != "2-MI"]
    for letter in ("A", "B"):
        p = person.fit(groups, person.Candidate(letter))
        assert np.isfinite(p.probabilities(GAMES["2-MI"], 0)).all()
        if letter == "A":
            idx = measure.STRATA["2-MI", 0]
            assert p.numbers[idx] == 0 and p.numbers[p.candidate.phase_size + idx] == 0


def test_nonconvergence_includes_the_optimiser_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    def failed(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(success=False, message="iteration limit", x=np.zeros(2))

    monkeypatch.setattr(person, "minimize", failed)
    with pytest.raises(RuntimeError, match="did not converge: iteration limit"):
        person.fit(tiny(), person.Candidate("A"))


def test_counts_starts_and_ridge_cover_every_declared_parameter() -> None:
    counts = {"A": 40, "B": 22, "C": 42, "D": 40, "E": 41, "F": 38}
    twins = {"A": 38, "B": 20, "C": 38, "D": 38, "E": 39, "F": 36}
    for c in person.CANDIDATES:
        assert c.count == (twins if c.twin else counts)[c.letter]
        assert len(c.start()) == c.count
        assert (c.start()[-len(c.traces) :] == 0.5).all() if c.traces else True
        assert np.array_equal(c.start()[: c.phase_size], c.start()[c.phase_size : 2 * c.phase_size])
        record = person.records(tiny(), c)
        numbers = c.start() + 0.1
        loss, _, logs = person.likelihood(numbers, record, c)
        assert loss + logs.sum() == pytest.approx(0.01 * float((numbers**2).sum()))


def test_fit_uses_one_start_and_the_declared_bounds_and_stopping_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls = 0

    def inspected(fun: Any, start: person.Floats, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        c = person.Candidate("C")
        assert np.array_equal(start, c.start())
        assert kwargs["method"] == "L-BFGS-B" and kwargs["jac"] is True
        assert kwargs["options"] == {"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50}
        bounds = kwargs["bounds"]
        assert bounds[-3:] == [(0.0, 1.0)] * 3
        for offset in (0, c.phase_size):
            assert bounds[offset + c.constants + 1 : offset + c.phase_size] == [(-8.0, 8.0)] * 3
        assert np.isfinite(fun(start)[0])
        return SimpleNamespace(success=True, x=start)

    monkeypatch.setattr(person, "minimize", inspected)
    person.fit(tiny(), person.Candidate("C"))
    assert calls == 1


def test_c_copies_the_missing_trace_after_updates_with_different_fractions() -> None:
    p = law("C")
    numbers = p.numbers.copy()
    numbers[-3:] = (0.3, 0.1, 0.9)
    p = replace(p, numbers=numbers)
    for cell, seat, source in (("2-AI", 0, 2), ("2-AI", 1, 2), ("2-FE", 0, 1), ("4-FE", 0, 1)):
        game = GAMES[cell]
        state = person.State()
        expected = None
        for r in range(10):
            row = tuple(e if r % 2 else 0 for e in game.endowments)
            observed = float(r % 2)
            expected = (
                observed
                if expected is None
                else expected + p.fractions[source] * (observed - expected)
            )
            state = p.update(state, game, seat, row)
            assert state.traces[1:] == pytest.approx([expected, expected])
    # Both traces remain separate when both kinds of other seat are present.
    game = GAMES["4-AI"]
    state = p.update(person.State(), game, 0, (0, 0, 0, 0))
    state = p.update(state, game, 0, (0, 36, 12, 12))
    assert state.traces[1:] == pytest.approx((0.1, 0.9))


def test_b_uses_the_four_game_numbers_in_the_original_point_unit() -> None:
    c = person.Candidate("B")
    for game in GAMES.values():
        for seat in range(game.size):
            assert c.base(game, seat) == pytest.approx((1.0, *measure.traits(game, seat)))


def test_a_has_both_fading_traces() -> None:
    p, game = law("A"), GAMES["2-FE"]
    empty = p.update(person.State(), game, 0, (0, 0))
    own = p.update(person.State(), game, 0, (24, 0))
    others = p.update(person.State(), game, 0, (0, 24))
    fading = p.update(others, game, 0, (0, 0))
    assert fading.traces == (0.0, 0.5)
    expected = [
        float(p.probabilities(game, 0, state) @ np.arange(5))
        for state in (empty, own, fading, others)
    ]
    assert expected[0] < expected[1] < expected[3]
    assert expected[0] < expected[2] < expected[3]


def test_b_game_numbers_change_its_distribution_without_any_game_constant() -> None:
    p = law("B")
    numbers = p.numbers.copy()
    numbers[1] = numbers[p.candidate.phase_size + 1] = 2.0
    p = replace(p, numbers=numbers)
    equal, unequal = GAMES["2-FE"], GAMES["2-AI"]
    for state in (person.State(), person.State((0.5, 0.5), seen=True)):
        a, b = p.probabilities(equal, 0, state), p.probabilities(unequal, 0, state)
        assert b @ np.arange(5) > a @ np.arange(5)
        assert p.probabilities(unequal, 1, state) @ np.arange(5) < a @ np.arange(5)


@pytest.mark.parametrize("letter", person.LETTERS)
def test_twins_live_updates_and_fit_records_ignore_every_other_seat(letter: str) -> None:
    p = law(letter, True)
    for game in GAMES.values():
        for seat in range(game.size):
            a, b = person.State(), person.State()
            rows_a: list[tuple[int, ...]] = []
            rows_b: list[tuple[int, ...]] = []
            for r in range(ROUNDS):
                own = (r * 7) % (game.endowments[seat] + 1)
                low = tuple(own if s == seat else 0 for s in range(game.size))
                high = tuple(own if s == seat else e for s, e in enumerate(game.endowments))
                rows_a.append(low)
                rows_b.append(high)
                a = p.update(a, game, seat, low)
                b = p.update(b, game, seat, high)
                assert a == b
            ra = person.records([Group(game.cell, "a", tuple(rows_a))], p.candidate)
            rb = person.records([Group(game.cell, "b", tuple(rows_b))], p.candidate)
            for name in ("base", "observations", "anchor", "choice", "sources"):
                assert np.array_equal(getattr(ra, name)[seat], getattr(rb, name)[seat])


def test_a_c_d_twins_have_identical_likelihood_gradient_fit_and_live_law() -> None:
    groups = tiny()[::3]
    candidates = [person.Candidate(letter, True) for letter in "ACD"]
    fitted = [person.fit(groups, c) for c in candidates]
    reference = fitted[0]
    for c, p in zip(candidates, fitted, strict=True):
        assert p.numbers == pytest.approx(reference.numbers, abs=1e-5)
        record = person.records(groups, c)
        loss, gradient, logs = person.likelihood(reference.numbers, record, c)
        expected = person.likelihood(
            reference.numbers, person.records(groups, candidates[0]), candidates[0]
        )
        assert loss == expected[0]
        assert np.array_equal(gradient, expected[1])
        assert np.array_equal(logs, expected[2])
        game = GAMES["4-AI"]
        state = p.update(person.State(), game, 1, (1, 7, 4, 12))
        assert state == reference.update(person.State(), game, 1, (1, 7, 4, 12))
        assert p.probabilities(game, 1, state) == pytest.approx(
            reference.probabilities(game, 1, state)
        )
