"""The game rules, checked against the worked examples in the participants' instructions."""

from fractions import Fraction

import pytest

from zoon.interaction import split
from zoon.interaction.games import GAMES


def test_full_contribution_matches_the_instructions_exercise() -> None:
    """Exercise 2 of the instructions has everyone contribute everything under aligned
    inequality. The pool is 84 points for two players and 336 for four, and every player
    ends the round with the same payoff."""
    assert GAMES["2-AI"].payoffs((36, 12)) == (Fraction(42), Fraction(42))
    assert GAMES["4-AI"].payoffs((36, 36, 12, 12)) == (Fraction(84),) * 4
    assert GAMES["2-AI"].payoffs((0, 0)) == (Fraction(36), Fraction(12))


def test_points_are_conserved_exactly() -> None:
    """What the players hold after a round is what they held before, minus what they put in,
    plus the pool. No float rounding is involved."""
    for game in GAMES.values():
        given = tuple(e // 3 for e in game.endowments)
        pool = sum((p * c for p, c in zip(game.productivities, given, strict=True)), Fraction(0))
        assert sum(game.payoffs(given)) == sum(game.endowments) - sum(given) + pool


def test_four_players_keep_the_return_per_point_of_two() -> None:
    """The authors doubled every productivity for four players, so one contributed point
    returns the same amount to its contributor at both sizes."""
    for treatment in ("FE", "AI", "MI"):
        pair, four = GAMES[f"2-{treatment}"], GAMES[f"4-{treatment}"]
        assert four.endowments == (pair.endowments[0],) * 2 + (pair.endowments[1],) * 2
        assert [p / 4 for p in four.productivities[::2]] == [p / 2 for p in pair.productivities]
    for game in GAMES.values():
        assert all(1 < p < game.size for p in game.productivities), "a social dilemma"


def test_a_contribution_outside_the_endowment_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        GAMES["2-EI"].payoffs((36, 13))
    with pytest.raises(ValueError, match="seats"):
        GAMES["4-FE"].payoffs((1, 1))


def test_two_fifths_of_each_game_are_held_out_whatever_the_order() -> None:
    groups = [str(n) for n in range(1, 28)]
    sealed = split.held_out("4-MI", groups)
    assert len(sealed) == 11
    assert split.held_out("4-MI", reversed(groups)) == sealed
    assert split.held_out("4-AI", groups) != sealed
    assert len(split.held_out("2-FE", [str(n) for n in range(1, 58)])) == 23
    assert len(split.held_out("2-FE", ["1"])) == 1
    assert split.group_index("2-FE", "1") != split.group_index("2-EI", "1")
