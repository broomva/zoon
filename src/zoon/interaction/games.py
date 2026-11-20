"""The rules of the eight linear public goods games, as the participants were told them.

Source: Wang, Hilbe and Zhang, "The dynamics of cooperation in asymmetric public goods
games", PNAS 123(5), 2026, https://doi.org/10.1073/pnas.2525760123. The two-player
parameters are Table S1 of its appendix and the four-player ones are Table S15.

Each round every player receives an endowment and chooses a whole number of points to
contribute. A contributed point adds the player's productivity to a common pool, and the pool
is split evenly. Points are kept exact as fractions, because the productivities have one
decimal place and a float would not conserve the pool.
"""

from dataclasses import dataclass
from fractions import Fraction

ROUNDS = 20


@dataclass(frozen=True)
class Game:
    """One treatment at one group size. Seats are in the order of the published data."""

    cell: str
    treatment: str
    endowments: tuple[int, ...]
    productivities: tuple[Fraction, ...]

    @property
    def size(self) -> int:
        return len(self.endowments)

    def payoffs(self, contributions: tuple[int, ...]) -> tuple[Fraction, ...]:
        """Return each seat's payoff for one round: what it kept plus its share of the pool."""
        if len(contributions) != self.size:
            raise ValueError(f"{self.cell} has {self.size} seats")
        for given, endowment in zip(contributions, self.endowments, strict=True):
            if not 0 <= given <= endowment:
                raise ValueError(f"a contribution of {given} is outside 0..{endowment}")
        pool = sum(
            (p * c for p, c in zip(self.productivities, contributions, strict=True)), Fraction(0)
        )
        share = pool / self.size
        return tuple(e - c + share for e, c in zip(self.endowments, contributions, strict=True))


def _tenths(*values: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(value, 10) for value in values)


def _pair(treatment: str, endowments: tuple[int, int], tenths: tuple[int, int]) -> Game:
    return Game(f"2-{treatment}", treatment, endowments, _tenths(*tenths))


def _four(treatment: str, endowments: tuple[int, int], tenths: tuple[int, int]) -> Game:
    """Four players mirror the pair: seats 1 and 2 are the first type, 3 and 4 the second."""
    first, second = endowments
    high, low = tenths
    return Game(
        f"4-{treatment}",
        treatment,
        (first, first, second, second),
        _tenths(high, high, low, low),
    )


# FE full equality, EI endowment inequality, PI productivity inequality, AI aligned
# inequality (the richer player is the more productive), MI misaligned inequality.
GAMES: dict[str, Game] = {
    game.cell: game
    for game in (
        _pair("FE", (24, 24), (16, 16)),
        _pair("EI", (36, 12), (16, 16)),
        _pair("PI", (24, 24), (19, 13)),
        _pair("AI", (36, 12), (19, 13)),
        _pair("MI", (36, 12), (13, 19)),
        _four("FE", (24, 24), (32, 32)),
        _four("AI", (36, 12), (38, 26)),
        _four("MI", (36, 12), (26, 38)),
    )
}
