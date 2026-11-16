"""Deterministic synthetic records and recorded-shape judge replies for dry runs."""

import math

import numpy as np

from zoon.pilot import clients, cohorts, fetch, judge, prices, prompts, rewrites


def people(size: int, seed: int) -> list[cohorts.Participant]:
    rng = np.random.default_rng(seed)
    result: list[cohorts.Participant] = []
    for i in range(size):
        trials = tuple(
            cohorts.Trial(
                fetch.DILEMMAS[int(d)],
                "Yes" if rng.random() < 0.5 else "No",
                bool(rng.integers(2)),
                int(rng.integers(-100, 101)),
            )
            for d in rng.permutation(13)
        )
        result.append(cohorts.Participant(i % 4 + 1, f"synthetic-{seed}-{i:04}", trials))
    return sorted(result, key=lambda p: (p.experiment, p.participant))


def client(family: clients.Family, records: list[rewrites.Record]) -> clients.StubClient:
    human = judge.human_ratings()
    recorded: dict[str, clients.Response] = {}
    for record in records:
        if record["set"] != "outside":
            continue
        rating = human[str(record["id"])]
        low, high = math.floor(rating), math.ceil(rating)
        probabilities = {
            str(i): float(i == low)
            if low == high
            else (high - rating if i == low else rating - low if i == high else 0.0)
            for i in range(1, 8)
        }
        version = prices.DECISION.model if family == "decision" else prices.SECOND.model
        recorded[clients.key(prompts.evaluative(record))] = clients.Response(
            probabilities, 0.9 if family == "second" else None, "", version, 12, 1
        )
    return clients.StubClient(family, recorded)
