"""Call accounting, character bounds and offline execution use synthetic training data."""

import inspect
from decimal import Decimal
from pathlib import Path

import pytest

from tests.synthetic import write_raw
from zoon.pilot import budget, cohorts, fetch, materials, prices, prompts, rewrites, split


def test_prices_and_rounding() -> None:
    assert prices.DECISION.model == "jev-1.14.0"
    assert prices.SECOND.model == "gemini-2.5-flash-lite"
    assert prices.DECISION.read == prices.SECOND.read == "2026-11-15"
    assert prices.DECISION.page == "https://docs.typesafe.ai/models"
    assert prices.SECOND.page == "https://cloud.google.com/vertex-ai/generative-ai/pricing"
    assert prices.DECISION.dollars(1_000_000, 123) == Decimal("0.042")
    assert prices.SECOND.dollars(1_000_000, 1_000_000) == Decimal("0.50")
    assert prices.SECOND.input_per_million == Decimal("0.10")
    assert prices.SECOND.output_per_million == Decimal("0.40")
    for n in range(10):
        assert budget.input_tokens("x" * n, "second") == (n + 2) // 3
        assert budget.input_tokens("x" * n, "decision") == (n + 2) // 3 + 400
    request = prompts.Decision("state", "question", prompts.declared(("Yes", "No")))
    assert (
        budget.input_tokens(request, "decision")
        == (len(prompts.decision_bytes(request).decode()) + 2) // 3 + 400
    )


def test_bound_covers_every_valid_trial_rendering() -> None:
    tasks = materials.load()
    all_trials = [
        prompts.trial(cohorts.Trial(d, a, s, -100), tasks)
        for d in fetch.DILEMMAS
        for a in ("Yes", "No")
        for s in (False, True)
    ]
    assert budget.longest_trials(tasks) == sorted(all_trials, key=len, reverse=True)[:12]
    bounded = budget.held_out_request("Two examples.", tasks)
    assert bounded.state == prompts.prompted_state(
        "Two examples.",
        "\n\n".join(sorted(all_trials, key=len, reverse=True)[:12]),
        max((t["text"] for t in tasks.values()), key=len),
    )
    for rating in (-100, -99, 0, 99, 100):
        for d in fetch.DILEMMAS:
            for a in ("Yes", "No"):
                for s in (False, True):
                    assert len(prompts.trial(cohorts.Trial(d, a, s, rating), tasks)) <= len(
                        prompts.trial(cohorts.Trial(d, a, s, -100), tasks)
                    )


def test_call_accounting_and_sealed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sealed = frozenset(
        (e, str(i))
        for e in fetch.SOURCES
        for i in range(1, 221)
        if split.assign(e, str(i)) == "held_out"
    )
    write_raw(tmp_path, 220, unreadable=sealed)
    people = cohorts.participants(fetch.load(tmp_path))
    document = cohorts.generate(people)
    tasks, records = materials.load(), rewrites.load()
    rows = budget.estimate(people, document, tasks, records)
    by_key = {(r.family, r.step): r for r in rows}
    for family in budget.FAMILIES:
        assert by_key[family, "outside judge"].calls == 25
        assert by_key[family, "task and strict judges"].calls == 13 + 5 * 38
        assert by_key[family, "reversal"].calls == sum("reversed" in r for r in records) == 17
        assert by_key[family, "calibration"].calls == 300 * 11 * 13
        assert by_key[family, "held-out bound"].calls == 680 * 13 * 2
        assert by_key[family, "recognition"].calls == (13 if family == "second" else 0)
    assert sum(r.calls for r in rows if r.family == "decision") == 60825
    assert sum(r.calls for r in rows if r.family == "second") == 60838
    assert by_key["second", "recognition"].output_tokens == 13 * 40
    assert all(
        r.output_tokens == (r.calls if r.family == "second" else 0)
        for r in rows
        if r.step != "recognition"
    )
    assert budget.total(rows) == budget.total(rows, "decision") + budget.total(rows, "second")
    assert budget.total(rows) <= Decimal("120")

    # Independently add the actual calibration requests and each rotation's bound.
    for family in budget.FAMILIES:
        actual = 0
        bounded = 0
        lookup = {(p.experiment, p.participant): p for p in people}
        for rotation in document["rotations"]:
            examples = prompts.examples(rotation, people, tasks)
            bound = budget.held_out_request(examples, tasks)
            as_family = bound if family == "decision" else prompts.second(bound)
            bounded += 680 * 2 * budget.input_tokens(as_family, family)
            for spec in document["calibration"]:
                person = lookup[spec["experiment"], spec["participant"]]
                for i, trial in enumerate(person.trials, 1):
                    request = prompts.prompted(person, i, examples, tasks)
                    rendered = request if family == "decision" else prompts.second(request)
                    assert budget.input_tokens(rendered, family) <= budget.input_tokens(
                        as_family, family
                    )
                    if trial.dilemma not in rotation["held_out"]:
                        actual += budget.input_tokens(rendered, family)
        assert by_key[family, "calibration"].input_tokens == actual
        assert by_key[family, "held-out bound"].input_tokens == bounded

    # The report may use only the training loader, not an additional count or data fetch.
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected data access")

    monkeypatch.setattr(fetch, "held_out_counts", forbidden)
    monkeypatch.setattr(fetch, "fetch", forbidden)
    monkeypatch.setattr(cohorts, "load_training", lambda: people)
    monkeypatch.setattr(cohorts, "load", lambda: document)
    committed = tmp_path / "cohorts.json"
    committed.write_bytes(cohorts.serialise(document))
    monkeypatch.setattr(cohorts, "FILE", committed)
    budget.run()
    printed = capsys.readouterr().out
    assert f"[budget] total USD = {budget.total(rows):.6f}" in printed
    assert "[budget] total at most 120 USD = True" in printed
    assert "held_out_counts" not in inspect.getsource(budget)


def test_pasted_budget_is_the_real_output(capsys: pytest.CaptureFixture[str]) -> None:
    if not all((fetch.RAW_DIR / s.filename).exists() for s in fetch.SOURCES.values()):
        pytest.skip("training raw files absent; budget paste regeneration requires training data")
    from tests.test_docs import split_note

    budget.run()
    _, paste = split_note("budget.md")
    assert capsys.readouterr().out.rstrip("\n") == paste
