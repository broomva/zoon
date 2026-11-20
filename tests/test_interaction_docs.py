"""The interaction notes quote only numbers that their pasted output contains."""

import re
from pathlib import Path

import pytest

from tests.test_interaction_measure import played_groups
from zoon.interaction import measure

NOTE = Path(__file__).parent.parent / "docs" / "interaction" / "measure.md"
# Lines the note's argument rests on. A run must print each one.
REPORTED = (
    "[open loop] own history adds, entered first",
    "[open loop] own history adds, entered second",
    "[open loop] others' history adds, entered first",
    "[open loop] others' history adds, entered second",
    "[open loop, round 1 only] per game, no history",
    "[open loop, round 1 only] per game, own and others",
    "[members apart, round 1] closer than chance by",
    "[members apart, rounds 16 to 20] closer than chance by",
    "[power] closed loop, all three",
    "[power] closed loop, members apart",
)
NUMBER = re.compile(r"[+-]?\d+\.\d{3,}")


def _split() -> tuple[str, str]:
    """Return the prose and the pasted output, which is the last code block."""
    text = NOTE.read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    return text[:start], text[start : text.rindex("\n```")]


def test_the_output_of_a_complete_run_is_pasted() -> None:
    _, paste = _split()
    assert paste.startswith("zoon interaction: what the others' history adds")
    assert "[power] closed loop" in paste


def test_every_quoted_number_is_in_the_pasted_output() -> None:
    """A figure with three or more decimals in the prose must have been printed by the run."""
    prose, paste = _split()
    printed = {number.lstrip("+-") for number in NUMBER.findall(paste)}
    quoted = {number.lstrip("+-") for number in NUMBER.findall(prose)}
    assert len(quoted) > 10, "the note quotes its figures"
    assert sorted(quoted - printed) == []


def test_the_pasted_output_reports_what_the_note_relies_on() -> None:
    """Both orders of entering the two histories, the check against chance for every block,
    and power for each closed-loop measure are printed, so the note cannot quote one without
    the other."""
    _, paste = _split()
    for label in REPORTED:
        assert label in paste, label


def test_a_live_run_prints_what_the_note_relies_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same lines are printed by the program itself, run here on synthetic groups, and
    the pasted output has a line for every label a run prints."""
    groups = played_groups(others=1.0, count=6, seed=51)
    monkeypatch.setattr(measure.data, "load", lambda: groups)
    monkeypatch.setattr(measure.data, "held_out_counts", lambda: {"2-FE": 4, "4-FE": 3})
    measure.main()
    printed = capsys.readouterr().out
    for label in REPORTED:
        assert label in printed, label
    _, paste = _split()

    def labels(text: str) -> list[str]:
        return [line.split(" = ")[0] for line in text.splitlines()]

    assert labels(printed) == labels(paste)


def test_the_notes_claim_prediction_and_not_influence() -> None:
    """The data cannot separate influence from a shared tendency, so neither note may say it
    did. Wording that asserts influence or an exclusion is refused."""
    notes = NOTE.parent
    text = " ".join(
        (notes / name).read_text() for name in ("measure.md", "task-selection.md", "candidates.md")
    )
    text = " ".join(text.split()).lower()
    for phrase in (
        "influenced one another",
        "it rules out",
        "is ruled out",
        "separates influence",
        "this is influence",
        "as strangers",
        "moved toward the people",
    ):
        assert phrase not in text, phrase
    assert "neither shows that people influence each other" in text


def candidate_note() -> tuple[str, str]:
    text = (NOTE.parent / "candidates.md").read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    return text[:start], text[start : text.rindex("\n```")]


def test_candidate_figures_are_printed_and_every_form_is_reported() -> None:
    from zoon.interaction import compare, person

    prose, paste = candidate_note()
    printed = {n.lstrip("+-") for n in NUMBER.findall(paste)}
    assert {n.lstrip("+-") for n in NUMBER.findall(prose)} <= printed
    assert paste.startswith("zoon interaction: candidates on training groups\n")
    lines = dict(line.split(" = ", 1) for line in paste.splitlines()[1:])
    for c in person.CANDIDATES:
        assert int(lines[f"[count] {c.name}"]) == c.count
        numbers = {
            label: value
            for label, value in lines.items()
            if label.startswith(f"[numbers, {c.name}]")
        }
        assert len(numbers) == c.count
        for trace in c.traces:
            assert 0 <= float(lines[f"[numbers, {c.name}] trace fraction {trace}"]) <= 1
        assert f"[open loop] {c.name}" in lines
        for summary in compare.METRICS:
            assert f"[closed loop, {summary}] {c.name}" in lines
        for summary in compare.SUMMARIES:
            assert 0 <= float(lines[f"[coverage, {summary}] {c.name}"]) <= 1
    for label in ("candidates", "twins"):
        tied = lines[f"[selection, {label}] tied at step 2"].split(", ")
        winner = lines[f"[selection, {label}] winner"]
        reference = lines[f"[selection, {label}] highest"]
        assert reference in tied and winner in tied
        assert int(lines[f"[count] {winner}"]) == min(
            int(lines[f"[count] {name}"]) for name in tied
        )
    for comparison in ("selected twin", "own twin"):
        assert f"[gain, {comparison}, open loop] candidate minus twin" in lines
        for summary in compare.METRICS:
            assert f"[gain, {comparison}, closed loop, {summary}] twin minus candidate" in lines
        for summary in ("H2", *compare.METRICS):
            assert f"[power, {comparison}, {summary}] training effect" in lines
            assert f"[power, {comparison}, {summary}] half effect" in lines
    for arm in ("candidate", "twin"):
        assert f"[game left out, {arm}] loss" in lines


def test_candidate_choices_in_the_note_match_the_fitting_code() -> None:
    from zoon.interaction import compare, person

    prose, _ = candidate_note()
    assert f"{compare.SIMULATED_GROUPS} simulated groups" in prose
    assert f"{person.RIDGE} times the sum" in prose
    assert f"{person.MAX_ITERATIONS} iterations" in prose
    assert f"from -{person.LOG_GAP_LIMIT:g} to {person.LOG_GAP_LIMIT:g}" in prose
    assert f"{person.TOLERANCE:.8f}" in prose
    assert f"{person.FLOOR:.10f}" in prose
    assert "one start for every fit" in prose
    assert "no payoff" in prose and "roughly the same game mixture" in prose
    assert "conditional on the realised simulated groups" in prose


def test_candidate_findings_and_tables_match_the_report() -> None:
    from zoon.interaction import compare, person

    prose, paste = candidate_note()
    lines = dict(line.split(" = ", 1) for line in paste.splitlines()[1:])
    winner = lines["[selection, candidates] winner"]
    twin = lines["[selection, twins] winner"]
    assert f"The candidate winner is {winner}. The twin winner is {twin}." in prose
    for kind in ("candidates", "twins"):
        tied = lines[f"[selection, {kind}] tied at step 2"]
        assert f"Step 2 ties among {kind}: {tied}." in prose
    assert f"The highest candidate score is {lines['[selection, candidates] highest']}'s." in prose
    assert twin == winner + " twin", "the note says that the coded-arm comparisons coincide"
    assert f"Use {winner} as the coded person and {twin} as the coded comparator." in prose
    for c in person.CANDIDATES:
        row = (
            f"| {c.name} | {lines[f'[count] {c.name}']} | "
            f"{lines[f'[open loop] {c.name}']} | {lines[f'[closed loop, composite] {c.name}']} |"
        )
        assert row in prose
    for metric in ("H2", *compare.METRICS):

        def label(opponent: str, metric: str = metric) -> str:
            return (
                f"[gain, {opponent}, open loop] candidate minus twin"
                if metric == "H2"
                else f"[gain, {opponent}, closed loop, {metric}] twin minus candidate"
            )

        assert (
            f"| {metric} | {lines[label('selected twin')]} | {lines[label('own twin')]} |" in prose
        )
        values = [
            lines[f"[power, {opponent}, {metric}] {effect}"]
            for opponent in ("selected twin", "own twin")
            for effect in ("training effect", "half effect")
        ]
        assert "| " + " | ".join((metric, *values)) + " |" in prose
    for summary in compare.SUMMARIES:
        a, b = lines[f"[coverage, {summary}] {winner}"], lines[f"[coverage, {summary}] {twin}"]
        assert f"| {summary} | {a} | {b} | {b} |" in prose
        assert float(a) < 0.9, "the note says coverage is below 90% on all five"
    for arm in ("candidate", "twin"):
        assert (
            f"| {arm} | {lines[f'[game left out, {arm}] score']} | "
            f"{lines[f'[game left out, {arm}] loss']} |" in prose
        )
    for cell in sorted(compare.GAMES):
        a, b = (
            lines[f"[game left out, candidate, {cell}] score"],
            lines[f"[game left out, twin, {cell}] score"],
        )
        assert f"| {cell} | {a} | {b} |" in prose

    def numbers(label: str) -> list[float]:
        return [float(n) for n in NUMBER.findall(lines[label])]

    for gain_label in (
        "[gain, selected twin, open loop] candidate minus twin",
        "[gain, selected twin, closed loop, composite] twin minus candidate",
    ):
        assert numbers(gain_label)[1] > 0, "the note says both intervals are above zero"
    gains = {
        summary: numbers(f"[gain, selected twin, closed loop, {summary}] twin minus candidate")
        for summary in compare.SUMMARIES
    }
    assert max(gains, key=lambda s: gains[s][0]) == "members apart"
    assert gains["trend"][2] < 0, "the note says trend scores worse"
    assert "own and others' trace fractions" in prose
    assert "expect about two\ndecimal places in fitted numbers" in prose
