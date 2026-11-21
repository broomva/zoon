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


def test_measure_paste_has_heading_and_power_label() -> None:
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
    """Check the required labels in the paste."""
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


def test_notes_exclude_phrases_asserting_influence() -> None:
    """Reject listed phrases in the three notes."""
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


def candidate_note() -> tuple[str, str]:
    text = (NOTE.parent / "candidates.md").read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    return text[:start], text[start : text.rindex("\n```")]


def test_candidate_figures_are_printed_and_every_form_is_reported() -> None:
    from zoon.interaction import compare, person

    prose, paste = candidate_note()
    decimals = NUMBER
    printed = {n.lstrip("+-") for n in decimals.findall(paste)}
    assert {n.lstrip("+-") for n in decimals.findall(prose)} <= printed
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


def test_candidate_outcomes_and_tables_match_the_paste() -> None:
    from zoon.interaction import compare, person

    prose, paste = candidate_note()
    lines = dict(line.split(" = ", 1) for line in paste.splitlines()[1:])
    winner = lines["[selection, candidates] winner"]
    twin = lines["[selection, twins] winner"]
    normal = " ".join(prose.split())
    training = re.search(r"on (\d+) training groups", normal)
    held = re.search(r"The (\d+) held-out groups", normal)
    first_cut = re.search(r"The first cut is now fixed at (\d+)\.", normal)
    assert training and held and first_cut
    assert training[1] == lines["[data] training groups"]
    assert held[1] == lines["[data] held-out groups"]
    assert first_cut[1] == lines["[fit] first cut"]
    selection = re.search(
        r"Among candidates, (.*?) tie at step 2; (.*?) has the highest score; "
        r"the rule chooses (.*?) because it has the fewest fitted numbers among those tied\.",
        normal,
    )
    assert selection
    assert selection.groups() == (
        lines["[selection, candidates] tied at step 2"],
        lines["[selection, candidates] highest"],
        winner,
    )
    tied = lines["[selection, candidates] tied at step 2"].split(", ")
    assert int(lines[f"[count] {winner}"]) < min(
        int(lines[f"[count] {name}"]) for name in tied if name != winner
    )
    twins = re.search(
        r"Among twins, (.*?) has the highest score, is alone at step 2, "
        r"and is chosen by the rule\.",
        normal,
    )
    assert twins
    assert (
        twins[1]
        == lines["[selection, twins] highest"]
        == lines["[selection, twins] tied at step 2"]
        == twin
    )

    tables = [
        [[value.strip() for value in row.strip("|").split("|")] for row in block.splitlines()]
        for block in re.findall(r"(?:^\|[^\n]*\n)+", prose, re.M)
    ]
    expected: list[list[list[str]]] = []
    expected.append(
        [
            [
                c.name,
                lines[f"[count] {c.name}"],
                lines[f"[open loop] {c.name}"],
                lines[f"[closed loop, composite] {c.name}"],
            ]
            for c in person.CANDIDATES
        ]
    )
    gains: list[list[str]] = []
    powers: list[list[str]] = []
    for metric in ("H2", *compare.METRICS):
        values: list[str] = []
        for opponent in ("selected twin", "own twin"):
            label = (
                f"[gain, {opponent}, open loop] candidate minus twin"
                if metric == "H2"
                else f"[gain, {opponent}, closed loop, {metric}] twin minus candidate"
            )
            values.append(lines[label])
        gains.append([metric, *values])
        powers.append(
            [
                metric,
                *[
                    lines[f"[power, {opponent}, {metric}] {effect}"]
                    for opponent in ("selected twin", "own twin")
                    for effect in ("training effect", "half effect")
                ],
            ]
        )
    expected.append(gains)
    expected.append(
        [
            [
                summary,
                *[
                    lines[f"[coverage, {summary}] {name}"]
                    for name in (winner, twin, winner + " twin")
                ],
            ]
            for summary in compare.SUMMARIES
        ]
    )
    expected.append(powers)
    expected.append(
        [
            [arm, lines[f"[game left out, {arm}] score"], lines[f"[game left out, {arm}] loss"]]
            for arm in ("candidate", "twin")
        ]
    )
    expected.append(
        [
            [
                cell,
                lines[f"[game left out, candidate, {cell}] score"],
                lines[f"[game left out, twin, {cell}] score"],
            ]
            for cell in sorted(compare.GAMES)
        ]
    )
    assert [table[2:] for table in tables] == expected
    assert [table[0] for table in tables] == [
        ["Form", "Fitted numbers", "Open-loop nats per decision", "Closed-loop composite"],
        ["Measure", "Against selected twin", "Against own twin"],
        ["Summary", "Selected candidate", "Selected twin", "Own twin"],
        [
            "Measure",
            "Selected twin, training effect",
            "Selected twin, half effect",
            "Own twin, training effect",
            "Own twin, half effect",
        ],
        ["Arm", "Nats per decision", "Loss from leaving the game out"],
        ["Game", "Candidate", "Twin"],
    ]
    reproduced = re.search(
        r"The largest absolute fitted-number changes over reversed groups and a nudged start "
        r"are ([\d.]+) for (.*?) and ([\d.]+) for (.*?)\.",
        normal,
    )
    assert reproduced
    assert reproduced.groups() == (
        lines[f"[reproducibility, {winner}] largest absolute change"],
        winner,
        lines[f"[reproducibility, {twin}] largest absolute change"],
        twin,
    )


def test_candidate_pasted_settings_match_the_current_code() -> None:
    from zoon.interaction import compare

    _, paste = candidate_note()
    settings = [
        line
        for line in paste.splitlines()
        if line.startswith(("[fit]", "[simulation]", "[interval]"))
    ]
    assert settings == [entry.render() for entry in compare.settings()]


def test_candidate_pasted_amounts_match_the_training_file() -> None:
    from zoon.interaction import compare, data

    if not data.TRAINING.is_file():
        pytest.skip("training file is absent; the pasted amount table needs training groups")
    _, paste = candidate_note()
    amounts = [line for line in paste.splitlines() if line.startswith("[amounts,")]
    assert amounts == [entry.render() for entry in compare.amount_entries(data.load())]


def prompted_note() -> tuple[str, str]:
    text = (NOTE.parent / "prompted.md").read_text()
    prefix, paste = text.rsplit("\n```\n", 2)[:2]
    return prefix, paste.rstrip("`\n")


def test_prompted_table_and_quoted_figures_match_the_paste() -> None:
    prose, paste = prompted_note()
    rows: list[str] = []
    for line in paste.splitlines():
        match = re.fullmatch(
            r"\[(open loop|recalibration|disguise|closed loop|recall|total)\] calls = (\d+) "
            r"\| input tokens <= (\d+) \| output tokens <= (\d+) \| USD <= (\S+)",
            line,
        )
        if match:
            rows.append("| " + " | ".join(match.groups()) + " |")
    quoted = [line for line in prose.splitlines() if re.match(r"\| (?!Call|---)", line)]
    assert len(rows) == 6
    assert quoted == rows
    # Attribution identifies the source and its licence; these are not measured figures.
    prose = re.sub(r"The instructions are used under CC BY.*?\n\n", "", prose, flags=re.S)
    pattern = re.compile(r"\d+(?:\.\d+)?")
    assert set(pattern.findall(prose)) <= set(pattern.findall(paste))
    assert paste.startswith("zoon interaction: prompted spending bound\n")
    assert "[budget] total at most proposed ceiling = " in paste


def test_prompted_pasted_settings_match_current_objects() -> None:
    from zoon.interaction import budget

    _, paste = prompted_note()
    assert [
        line for line in paste.splitlines() if line.startswith("[settings]")
    ] == budget.settings()


def test_prompted_full_paste_matches_training_only_bound() -> None:
    from zoon.interaction import budget, data

    if not data.TRAINING.is_file():
        pytest.skip("training file is absent; the pasted bound needs training groups and counts")
    _, paste = prompted_note()
    assert paste == "\n".join(budget.report(data.load(), data.held_out_counts()))
