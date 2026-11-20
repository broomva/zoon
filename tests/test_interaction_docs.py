"""The interaction notes quote only numbers that their pasted output contains."""

import re
from pathlib import Path

NOTE = Path(__file__).parent.parent / "docs" / "interaction" / "measure.md"
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
    for label in (
        "[open loop] own history adds, entered first",
        "[open loop] own history adds, entered second",
        "[open loop] others' history adds, entered first",
        "[open loop] others' history adds, entered second",
        "[members apart, round 1] closer than chance by",
        "[members apart, rounds 16 to 20] closer than chance by",
        "[power] closed loop, all three",
        "[power] closed loop, members apart",
    ):
        assert label in paste, label


def test_the_notes_claim_prediction_and_not_influence() -> None:
    """The data cannot separate influence from a shared tendency, so neither note may say it
    did. Wording that asserts influence or an exclusion is refused."""
    notes = NOTE.parent
    text = " ".join((notes / name).read_text() for name in ("measure.md", "task-selection.md"))
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
