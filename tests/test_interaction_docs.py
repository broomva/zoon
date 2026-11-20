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
