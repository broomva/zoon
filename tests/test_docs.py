"""The power note quotes only numbers that its pasted report output contains."""

import re
from pathlib import Path

NOTE = Path(__file__).parent.parent / "docs" / "pilot" / "power.md"
NUMBER = re.compile(r"[+-]?\d+\.\d{3,}")


def split_note() -> tuple[str, str]:
    """Return the prose and the pasted report output, which is the last code block."""
    text = NOTE.read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    paste = text[start : text.rindex("\n```")]
    return text[:start], paste


def test_report_output_is_pasted() -> None:
    _, paste = split_note()
    assert paste.startswith("zoon pilot: power calculation")
    assert "[pooled] held-out participants" in paste


def test_every_quoted_number_is_in_the_report_output() -> None:
    """A figure in the prose with three or more decimals must appear in the pasted output."""
    prose, paste = split_note()
    printed = {number.lstrip("+") for number in NUMBER.findall(paste)}
    prose = re.sub(r"https?://\S+", "", prose)  # a DOI is not a figure
    quoted = {number.lstrip("+") for number in NUMBER.findall(prose)}
    assert quoted, "the note quotes no figures"
    assert sorted(quoted - printed) == []
