"""The pilot notes quote only numbers that their pasted report output contains."""

import re
from pathlib import Path

import pytest

NOTES = Path(__file__).parent.parent / "docs" / "pilot"
NUMBER = re.compile(r"[+-]?\d+\.\d{3,}")
# Each note that pastes report output, with the first line of that output and a line that
# only a complete run prints.
PASTED = {
    "power.md": ("zoon pilot: power calculation", "[pooled] held-out participants"),
    "lean.md": (
        "zoon pilot: a lean from published human norms and codes",
        "[experiment 4] the whole thing | coded + learning minus pooled + learning",
    ),
}


def split_note(name: str) -> tuple[str, str]:
    """Return the prose and the pasted report output, which is the last code block."""
    text = (NOTES / name).read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    paste = text[start : text.rindex("\n```")]
    return text[:start], paste


@pytest.mark.parametrize("name", PASTED)
def test_report_output_is_pasted(name: str) -> None:
    _, paste = split_note(name)
    first, later = PASTED[name]
    assert paste.startswith(first)
    assert later in paste


@pytest.mark.parametrize("name", PASTED)
def test_every_quoted_number_is_in_the_report_output(name: str) -> None:
    """A figure in the prose with three or more decimals must appear in the pasted output.
    Protects the entry for each note in PASTED: without it, a figure in that note that the
    report never printed would pass."""
    prose, paste = split_note(name)
    printed = {number.lstrip("+") for number in NUMBER.findall(paste)}
    prose = re.sub(r"https?://\S+", "", prose)  # a DOI is not a figure
    quoted = {number.lstrip("+") for number in NUMBER.findall(prose)}
    assert quoted, "the note quotes no figures"
    assert sorted(quoted - printed) == []
