"""The run instructions state the same boundaries as the registration."""

from zoon.pilot import fetch


def test_instructions_describe_registration_repairs_interruptions_and_recognition() -> None:
    text = (fetch.REPO / "docs/pilot/running.md").read_text()
    for phrase in (
        "before judge-check",
        "every tracked file under src/",
        "full difference from the registered commit",
        "including current digests",
        "Four things end it with no verdict",
        "Anything else only interrupts a step",
        "dilemmas that have a reply and no flag",
        "listed separately",
    ):
        assert phrase in text
    assert "re-binds" not in text
    assert "service failure after bounded retries removes" not in text
    assert "recognition not run" not in text
