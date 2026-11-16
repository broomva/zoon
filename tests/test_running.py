"""The run instructions state the same boundaries as the registration."""

from zoon.pilot import fetch


def test_instructions_describe_registration_repairs_interruptions_and_recognition() -> None:
    text = (fetch.REPO / "docs/pilot/running.md").read_text()
    for phrase in (
        "before judge-check",
        "every tracked file at the tag or at HEAD",
        "full difference from the registered commit",
        "including current digests",
        "closed run-ending reasons",
        "Anything else only interrupts a step",
        "dilemmas that have a reply and no flag",
        "listed separately",
    ):
        assert phrase in text
    assert "re-binds" not in text
    assert "service failure after bounded retries removes" not in text
    assert "recognition not run" not in text


def test_instructions_describe_guard_order_and_bound_artifacts() -> None:
    text = (fetch.REPO / "docs/pilot/running.md").read_text()
    for phrase in (
        "even above the cap",
        "last recorded repair snapshot",
        "first line records the registered commit and mode",
        "one atomic state file",
        "report can only be completed or failed",
        "complete-records failure in a second-family arm",
        "every attempt at one request",
    ):
        assert phrase in text
    assert "connection metadata found in a response" not in text
