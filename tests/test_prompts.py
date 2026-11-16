"""Character-level request contracts and reproducible training-only fixtures."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from tests.synthetic import write_raw
from zoon.pilot import cohorts, fetch, materials, outside, prompts, rewrites, split


def test_judge_requests() -> None:
    record = next(r for r in rewrites.load() if r["id"] == "Medicine")
    for reverse in (False, True):
        text = str(record["reversed" if reverse else "rewrite"])
        action = str(record["reversed_suggested_action" if reverse else "suggested_action"])
        expected_state = text[: text.rindex("Do you ")].rstrip()
        evaluative = prompts.evaluative(record, reverse=reverse)
        assert evaluative.state == expected_state
        assert evaluative.question == f"How appropriate is it to {action}?"
        assert [(a.answer, a.description) for a in evaluative.answers] == [
            ("1", "1 (completely inappropriate)"),
            ("2", "2"),
            ("3", "3"),
            ("4", "4"),
            ("5", "5"),
            ("6", "6"),
            ("7", "7 (completely appropriate)"),
        ]
        assert prompts.second(evaluative) == (
            expected_state + "\n\n" + evaluative.question + "\n" + prompts.SCALE
        )
        for i, question in enumerate(prompts.STRICT):
            request = prompts.strict(record, i, reverse=reverse)
            assert request.state == expected_state
            if i < 4:
                assert request.question == question + f" The suggested action is: {action}."
                assert request.answers == (prompts.Answer("Yes", "Yes"), prompts.Answer("No", "No"))
                assert prompts.second(request) == (
                    expected_state + "\n\n" + request.question + "\n" + prompts.YES_NO
                )
            else:
                assert request.question == question
                assert tuple(a.answer for a in request.answers) == prompts.contexts()
                lines = prompts.second(request).splitlines()
                assert lines[-10:-1] == [
                    f"{letter}. {context}"
                    for letter, context in zip("ABCDEFGHI", prompts.contexts(), strict=True)
                ]
                assert lines[-1] == "Answer with one letter."
    assert prompts.recognition(record) == str(record["rewrite"]) + "\n\n" + prompts.RECOGNITION
    assert rewrites.question(str(record["rewrite"]))[0].rstrip() == expected_state


def test_exact_strict_words_and_context_order() -> None:
    assert prompts.STRICT == (
        "Does the suggested action involve killing a person?",
        "Are the people who would be harmed by the suggested action among the people who would "
        "benefit from it?",
        "Would the person deciding benefit personally from the suggested action?",
        "Would it be reasonable to describe this dilemma the other way round, asking about the "
        "opposite action?",
        "Which of these contexts does the dilemma belong to?",
    )
    seen: list[str] = []
    for dilemma in outside.published().values():
        if dilemma.codes is not None:
            for context in dilemma.codes.context.split("/"):
                if context.strip() not in seen:
                    seen.append(context.strip())
    assert prompts.contexts() == tuple(seen) == outside.CONTEXTS


def test_prompted_characters_and_target_is_not_read() -> None:
    tasks: dict[str, materials.Task] = {
        name: {
            "text": f"Text {name}?",
            "outcomes": {
                answer: {"success": "Good.", "failure": "Bad."} for answer in ("Yes", "No")
            },
        }
        for name in fetch.DILEMMAS
    }
    trials = tuple(cohorts.Trial(name, "Yes", True, 41) for name in fetch.DILEMMAS)
    person = cohorts.Participant(1, "synthetic", trials)
    assert prompts.trial(trials[0], tasks) == (
        "Dilemma:\nText Ladder?\nAnswer: Yes\nWhat happened: Good.\nRating: 41"
    )
    shown = prompts.example(person, 2, "A", tasks)
    assert shown == (
        "Person A:\nDilemma:\nText Ladder?\nAnswer: Yes\nWhat happened: Good.\nRating: 41"
        "\n\nDilemma:\nText Tyran?\nAnswer: Yes"
    )
    request = prompts.prompted(person, 1, shown, tasks)
    assert request.state == (
        prompts.INTRO
        + "\n\nHere are records from other people in the same study.\n\n"
        + shown
        + "\n\nHere is the record of the person you are asked about.\n\n"
        "This person has not answered any dilemma yet.\n\nCurrent dilemma:\nText Ladder?"
    )
    request = prompts.prompted(person, 5, shown, tasks)
    assert request.state.count("What happened:") == 5  # one example and four own trials
    changed = replace(
        person, trials=(*trials[:4], replace(trials[4], answer="No", rating=-99), *trials[5:])
    )
    assert prompts.prompted(changed, 5, shown, tasks) == request
    assert request.question == "Which answer does this person give to the current dilemma?"
    assert prompts.second(request).endswith(request.question + "\nAnswer with one word: Yes or No.")
    # A person's own history retains every earlier trial, including rotation-held-out ones.
    assert all(f"Text {t.dilemma}?" in request.state for t in trials[:4])


def test_fixture_generation_with_sealed_synthetic_data(tmp_path: Path) -> None:
    sealed = frozenset(
        (e, str(i))
        for e in fetch.SOURCES
        for i in range(1, 221)
        if split.assign(e, str(i)) == "held_out"
    )
    write_raw(tmp_path, 220, unreadable=sealed)
    people = cohorts.participants(fetch.load(tmp_path))
    document = cohorts.generate(people)
    made = prompts.fixtures(people, document, materials.load(), rewrites.load())
    assert len(made) == 17
    assert made == prompts.fixtures(people, document, materials.load(), rewrites.load())
    assert "second-recognition.txt" in made and "decision-recognition.json" not in made
    rotation = document["rotations"][0]
    rotation["held_out"] = list(fetch.DILEMMAS)
    with pytest.raises(ValueError, match="held-out"):
        prompts.examples(rotation, people, materials.load())


def test_committed_fixtures_regenerate() -> None:
    if not all((fetch.RAW_DIR / s.filename).exists() for s in fetch.SOURCES.values()):
        pytest.skip(
            "training raw files absent; request fixtures require the published training data"
        )
    made = prompts.fixtures(
        cohorts.load_training(), cohorts.load(), materials.load(), rewrites.load()
    )
    committed = {
        p.name: p.read_bytes() for p in prompts.DIRECTORY.iterdir() if p.name != "README.md"
    }
    assert made == committed


def test_invalid_requests() -> None:
    with pytest.raises(ValueError, match="index"):
        prompts.strict({}, 5)
    with pytest.raises(ValueError, match="unsupported"):
        prompts.second(prompts.Decision("state", "question", prompts.declared(("Maybe",))))
    person = cohorts.Participant(1, "synthetic", ())
    with pytest.raises(ValueError, match="current"):
        prompts.prompted(person, 1, "examples", {})
    with pytest.raises(ValueError, match="first trial"):
        prompts.example(person, 1, "A", {})
    with pytest.raises(ValueError, match="two examples"):
        prompts.examples(cast(cohorts.Rotation, {"held_out": [], "examples": []}), [], {})
