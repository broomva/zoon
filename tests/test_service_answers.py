"""Generated service classes use the real client, HTTP parser and redirect handler."""

import pytest

from zoon.interaction import chat, service_answers
from zoon.interaction.service_answers import Answer


def test_service_answer_table_regenerates() -> None:
    assert service_answers.render() == service_answers.TABLE.read_bytes()


@pytest.mark.parametrize("stage", service_answers.STAGES)
@pytest.mark.parametrize(
    "answer,category,requests,acceptable",
    [
        (Answer(200), "reply accepted", (2, 2), False),
        (Answer(204), "transport interruption", (4, 5), True),
        (Answer(200, "not a JSON object"), "transport interruption", (4, 5), True),
        (Answer(200, "empty body"), "transport interruption", (4, 5), True),
        (Answer(200, "no model version"), None, (2, 5), None),
        (Answer(200, "bad token counts"), None, (2, 5), None),
        (Answer(200, "secret in text"), None, (2, 2), None),
        (Answer(302, "redirect to reachable"), "reply accepted", (3, 3), False),
        (Answer(302, "redirect to unreachable"), "transport interruption", (8, 9), True),
        (Answer(None), "transport interruption", (4, 5), True),
    ],
)
def test_http_response_shapes_and_redirects(
    stage: str,
    answer: Answer,
    category: str | None,
    requests: tuple[int, int],
    acceptable: bool | None,
) -> None:
    generating = stage == "generating request"
    if category is None:
        category = (
            ("secret refusal" if answer.kind == "secret in text" else "transport interruption")
            if generating
            else "reply accepted"
        )
    if acceptable is None:
        acceptable = generating
    assert service_answers.observe(stage, answer) == service_answers.Outcome(
        category, requests[int(generating)], acceptable
    )


def test_format_check_uses_the_real_request_and_response_path() -> None:
    with service_answers.network("metadata check", [Answer(200)]) as requests:
        client = service_answers.client()
        client.preflight()
        lines = chat.smoke(client)
    assert len(lines) == 12
    assert requests[0].startswith("GET ")
    assert len(requests) == 13 and all(line.startswith("POST ") for line in requests[1:])
    assert all("usable = True" in line for line in lines)


@pytest.mark.parametrize("stage", service_answers.STAGES)
@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("target", ["reachable", "unreachable"])
def test_redirect_methods_follow_the_http_handler(stage: str, status: int, target: str) -> None:
    if stage == "generating request":
        expected = service_answers.Outcome("configuration refusal", 2, True)
    elif target == "reachable":
        expected = service_answers.Outcome("reply accepted", 3, False)
    else:
        expected = service_answers.Outcome("transport interruption", 8, True)
    assert service_answers.observe(stage, Answer(status, "redirect to " + target)) == expected


@pytest.mark.parametrize("stage", service_answers.STAGES)
@pytest.mark.parametrize(
    "answer,category,counts",
    [
        (Answer(200, "body timeout"), "transport interruption", (4, 5)),
        (Answer(302, "redirect to loop"), "configuration refusal", (5, 6)),
        (Answer(200, "non-finite number"), "transport interruption", (2, 5)),
        (Answer(200, "non-finite number with secret"), "transport interruption", (2, 5)),
        (Answer(401, "refreshed credential succeeds"), "reply accepted", (3, 3)),
        (Answer(401, "refreshed credential rejected"), "transport interruption", (4, 5)),
    ],
)
def test_extended_http_answers(
    stage: str, answer: Answer, category: str, counts: tuple[int, int]
) -> None:
    if stage == "metadata check" and "non-finite" in answer.kind:
        category = "reply accepted"
    result = service_answers.observe(stage, answer)
    assert result == service_answers.Outcome(
        category, counts[int(stage == "generating request")], category != "reply accepted"
    )
