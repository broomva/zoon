"""Exact registered wording for implemented operator and answer-class behavior."""

from pathlib import Path

import pytest

PINNED = (
    ("The `chat` arm decides nothing."),
    (
        "Three unusable replies for one decision, the model's withdrawal, a\n"
        "version change, reaching the chat cap, a failed replay of a chat game, or a "
        "first\n"
        "reported version that does not begin with the model name, or an operator end"
        "ing,\n"
        "make `chat` unusable with no scores."
    ),
    (
        "The operator may use `python -m zoon.interaction.run closed-loop --end-chat "
        "REASON`\n"
        "or `python -m zoon.interaction.run recall --end-chat REASON`.\n"
        "A blank reason is refused before any work.\n"
        "For a resumable step, the flag is accepted only after a saved chat transport"
        "\n"
        "interruption of that step, or when that step already has a saved ending.\n"
        "Otherwise repeat the step without the flag; a later chat transport interrupt"
        "ion\n"
        "makes the flag acceptable on the next attempt.\n"
        "The flag ends the arm only on a chat transport interruption in this attempt,"
        "\n"
        "including its metadata check, before any generating request returns a reply."
        "\n"
        "A returned reply uses up the flag before transcript append, so a later chat "
        "transport\n"
        "failure interrupts the step; replies reused from the transcript do not use i"
        "t up.\n"
        "A decision-model failure cannot create an operator ending.\n"
        "\n"
        "The ending records OPERATOR, the reason, step, this step's count of recorded"
        " chat\n"
        "transport interruptions including this one, and number of answered chat deci"
        "sions;\n"
        "recall also records the number of answered recall questions.\n"
        "Service-client endings also record the failed stage and last HTTP status.\n"
        "The last status is the final HTTP status of the last attempt after redirects"
        ",\n"
        "or null if that attempt received no final HTTP answer.\n"
        "An operator ending makes chat unusable with no scores and completes this ste"
        "p;\n"
        'recall retains and judges its answers and marks the rest "not asked".\n'
        "On completion the command prints the chat status and ending records, and pri"
        "nts\n"
        '"flag had no effect" once if this flagged attempt created no ending.\n'
        "If a flagged attempt raises after a generating request returned in that atte"
        "mpt,\n"
        'it prints "flag used up: a chat request succeeded in this attempt" before pr'
        "opagating the failure.\n"
        "An ending already saved in this resumable step is retained without new chat "
        "requests;\n"
        "a completed or terminal calling step is still refused.\n"
        "Apart from a declared repair, the incomplete-line cut and restoring a guarde"
        "d output\n"
        "under Guard, repairs and persistence, this is the only operator decision the"
        " run\n"
        "allows, and it cannot be taken once `held-out` has started."
    ),
    (
        "The client has five outcome classes: an accepted generating reply is returne"
        "d\n"
        "for recording and uses up the flag; withdrawal counted on every bounded gene"
        "rating\n"
        "attempt makes chat unusable; a transport interruption leaves the request unr"
        "ecorded\n"
        "and interrupts the step unless the eligible flag ends the arm; configuration"
        " and\n"
        "guard refusals leave the request unrecorded and interrupt the step without e"
        "nding the arm.\n"
        "The generated [service-answer table](service-answers.md) records which answe"
        "rs fall\n"
        "in each class, their request counts, and whether a persistent answer leaves "
        "the flag\n"
        "acceptable on the next attempt.\n"
        "The request format is checked against the live service before the registrati"
        "on is\n"
        "tagged, so a persistent generating refusal in the transport-interruption cla"
        "ss is\n"
        "treated during the registered run as a service change on which the operator "
        "may end the arm.\n"
        "\n"
        "An uncorrectable configuration refusal, an unavailable credential, or a pers"
        "istent\n"
        "guard refusal from a service putting the project identifier in every process"
        "able\n"
        "generating reply cannot be ended by the flag and can block this step and hel"
        "d-out.\n"
        "Correct the configuration or obtain the credential and repeat without the fl"
        "ag;\n"
        "for persistent guard refusals there is no permitted way around the secret ch"
        "eck.\n"
        "For a JSON-object generating answer entering response processing, the secret"
        " check\n"
        "runs first on all readable text parts joined together: a secret refuses the "
        "reply\n"
        "even if processing would otherwise fail, with no record or use of the flag.\n"
    ),
    (
        "| `src/zoon/interaction/steps.py` | Existing today | Closed-loop, recall and"
        " operator chat endings |"
    ),
    (
        "| `interaction/chat-requests/` | Existing today | Committed chat request fix"
        "tures |\n"
        "| `src/zoon/interaction/service_answers.py` | Existing today | Service-answe"
        "r table regeneration |\n"
        "| `interaction/service-answers.md` | Existing today | Generated service-answ"
        "er outcomes |"
    ),
    (
        "The run module dispatches calibration, closed-loop and recall today. The exi"
        "sting spending script\n"
        "`src/zoon/interaction/budget.py` produces both bounds. The second sealing st"
        "ep,\n"
        "scoring and intervals, run guards, step state, calibration, closed-loop and "
        "recall are implemented; held-out and report remain\n"
        "open work. A material-regeneration test must compare registered instructions"
        ",\n"
        "fixtures and the amount table with their committed bytes; the fit test uses "
        "the\n"
        "probability tolerance above."
    ),
    (
        "Merge the registration, analysis code and registered materials, then create "
        "the\n"
        "tag as specified above. Calibration, closed-loop and recall are implemented "
        "today; held-out and report remain requirements.\n"
        "Run them in this order:"
    ),
    (
        "The transcript cannot show a withdrawal or an operator ending; the run\n"
        "step records them, and that record governs later steps."
    ),
)


@pytest.mark.parametrize("passage", PINNED)
def test_operator_registration_contract(passage: str) -> None:
    text = (Path(__file__).parents[1] / "interaction/registration.md").read_text()
    assert passage in text
