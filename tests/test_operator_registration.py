"""Exact registered wording for implemented operator and answer-class behavior."""

from pathlib import Path

import pytest

PINNED = (
    (
        "Both flag reasons are refused before any work if they contain the proj"
        "ect identifier,\nand an ending reason is checked against the client's h"
        "eld bearer before recording."
    ),
    ("The `chat` arm decides nothing."),
    (
        "Three unusable replies for one decision, the model's withdrawal, a\nver"
        "sion change, reaching the chat cap, a failed replay of a chat game, or"
        " a first\nreported version that does not begin with the model name, or "
        "an operator ending,\nmake `chat` unusable with no scores."
    ),
    (
        "A blank reason is refused before any work.\nBoth flag reasons are refus"
        "ed before any work if they contain the project identifier,\nand an endi"
        "ng reason is checked against the client's held bearer before recording"
        ".\nA chat request failure is a transport interruption, a configuration "
        "refusal\n(including an unavailable credential), or a secret refusal of "
        "a reply from the\nmetadata check or a generating request, leaving no re"
        "corded reply and no counted withdrawal;\nthe flag applies to all three "
        "classes.\nFor an unfinished closed-loop or recall step, the flag is acc"
        "epted only after\nthat step saved an interruption by a chat request fai"
        "lure, or an ending of that\nstep is already saved; otherwise repeat the"
        " step without the flag.\nIt ends the arm only if a chat request fails i"
        "n this attempt before any generating\nrequest returns a reply; a return"
        "ed reply uses up the flag before transcript append,\nwhile replies reus"
        "ed from the transcript do not use it up.\nDecision-model and storage fa"
        "ilures, three unusable replies, the cap, a version\nchange and withdraw"
        "al never create an operator ending.\n\nThe run cannot check the operator"
        "'s reason or distinguish a service failure from\nan operator-side fault"
        " such as a broken network or a rejected or missing credential.\nThe end"
        "ing records OPERATOR, the class, stage, last HTTP status, reason, step"
        ",\nthis step's count of recorded chat request failures including this o"
        "ne, and answered\nchat decisions (and answered recall questions in reca"
        "ll); it removes chat scores\nand changes no verdict.\nThe last status is"
        " the final HTTP status of the last attempt after redirects,\nor null if"
        " that attempt received no final HTTP answer.\nAn operator ending comple"
        "tes this step; recall retains and judges its answers\nand marks the res"
        't "not asked".\nOn completion the command prints the chat status and en'
        'ding records, and prints\n"flag had no effect" once if this flagged att'
        "empt has no ending.\nIf a flagged attempt raises after a generating req"
        'uest returned in that attempt,\nit prints "flag used up: a chat request'
        ' succeeded in this attempt" before propagating the failure.\nAn ending '
        "already saved in this unfinished step is retained without new chat req"
        'uests\nand printed as "ending already recorded"; a completed or termina'
        "l step is still refused.\nAn ending during closed-loop does not stop re"
        "call from asking, so a service still\nfailing then needs its own interr"
        "upted attempt and its own flagged attempt.\n"
    ),
    (
        "A reply whose text holds a secret is never recorded.\nThe generated [se"
        "rvice-answer table](service-answers.md) records the outcome class,\nreq"
        "uest count and eligibility of the flag on the next attempt for each an"
        "swer.\n"
    ),
    (
        "| `src/zoon/interaction/steps.py` | Existing today | Closed-loop, reca"
        "ll and operator chat endings |"
    ),
    (
        "| `interaction/chat-requests/` | Existing today | Committed chat reque"
        "st fixtures |\n| `src/zoon/interaction/service_answers.py` | Existing t"
        "oday | Service-answer table regeneration |\n| `interaction/service-answ"
        "ers.md` | Existing today | Generated service-answer outcomes |"
    ),
    (
        "The run module dispatches calibration, closed-loop and recall today. T"
        "he existing spending script\n`src/zoon/interaction/budget.py` produces "
        "both bounds. The second sealing step,\nscoring and intervals, run guard"
        "s, step state, calibration, closed-loop and recall are implemented; he"
        "ld-out and report remain\nopen work. A material-regeneration test must "
        "compare registered instructions,\nfixtures and the amount table with th"
        "eir committed bytes; the fit test uses the\nprobability tolerance above"
        "."
    ),
    (
        "Merge the registration, analysis code and registered materials, then c"
        "reate the\ntag as specified above. Calibration, closed-loop and recall "
        "are implemented today; held-out and report remain requirements.\nRun th"
        "em in this order:"
    ),
    (
        "The transcript cannot show a withdrawal or an operator ending; the run"
        "\nstep records them, and that record governs later steps."
    ),
)


@pytest.mark.parametrize("passage", PINNED)
def test_operator_registration_contract(passage: str) -> None:
    text = (Path(__file__).parents[1] / "interaction/registration.md").read_text()
    assert passage in text
