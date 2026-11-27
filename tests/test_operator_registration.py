"""Exact registered wording for implemented operator and answer-class behavior."""

from pathlib import Path

import pytest

PINNED = (
    ("The `chat` arm decides nothing."),
    (
        "Three unusable replies for one decision, the model's withdrawal, a\nver"
        "sion change, reaching the chat cap, a failed replay of a chat game, or"
        " a first\nreported version that does not begin with the model name, or "
        "an operator ending,\nmake `chat` unusable with no scores."
    ),
    (
        "A blank reason is refused before any work.\nA chat request failure is a"
        " transport interruption, a configuration refusal\n(including an unavail"
        "able credential), or a secret refusal of a reply from the\nmetadata che"
        "ck or a generating request, leaving no recorded reply and no counted w"
        "ithdrawal;\nthe flag applies to all three classes.\nFor an unfinished cl"
        "osed-loop or recall step, the flag is accepted only after\nthat step sa"
        "ved an interruption by a chat request failure, or an ending of that\nst"
        "ep is already saved; otherwise repeat the step without the flag.\nIt en"
        "ds the arm only if a chat request fails in this attempt before any gen"
        "erating\nrequest returns a reply; a returned reply uses up the flag bef"
        "ore transcript append,\nwhile replies reused from the transcript do not"
        " use it up.\nDecision-model and storage failures, three unusable replie"
        "s, the cap, a version\nchange and withdrawal never create an operator e"
        "nding.\n\nThe run cannot check the operator's reason or distinguish a se"
        "rvice failure from\nan operator-side fault such as a broken network or "
        "a rejected or missing credential.\nThe ending records OPERATOR, the cla"
        "ss, stage, last HTTP status, reason, step,\nthis step's count of record"
        "ed chat request failures including this one, and answered\nchat decisio"
        "ns (and answered recall questions in recall); it removes chat scores\na"
        "nd changes no verdict.\nThe last status is the final HTTP status of the"
        " last attempt after redirects,\nor null if that attempt received no fin"
        "al HTTP answer.\nAn operator ending completes this step; recall retains"
        ' and judges its answers\nand marks the rest "not asked".\nOn completion '
        'the command prints the chat status and ending records, and prints\n"fla'
        'g had no effect" once if this flagged attempt has no ending.\nIf a flag'
        "ged attempt raises after a generating request returned in that attempt"
        ',\nit prints "flag used up: a chat request succeeded in this attempt" b'
        "efore propagating the failure.\nAn ending already saved in this unfinis"
        'hed step is retained without new chat requests\nand printed as "ending '
        'already recorded"; a completed or terminal step is still refused.\nAn e'
        "nding during closed-loop does not stop recall from asking, so a servic"
        "e still\nfailing then needs its own interrupted attempt and its own fla"
        "gged attempt.\n"
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
