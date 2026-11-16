"""The fixed report, rendered from completed state and transcripts without model calls."""

from typing import TYPE_CHECKING, cast

import numpy as np

from zoon.pilot import budget, clients, fetch, lean, prompts, score

if TYPE_CHECKING:
    from zoon.pilot.run import Run, State


def display(comparison: score.Comparison | None) -> str:
    if comparison is None:
        return "not estimable"
    return (
        f"mean {comparison.mean:+.8f}; interval [{comparison.interval[0]:+.8f}, "
        f"{comparison.interval[1]:+.8f}]; floor {comparison.floor}; "
        f"lower above zero {comparison.positive}"
    )


def render(run: "Run", states: "dict[str, State]") -> list[str]:
    from zoon.pilot.run import floats

    lines: list[str] = []

    def add(tag: str, key: str, value: object) -> None:
        lines.append(run.line(tag, key, value))

    completed = "held-out" in states and states["held-out"]["status"] == "completed"
    payload = clients.obj(states["held-out"]["payload"]) if completed else {}
    unavailable = clients.obj(payload.get("unavailable", {}))
    logs = {name: floats(value) for name, value in clients.obj(payload.get("logs", {})).items()}
    identities = [
        (int(clients.number(clients.items(i)[0])), str(clients.items(i)[1]))
        for i in clients.items(payload.get("identities", []))
    ]
    dilemma = np.array(cast(list[list[int]], payload.get("dilemma", [])), dtype=np.int64)
    comparisons: dict[str, tuple[str, str]] = {}
    primary: list[score.Comparison] = []
    for family in budget.FAMILIES:
        for state in states.values():
            family_result = clients.obj(clients.obj(state["payload"]).get(family, {}))
            if family_result.get("usable") is False or family_result.get("passed") is False:
                unavailable[family] = family_result.get("reason") or "not usable"
            if (
                family == "decision"
                and state["status"] == "stopped"
                and str(state["reason"]).startswith("decision model")
            ):
                unavailable[family] = state["reason"]
        try:
            run.session.check_family(family)
        except clients.StopRun as error:
            unavailable[family] = str(error)
    if "second" in unavailable:
        logs = {name: value for name, value in logs.items() if not name.startswith("second/")}
    cap_exceeded = run.session.spent > clients.CAP
    if cap_exceeded:
        completed = False
        add("verdict", "outcome", "no verdict: spending cap reached; no verdict")
    elif completed:
        required = [
            "decision/zoon-evaluative",
            "decision/no-state-evaluative",
            "decision/zoon-strict",
            "decision/no-state-strict",
            "decision/prompted",
            "laws-only",
            "sign",
            "sign + learning",
            "coded",
            "coded + learning",
            "norm",
            "norm + learning",
            "known share",
            "known share + learning",
            "pooled",
        ]
        if not set(required) <= logs.keys() or any(
            v.shape != dilemma.shape or not np.isfinite(v).all() for v in logs.values()
        ):
            completed = False
            add("verdict", "outcome", "no verdict: complete-records failure")
        else:
            primary = [
                cast(
                    score.Comparison,
                    score.compare(
                        logs["decision/zoon-evaluative"], logs[other], identities, dilemma
                    ),
                )
                for other in ("decision/prompted", "laws-only", "decision/no-state-evaluative")
            ]
            add("verdict", "outcome", score.verdict(primary))
    else:
        terminal = next(s for s in states.values() if s["status"] != "completed")
        reason = str(terminal["reason"])
        add(
            "verdict",
            "outcome",
            reason if reason == "the judge could not be calibrated" else "no verdict: " + reason,
        )

    for index, other in enumerate(
        ("decision/prompted", "laws-only", "decision/no-state-evaluative")
    ):
        label = f"decision/zoon-evaluative minus {other}"
        add("H1-evaluative", label, display(primary[index]) if primary else "not estimable")
        if completed:
            comparisons[label] = ("decision/zoon-evaluative", other)

    # Items beside the verdict follow the registration's order.
    for family, kind in (("decision", "strict"), ("second", "evaluative"), ("second", "strict")):
        main = f"{family}/zoon-{kind}"
        valid = completed and main in logs and f"{family}/prompted" in logs
        results: list[score.Comparison] = []
        for other in (f"{family}/prompted", "laws-only", f"{family}/no-state-{kind}"):
            label = f"{main} minus {other}"
            comparison = (
                score.compare(logs[main], logs[other], identities, dilemma) if valid else None
            )
            add(f"H1-{kind}", label, display(comparison))
            if comparison is not None:
                comparisons[label] = (main, other)
                results.append(comparison)
        add(
            f"H1-{kind}",
            f"{family} rule",
            score.verdict(results) if len(results) == 3 else "not estimable",
        )

    for name in ("sign", "coded", "norm", "known share"):
        for suffix in ("", lean.LEARNING):
            arm = name + suffix
            add(
                "reference",
                f"{arm} mean log score",
                f"{logs[arm].mean():+.8f}" if completed else "not estimable",
            )
            if completed:
                other = "laws-only" if suffix else "pooled"
                label = f"{arm} minus {other}"
                comparisons[label] = (arm, other)
                add(
                    "reference",
                    label,
                    display(score.compare(logs[arm], logs[other], identities, dilemma)),
                )
        if completed:
            comparisons[f"{name + lean.LEARNING} minus {name}"] = (name + lean.LEARNING, name)
    for family in budget.FAMILIES:
        main = f"{family}/zoon-evaluative"
        label = f"{main} minus sign + learning"
        if completed and main in logs:
            comparisons[label] = (main, "sign + learning")
            add(
                "rating contribution",
                label,
                display(score.compare(logs[main], logs["sign + learning"], identities, dilemma)),
            )
        else:
            add("rating contribution", label, "not estimable")

    for label, (first, second) in comparisons.items():
        for experiment in sorted({i[0] for i in identities}):
            add(
                f"experiment {experiment}",
                label,
                display(
                    score.compare(
                        logs[first], logs[second], identities, dilemma, experiment=experiment
                    )
                ),
            )
        add(
            "trials 2-13",
            label,
            display(score.compare(logs[first], logs[second], identities, dilemma, later=True)),
        )
    if not comparisons:
        add("breakdowns", "by experiment and trials 2-13", "not estimable")
    for label, (first, second) in comparisons.items():
        add("dilemma interval", label, score.dilemma_interval(logs[first], logs[second], dilemma))
    if not comparisons:
        add("dilemma interval", "comparisons", "not estimable")
    for family in budget.FAMILIES:
        for kind in ("evaluative", "strict"):
            main = f"{family}/zoon-{kind}"
            for reference in ("norm + learning", "known share + learning"):
                value: str | float = "not estimable"
                if completed and main in logs:
                    numerator = float((logs[main] - logs["laws-only"]).mean())
                    denominator = float((logs[reference] - logs["laws-only"]).mean())
                    if denominator != 0:
                        value = numerator / denominator
                add(
                    "reading share",
                    f"{main} gain over laws-only / {reference} gain over laws-only",
                    value,
                )

    judged = (
        clients.obj(states["judge"]["payload"])
        if "judge" in states and states["judge"]["status"] == "completed"
        else {}
    )
    recognised = clients.obj(judged.get("recognition", {}))
    add(
        "recall",
        "recognition flags",
        sum(bool(clients.obj(v)["flagged"]) for v in recognised.values())
        if recognised
        else "not available",
    )
    for family in budget.FAMILIES:
        one = clients.obj(judged.get(family, {}))
        if one.get("usable") and family not in unavailable:
            for name, difference in clients.obj(one["reversal"]).items():
                add("reversal", f"{family} {name}, reversed minus original", difference)
            for kind in ("evaluative", "strict"):
                add(
                    "agreement with norm",
                    f"{family} {kind}, Pearson correlation",
                    one[f"norm_correlation_{kind}"],
                )
        else:
            add("recall", f"{family} reversal and agreement with norm", "not available")

    calibrated = (
        clients.obj(states["calibrate"]["payload"])
        if "calibrate" in states and states["calibrate"]["status"] == "completed"
        else {}
    )
    rotations = clients.items(calibrated.get("rotations", []))
    for index, raw in enumerate(rotations):
        row = clients.obj(raw)
        for name, fit in clients.obj(row["arms"]).items():
            if name.startswith("second/") and "second" in unavailable:
                continue
            add("parameter count", f"rotation {index + 1} {name}", clients.obj(fit)["count"])
    if not rotations:
        add("parameter count", "arms", "not fitted")

    # The family tallies occur here in the registered beside-verdict list and are repeated
    # with digests at the end so that the complete accounting is easy to copy.
    def accounting() -> None:
        for family, transcript in run.session.transcripts.items():
            responses = [e.response for e in transcript.entries.values()]
            add("spending", f"{family} calls", len(responses))
            add("spending", f"{family} input tokens", sum(r.input_tokens for r in responses))
            add("spending", f"{family} output tokens", sum(r.output_tokens for r in responses))
            add("spending", f"{family} USD", f"{transcript.spent:.8f}")

    accounting()

    for name in fetch.DILEMMAS:
        record = clients.obj(recognised[name]) if name in recognised else None
        add(
            "recognition",
            f"{name} reply and flag",
            json_reply(record) if record is not None else "not run",
        )
    keep = [
        i
        for i, name in enumerate(fetch.DILEMMAS)
        if name in recognised and not clients.obj(recognised[name])["flagged"]
    ]
    for label, (first, second) in comparisons.items():
        add(
            "recognition filtered",
            label,
            display(score.compare(logs[first], logs[second], identities, dilemma, keep=keep)),
        )
        add(
            "recognition filtered dilemma interval",
            label,
            score.dilemma_interval(logs[first], logs[second], dilemma, keep),
        )
        for experiment in sorted({i[0] for i in identities}):
            add(
                f"recognition filtered experiment {experiment}",
                label,
                display(
                    score.compare(
                        logs[first],
                        logs[second],
                        identities,
                        dilemma,
                        keep=keep,
                        experiment=experiment,
                    )
                ),
            )
        add(
            "recognition filtered trials 2-13",
            label,
            display(
                score.compare(logs[first], logs[second], identities, dilemma, keep=keep, later=True)
            ),
        )
    if not comparisons:
        add("recognition filtered", "comparisons", "not estimable")
    for index, raw in enumerate(rotations):
        for name, raw_fit in clients.obj(clients.obj(raw)["arms"]).items():
            if name.startswith("second/") and "second" in unavailable:
                continue
            add("fitted", f"rotation {index + 1} {name}", json_reply(raw_fit))
    for family in budget.FAMILIES:
        check = clients.obj(
            clients.obj(states.get("judge-check", {}).get("payload", {})).get(family, {})
        )
        if family not in unavailable:
            add("judge calibration", f"{family} correlation and gate", json_reply(check))
        one = clients.obj(judged.get(family, {}))
        if one.get("usable") and family not in unavailable:
            penalty = one["penalty"]
            add(
                "strict map",
                f"{family} result",
                "these descriptive questions carried no usable lean"
                if penalty == "inf"
                else "finite map selected",
            )
            weights = floats(one["map_weights"])
            add(
                "strict map",
                f"{family} nonzero outside weights plus offset",
                1 + int(np.count_nonzero(weights)),
            )
            for key in (
                "penalty",
                "map_offset",
                "map_weights",
                "finite_penalty",
                "finite_error",
                "finite_correlation",
                "finite_task_ratings",
                "agreement_outside",
                "agreement_task",
            ):
                add("strict map", f"{family} {key}", json_reply(one[key]))
        cal = clients.obj(calibrated.get(family, {}))
        for index, raw_fit in enumerate(clients.items(cal.get("fits", []))):
            if family == "second" and family in unavailable:
                continue
            add(
                "fitted",
                f"rotation {index + 1} {family}/prompted, two parameters",
                json_reply(raw_fit),
            )
        reason = (
            unavailable.get(family) or cal.get("reason") or one.get("reason") or check.get("reason")
        )
        add(
            "family",
            f"{family} availability",
            reason or ("usable" if check.get("passed") else "not assessed"),
        )
    for family, transcript in run.session.transcripts.items():
        grouped: dict[str, list[float]] = {
            kind: [] for kind in ("evaluative", "strict", "prompted")
        }
        for entry in transcript.entries.values():
            if isinstance(entry.request, str):
                continue
            document = clients.obj(entry.request)
            question = str(
                document.get("question")
                or str(document["text"]).rsplit("\n\n", 1)[-1].splitlines()[0]
            )
            kind = (
                "prompted"
                if question == prompts.PREDICT
                else "evaluative"
                if question.startswith("How appropriate")
                else "strict"
            )
            share = entry.response.share
            if share is None and family == "decision":
                share = sum(entry.response.probabilities.values())
            if share is not None:
                grouped[kind].append(share)
        for kind, shares in grouped.items():
            add(
                "declared answers",
                f"{family} {kind} share count/min/mean/max/below half",
                (
                    f"{len(shares)}/{min(shares):.8f}/{np.mean(shares):.8f}/"
                    f"{max(shares):.8f}/{sum(s < 0.5 for s in shares)}"
                )
                if shares
                else "not available",
            )
    accounting()
    for family, transcript in run.session.transcripts.items():
        add("transcript", f"{family} SHA-256", transcript.digest())
        add(
            "versions",
            family,
            ", ".join(sorted({e.response.version for e in transcript.entries.values()})) or "none",
        )
    from zoon.pilot.run import git

    registered = payload.get("registered_commit")
    if not registered:
        try:
            registered = git(
                run.root, "rev-parse", "HEAD" if run.synthetic else "refs/tags/registered^{commit}"
            )
        except clients.StopRun:
            registered = "not recorded"
    add("run", "registered commit", registered)
    add("run", "report commit", git(run.root, "rev-parse", "HEAD"))
    for step, state in states.items():
        add("run", f"{step} commit", state["commit"])
    repairs = clients.items(run.binding()["repairs"])
    if not repairs:
        add("Repairs", "history", "none")
    for index, raw in enumerate(repairs, 1):
        repair = clients.obj(raw)
        add(
            "Repairs", str(index), json_reply({k: repair[k] for k in ("reason", "commit", "files")})
        )
    return lines


def json_reply(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
