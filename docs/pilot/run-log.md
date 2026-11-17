# Log of the registered run

This is the operator's record of how the registered pilot was run. The result itself is
[result.md](result.md), which the report command wrote and which is committed as written.
The rules are in [prereg/h1.md](../../prereg/h1.md). Times are UTC.

## What was run

| Step | When | What happened |
|---|---|---|
| Tag and publication | 2026-11-16 20:42 | The tag `registered` was put on commit `97d9256` and the repository was made public. No judge had been called and no held-out participant had been read |
| 2. `judge-check` | 2026-11-16 20:42 | Completed. The decision model passed the stop rule. The second family was excluded: one of its 25 replies put no probability on the declared answers |
| 3. `judge` | 2026-11-16 20:42 to 20:43 | Completed |
| 4. `calibrate` | 2026-11-16 20:43 to 20:59 | Interrupted. See below |
| 4. `calibrate`, repeated | 2026-11-16 21:00 | Interrupted again at once, for the same reason |
| 4. `calibrate`, repeated | 2026-11-17 03:20 to 03:27 | Completed |
| 5. `held-out` | 2026-11-17 03:27 to 03:42 | Completed. This is the step that read the held-out participants |
| 6. `report` | 2026-11-17 03:43 | Completed. It wrote `result.md` |

Every step ran at the registered commit. No repair was made, and the report's own list of
repairs and of files differing from the registered commit is empty.

## The interruption at step 4

The decision model's service stopped answering part of the way through step 4. Its reply
was that the account had no credit left. The pre-registration counts a fault of
configuration as an interruption and not as an ending: the step is run again, keeps every
answer already recorded, and sends only the requests that have none.

Credit was added on a second account and the same command was repeated with that account's
key. The model and the version it reported were the same before and after, as the version
rule requires and as the report shows. Answers recorded before the interruption were not
asked for again.

The first run of step 4 used 12 workers and the later ones 8. Results do not depend on the
number of workers; a test in this repository checks that.

## The second family

The second family's results are not given, as the pre-registration requires when any of its
replies puts less than half its probability on the declared answers. Its recognition replies
are in the report. This is a loss: the verdict rests on one model family.

## Reading the result

The verdict is the one the pre-registration defines: H1-evaluative, with the decision model
as the judge. It passed. The pre-registration says what a pass means and what it does not,
under "What a pass means", and that section was written before any of this was known. Four
things in the report bear on it and should be read with the verdict:

- **H1-strict failed.** The descriptive questions carried no usable lean, as they had not
  with the published codes. The hypothesis the design first wrote is this one.
- **The intervals over dilemmas include zero** for the comparisons with `prompted` and with
  `laws-only`. The intervals that decide are over participants. So the result holds for
  these 13 dilemmas and says little about others, as the pre-registration warned.
- **The comparison with `no-state` passed, as expected.** The pre-registration said that it
  would say little about the judge.
- **Two rewrites were flagged by the recognition check.** The comparisons repeated without
  them are in the report. A flag is a string match and shows nothing about recall either way.

## Records kept

The transcripts are not in this repository. Their digests are in the report, and the files
are kept by the operator.
