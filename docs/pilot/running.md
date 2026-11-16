# Running the registered pilot

Use Python 3.12 and uv. Keep the registered checkout clean.
The published raw task files and outside-study files must already be downloaded.
They must match their manifests.
Live calibration and references use US means computed from the outside-study files.
These commands do not download data.

Install the optional live dependency before the run:

```sh
uv sync --extra live
```

Set TYPESAFE_API_KEY and ZOON_VERTEX_PROJECT.
The second family uses application-default authentication through gcloud.
No project name or token belongs in a command, report or transcript.
Each calling step first checks ZOON_VERTEX_PROJECT and authentication using a
non-generating model-metadata request, before sending any question.
A 401 receives one refresh; a repeated 401 or a 403 fails the step resumably.
The operator-fault message names only ZOON_VERTEX_PROJECT.

An optional smoke command sends three unrelated requests to each family:
a binary question, a seven-answer question and a nine-answer question.
The second family uses digits for the seven options and letters for the nine options.
It writes no transcript and uses neither judge materials nor participant records.
It prints the reported versions, probabilities, sum before division, declared-answer
share, tokens and cost. The four content-filter settings are kept together in
SAFETY_SETTINGS; a live smoke test must confirm that the service accepts OFF.

```sh
uv run --extra live python -m zoon.pilot.run smoke
```

The owner creates the git tag registered on the merged registration commit, before judge-check.
Run these commands in order. Replace REGISTERED_COMMIT with the full commit that
the registered tag points to. That commit must be HEAD or an ancestor of HEAD;
every registered file must remain byte-identical. Any code difference requires a recorded
repair covering its full current snapshot.

```sh
uv run --extra live python -m zoon.pilot.run judge-check
uv run --extra live python -m zoon.pilot.run judge
uv run --extra live python -m zoon.pilot.run calibrate
uv run --extra live python -m zoon.pilot.run held-out --registered-commit REGISTERED_COMMIT
uv run python -m zoon.pilot.run report
```

The first four commands need TYPESAFE_API_KEY and ZOON_VERTEX_PROJECT.
The report needs no credentials and makes no model calls.
Use --workers to set the thread pool size. The default is four.
Run only one command at a time.

judge-check checks the outside evaluative correlation.
The decision family must have a finite correlation of at least 0.50.
A failure ends the run as “the judge could not be calibrated”.
A second-family failure removes that family's numerical results.
judge obtains the task ratings, strict features, reversals and recognition replies.
Recognition runs even if the second family's numerical results are unusable.
calibrate fits prompted recalibration and all coded arms on training participants.
Every live step requires a clean work tree and regenerates the materials digests,
request fixtures and cohorts before any question is sent. Report also requires a
clean work tree. Stub commands accept any work tree.
Every live command checks the registered tag, ancestry, registered files and repair
snapshot before doing its work. held-out also checks --registered-commit against the tag
and the earlier gates before loading records.
It is the only step that loads held-out participants.
report writes docs/pilot/result.md and prints the same labelled lines.
After a stopped step, run report to record the outcome.
Report refuses any failed step; repeat that step before reporting.
Commit the report after inspecting it; the report command does not commit it.

Every step checks registration digests and previous state.
A decision-model version change stops the run with no verdict.
A second-family version change or declared-answer share below one half removes its results.
An unreadable or blocked second-family reply is recorded as a share of zero.
A request failure after bounded retries in either family interrupts the step resumably.
Decision probabilities are divided by their sum when it is within 0.05 of one.
Both the returned and used probabilities are retained in the transcript.
Invalid decision replies are retried; exhaustion fails the step resumably.
Recognition replies are retained regardless of those numerical gates.
A reply cut off at the token limit is used as it stands. A blocked or empty reply
is listed as having no reply and does not remove numerical results. A recognition reply
is flagged whenever it contains a registered key, even if it also says unknown.
Every comparison is repeated on the dilemmas that have a reply and no flag.
Flagged dilemmas and dilemmas with no reply are listed separately. If none remain,
the repeated comparisons are not estimable. Nothing is refitted or requested again.
An unusable second family still has its recognition replies and flags reported.
The spending cap is shared across both families and is recomputed from transcripts.
The guard checks it at step entry; request handling checks it before replay or admission
and after each recorded window, before raising any request failure.
Every comparison uses the session's cap, which defaults to 150 dollars.
Report gives no verdict above that cap regardless of the saved step statuses.
Admission reserves the registered estimate for outstanding requests.
A request that would exceed 150 dollars is not sent.
A cap stop gives no verdict.

Transcripts are append-only JSON lines in data/transcripts/live/.
State files are in data/run/live/.
Both are ignored by git. Do not delete either during a run.
Each transcript entry is keyed by the canonical request's SHA-256.
If interrupted, repeat the same command. Completed requests are read from the transcript.
Completed steps return their recorded status.
Transient request failures receive four attempts with delays of one, two and four seconds.
An exhausted decision transport or reply fault records a failed step.
An authentication preflight fault also records a failed step.
Repeat that command after service or authentication recovers.
Completed requests are reused. Later steps refuse a failed prerequisite.
A stopped gate or cap is terminal. Run report for that outcome.
Every live step uses one guard measured from the commit tagged registered, including
report. Its bound set is every tracked file under src/, plus pyproject.toml, uv.lock,
prereg/, pilot/ and the data-manifest*.json files, from both that commit and HEAD.
The guard compares disk bytes with the registered blobs, so hidden edits are seen;
tracked files must also have a clean git status. Added and deleted files count.
Registered files are prereg/, pilot/, the manifests and CSV tables under src/.
A difference in any registered file always refuses continuation, even with --repair.
Code is the rest. Do not reset state to continue a confirmatory run.
If only code differs, commit the repair and keep the work tree clean.
Resume the same command with --repair "Reason for the repair".
The command records the reason, current commit and full difference from the registered commit
for code, including current digests. It reuses recorded answers. The difference must
exactly match a recorded repair snapshot; any further difference needs another repair.
There is no baseline taken from the first step.
The report's Repairs heading lists every repair, or none, and computes afresh the full
list of files differing between the registered commit and the report's commit. It also
reports the registered commit and the commit each step ran at. Readers judge whether
each repair preserved all registered definitions; the guard cannot certify that.

Four things end it with no verdict: the decision model is withdrawn or reports another
version; the spending cap is reached; the complete-records rule fails; a change that is
not a repair is made. The stop rule ends it with its own outcome.
Anything else only interrupts a step: a request that fails or a reply that cannot be read
after the retries, or a fault of authentication in either family. The step is then run
again. It keeps every answer already recorded and sends only the requests that have none,
so running a step again cannot change an answer. A terminal stop wins over a retryable
failure in the same request window; among terminal outcomes the cap comes first.
Spending counts the tokens in recorded responses; discarded unreadable replies and the
six unrelated connection-test requests are not counted.
A partial final transcript line requires recovery of the last complete durable record.
A service reply lost before it was durably recorded cannot be recovered by this program.

For a synthetic dry run, add --stub to each of the five commands.
Use the current full commit for --registered-commit.
No environment variables or downloaded data are needed.
The dry run uses 48 synthetic training participants, eight for calibration and 12 for scoring.
It uses all 13 rotations and the committed materials.
Every printed line starts with a stub label.
It writes data/transcripts/stub/, data/run/stub/ and data/run/stub/result.md.
These files cannot replace the live result.
