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

The owner creates the git tag registered on the merged registration commit.
Run these commands in order. Replace REGISTERED_COMMIT with the full commit that
the registered tag points to. That commit must be HEAD or an ancestor of HEAD;
every registered file must remain byte-identical. A later HEAD requires a recorded
repair covering the current code digests.

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
held-out checks the earlier gates, registered tag, ancestry, registered files,
repair coverage and clean work tree before loading records, including on direct calls.
It is the only step that loads held-out participants.
report writes docs/pilot/result.md and prints the same labelled lines.
After a stopped step, run report to record the outcome.
Commit the report after inspecting it; the report command does not commit it.

Every step checks registration digests and previous state.
A decision-model version change stops the run with no verdict.
A second-family version change or declared-answer share below one half removes its results.
An unreadable or blocked second-family reply is recorded as a share of zero.
A second-family service failure after bounded retries removes only that family's results.
Decision probabilities are divided by their sum when it is within 0.05 of one.
Both the returned and used probabilities are retained in the transcript.
Invalid decision replies are retried; exhaustion fails the step resumably.
Recognition replies are retained regardless of those numerical gates.
The spending cap is shared across both families.
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
Registered-file digests and code digests are bound separately.
A change to the pre-registration, rewrites and digest, task texts and digest,
cohorts, request fixtures or data manifests always refuses further work.
Do not reset state to continue a confirmatory run.
If only bound code has changed, commit the repair and keep the work tree clean.
Resume the same command with --repair "Reason for the repair".
The command records the reason, current commit and changed code files, re-binds
the code digests and reuses recorded answers. A later code change needs another repair.
The report's Repairs heading lists every repair, or none, and reports the registered
commit and the commit each step ran at. Readers judge whether each repair preserved
all registered definitions; the digest guard cannot certify that.
A partial final transcript line requires recovery of the last complete durable record.
A service reply lost before it was durably recorded cannot be recovered by this program.

For a synthetic dry run, add --stub to each of the five commands.
Use the current full commit for --registered-commit and keep the checkout clean.
No environment variables or downloaded data are needed.
The dry run uses 48 synthetic training participants, eight for calibration and 12 for scoring.
It uses all 13 rotations and the committed materials.
Every printed line starts with a stub label.
It writes data/transcripts/stub/, data/run/stub/ and data/run/stub/result.md.
These files cannot replace the live result.
