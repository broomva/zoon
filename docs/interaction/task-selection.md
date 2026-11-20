# Choosing the data for the first interaction test

Surveyed on 2026-11-20. The pilot tested one simulated person on a task where nobody
interacts. The next test needs people who affect one another, and it needs two comparisons
against human data:

1. **Open loop.** Predict each held-out person's next action in a repeated group game, given
   what the other members of the group actually did in earlier rounds.
2. **Closed loop.** Simulate whole groups from the first round with no human data inside the
   loop, and compare the simulated groups with held-out human groups.

Both are also run with language-model agents prompted to play the game, since that is how
simulated societies are usually built.

**Recommendation: the linear public goods games of Wang, Hilbe and Zhang (2026), first
session only.** The change that adds this note also splits the groups, and keeps the held-out
ones out of the file the analysis reads.
When the split was made, nothing about any group's behaviour had been computed. What has been
measured since, on training groups only, is in `measure.md`.

Verification key: **[me]** fetched and read, or computed from the files, by the author;
**[R]** read by a research pass and quoted from it, not yet checked independently; **[U]** not
opened or not confirmed.

## The data

| Property | Value |
|---|---|
| Data | https://zenodo.org/records/16918146, published 2025-08-21, licence "cc-by-4.0" in the record's metadata [me] |
| Paper | Wang, Hilbe and Zhang, "The dynamics of cooperation in asymmetric public goods games", PNAS 123(5), e2525760123, https://doi.org/10.1073/pnas.2525760123 [me] |
| Participants | Undergraduates at one university in Beijing, in a computer lab, with no communication allowed. Instructions were in Chinese; the appendix prints an English translation for one treatment at each group size [me] |
| The game | Each round every player receives points and decides how many to put in a common pool. Each contributed point is multiplied by that player's productivity, and the pool is split evenly. Everyone then sees every contribution and payoff [me] |
| Groups | Fixed for a session. Pairs in five treatments; groups of four in three treatments [me] |
| Rounds | "At least 20", then a 50% chance of another after each round. The first 20 are published for groups of four and 22 for pairs; the authors analyse the first 20 [me] |
| Sessions | Every participant plays twice: a second session in a new group with the roles swapped [me] |
| Also collected | After both sessions, each participant states a preferred contribution pattern and how much they would give for each possible contribution of the other player [me] |

### The eight games

Endowments and productivities are Table S1 (pairs) and Table S15 (fours) of the paper's
appendix [me]. In a group of four, seats 1 and 2 hold the first type and seats 3 and 4 the
second. Productivities are doubled for four players, so a contributed point returns the same
amount to its contributor at both sizes.

| Game | Endowments | Productivities | What differs between the players |
|---|---|---|---|
| `2-FE` | 24, 24 | 1.6, 1.6 | Nothing |
| `2-EI` | 36, 12 | 1.6, 1.6 | Endowment |
| `2-PI` | 24, 24 | 1.9, 1.3 | Productivity |
| `2-AI` | 36, 12 | 1.9, 1.3 | Both; the richer player is the more productive |
| `2-MI` | 36, 12 | 1.3, 1.9 | Both; the richer player is the less productive |
| `4-FE` | 24 each | 3.2 each | Nothing |
| `4-AI` | 36, 36, 12, 12 | 3.8, 3.8, 2.6, 2.6 | Both, aligned |
| `4-MI` | 36, 36, 12, 12 | 2.6, 2.6, 3.8, 3.8 | Both, misaligned |

The instructions' own worked example checks these numbers: under aligned inequality, if
everyone contributes everything the pool is 84 points for a pair and 336 for four, and every
player ends the round with the same payoff. `tests/test_interaction_games.py` asserts it.

### What is used, and what is left unread

- **Used: the linear game, first session, first 20 rounds.** 353 groups: 275 pairs and 78
  groups of four.
- **Not read: the second session.** The same people play it. The file for groups of four gives
  a participant a new identifier in the second session, so a second-session group cannot be
  kept apart from the first-session groups its members came from. A test of whether a
  person's state carries over to a new group could use it later.
- **Not read: the threshold game**, where a fixed reward is paid once contributions pass a
  threshold. It is a different game and is kept for a later test.
- **Not read: the questionnaire answers.** They are the participants' own statements of what
  is fair. They could later anchor a model's reading of the rules, as published ratings did
  in the pilot.

### The split

A group is the unit. Its members played together and saw each other's contributions, so their
records are not independent, and a group is never divided between
training and held-out data. Within each game the groups are ranked by a salted hash of their
identifier in the file, and the first two fifths, rounded up, are held out. The files are
pinned by hash in `interaction/data-manifest.json`, so the assignment cannot move.

| Game | Training groups | Held-out groups |
|---|---|---|
| `2-FE` | 34 | 23 |
| `2-EI` | 33 | 22 |
| `2-PI` | 33 | 22 |
| `2-AI` | 31 | 22 |
| `2-MI` | 33 | 22 |
| `4-FE` | 15 | 10 |
| `4-AI` | 15 | 11 |
| `4-MI` | 16 | 11 |
| Total | 210 | 143 |

`uv run --extra groups python -m zoon.interaction.data` downloads the two raw files, checks
them against the manifest, writes the training file and prints this table.

How the held-out groups are kept out of the analysis:

- One function, `seal`, is the only code that opens the raw files. It writes
  `data/groups/training.json`: the training groups, and a count of held-out groups per game.
- Everything else, the measurement included, reads only that file. A test deletes the raw
  files, disables the libraries that read them, and loads.
- Both raw files and the training file are pinned by hash in the manifest. A changed raw
  file could change which groups rank as held out, so each function that opens one checks
  it first; the loader checks the training file.
- For the spreadsheet of pairs, `seal` asks only for the identifier columns and the round
  cells of training pairs; a test records every range asked for. The file for groups of
  four is decoded whole by its reader library, and `seal` keeps training rows only.

This guards against reading a held-out group by accident. It is not a defence against the
person running the code: the raw files are public and can be opened with other code, or
with this module's private functions.

Two fifths is more than the pilot held out. The comparisons here are between groups, there
are few groups, and the models will have few fitted numbers, so precision at evaluation is
the scarcer resource.

## What was known before the split was made

The split was written after reading the paper, and a published paper describes everyone,
including the groups now held out. What was read:

- The paper's summary of results: average contributions by treatment (about 70% of the
  endowment in three pair treatments and about 55% under misaligned inequality), that
  contributions are "relatively stable" over rounds with a rising trend in two pair
  treatments and a falling one for four players under full equality, and that richer players
  give more points and a smaller share.
- The appendix table of average contributions by session for groups of four (59% to 70%).
- The authors' own model: players who respond to the other's last contribution and prefer
  equal shares, with its two fitted weights.

**The published averages constrain the held-out groups.** The paper gives first-session
averages for each treatment over all groups. Anyone holding those and the training groups
can work out the average of the held-out groups of each game. So the held-out level of each
game is not unknown in the way a single held-out group is. The guard against using it is
procedural: every arm is fitted on training groups by a procedure fixed in the
pre-registration, and no number is set by hand. What the published averages cannot give is
what the tests are about: who did what in which round, and how the members of a group moved
relative to each other.

From the files themselves, only structure was read before the split: sheet and column names,
identifiers, counts of rows, groups and rounds, and whether cells were empty. No contribution
of any group was printed or summarised. Since the split, the analysis has read training groups
only.

## Risks, in order of seriousness

1. **Few independent groups.** 143 held-out groups, 32 of them of four. Whether that can
   detect what the others' history adds was estimated on training groups; the figures and
   their assumptions are in `measure.md`.
2. **There may be little for a language model to read.** The eight games differ by four
   numbers. A coded model that takes those numbers as input may predict a game it has not
   seen as well as any reading of the rules. The pilot met the same problem and it changed
   the hypothesis; it has to be measured here too.
3. **The classic decline is mostly absent.** Public goods games are known for contributions
   that fall over time. Here the horizon is uncertain and the paper reports stable
   contributions in most treatments. What a closed loop has to reproduce is how members of a
   group move toward each other, and how far groups end up apart.
4. **Up to 37 possible contributions.** An arm that takes its probabilities from a model's
   declared answers cannot spread them over 37 options with two decimals each. The response
   will probably have to be scored in bands, fixed before registration.
5. **One lab, one country, translated instructions.** A model reads an English rendering of
   Chinese instructions. Only two of the eight are printed; the others are described as
   analogous.
6. **Recall.** The data has been public since August 2025 and the paper since 2026. That is
   later than most models' training, which lowers the chance of recall and does not remove
   it. The general finding that people reciprocate in such games is in every textbook.

## Other candidates

| Dataset | Why it was not chosen first |
|---|---|
| Alsobay, Rand, Watts and Almaatouq, public goods games across a design space, https://osf.io/2d56w/ | The largest and most varied: 836 games in its two game tables [me]. No licence is stated on the deposit [R]. More than half the games have free-text chat, half have a punishment stage and half a reward stage, and players who stop responding are removed mid-game [me]. Fewer than a hundred games have none of the three [me]. Other groups have already run language-model simulations and likelihood benchmarks on it [R]. It is the natural second stage, once the coded person has laws for punishment and reward |
| Mao, Dworkin, Suri and Watts 2017, repeated prisoner's dilemma, https://osf.io/64z8u/ | CC BY 4.0 [R]. 113 people play thousands of ten-round games over 20 days with changing partners [R], so pairs are not independent of one another |
| Akata et al. 2025, repeated two-by-two games | The human participants played against language models, not each other [R] |
| Mikus et al. 2023, repeated trust game | The partner was programmed [R] |
| Herrmann, Thöni and Gächter 2008, public goods with punishment in 16 participant pools | The deposit is marked CC0 but its files could not be downloaded on the day [R] |
| Fischbacher and Gächter 2010 | Groups were re-formed at random after every period, and no data deposit was found [R] |
| Psych-101 | No group game with recoverable group membership was found in it [R] |

## Earlier work on the same comparison

All [R]: found and quoted by the research pass, not yet read by the author.

- Na, Zhu, Cross and Bakker 2026, https://arxiv.org/abs/2611.00661: language-model agents
  with behavioural profiles, evaluated against held-out groups of the Alsobay experiment.
- Binz et al. 2026, https://arxiv.org/abs/2605.07632: next-response likelihood for language
  models on many tasks, the Alsobay experiment among them.
- Akata et al. 2025, https://doi.org/10.1038/s41562-025-02172-y: language models in repeated
  games against coded strategies and people.
- Guzman Piedrahita et al. 2025, https://arxiv.org/abs/2506.23276, and Teo 2026,
  https://arxiv.org/abs/2606.27845: language models in repeated public goods games compared
  with published human patterns. The second reports that human contributions fell where the
  models' stayed flat or rose.
- Xie et al. 2024, https://arxiv.org/abs/2402.04559: language-model agents in trust games.

No comparison was found that puts a coded person with a model in the loop, a prompted model
and held-out human groups side by side, both open and closed loop. The search was one pass on
one day. No novelty is claimed on that basis.
