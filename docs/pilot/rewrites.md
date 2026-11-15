# Draft de-identified dilemmas for the pilot judge

These are **drafts**, awaiting the author’s comparison against the source texts and the
maintainer’s reading. They are not fixed stimuli and have not been sent to either judge.
The 38 JSON files in [`pilot/rewrites/`](../../pilot/rewrites/) cover the task’s 13 dilemmas
and the 25 other factual dilemmas in Körner and Deutsch’s set. They implement the rewrite
proposal in [the arms note](arms.md#what-the-judge-is-asked).

The judge is to read only a file’s `rewrite` (or its `reversed` for the reversal check), not
its ID, source, lists or notes. Those fields are for researchers checking the materials.
The IDs deliberately match the repository’s existing names and therefore still identify
cases. The outside texts support the judge-only check before the task’s held-out data is
read; they are not additional participant trials.

## Purpose and limits

The source dilemmas describe historic events, and human ratings of the originals are
published. Recognising an event could let a judge retrieve a rating. Removing explicit
identifiers reduces those cues but does not establish that recall is absent. Event
structure, preserved numbers and causal details can still identify a case. A drafting
model can also have encountered the same sources.

**No human has rated these rewrites, so nothing shows that people judge them as they judge
the originals.** Removing historical context can change perceived realism and moral
judgment. Agreement with published ratings could reflect either judgment or recall. A
successful pilot would support a prediction claim about this pipeline, not an inference
that the judge reasoned afresh.

## Drafting and review

A language model from a vendor other than the two judge families drafted these rewrites.
It worked from the task PDF’s vignette and question for the 13 task dilemmas, and from the
English section of the original study’s `factualDilemmas-texts-appendix.pdf` for the other
25. All 25 English texts were obtained through the original study’s OSF materials at
[the appendix download](https://osf.io/download/ephu6/). The source PDFs and extracted texts
were used temporarily and are not committed. No outcome text was used to draft a rewrite.
No participant data or language-model API was used in preparing these files.

The author must compare each draft, including each list of preserved and removed facts,
with its specified source. The maintainer must then read each version before the stimuli
are fixed and committed for judge calls. The mechanical checks below do not replace those
steps. In particular, they do not determine semantic equivalence, whether two questions
are true opposites, or whether an omitted cue matters morally.

## Rules applied

- Preserve the trade-off, the decision-maker’s generic role, who acts on whom, the suggested
  action, and the consequences of taking it or refusing it. Preserve numbers of people,
  other harms and resource amounts, hedges and conditional probabilities, personal benefit,
  overlap between victims and beneficiaries, and whether harm is direct and done by the
  decision-maker. Do not infer a benefit or group membership that the vignette leaves open.
- Remove the incident’s explicit identifiers: names, calendar dates and years, countries,
  cities, named organisations, ships and events, and other case-specific cues where removal
  does not change the trade-off. Substitute a generic setting of the same kind and retain
  the same role. Describe unavoidable losses or remaining distinctive details in `notes`.
- Use second person and a final yes/no question about the suggested action. Keep the length
  within 75–125% of the source vignette plus question. Do not add facts, reasons, moral
  justifications or emotional colour. Emotional descriptions already in the source remain.
- Use the task’s adapted version for the 13, including its role changes, lung operations,
  additional consequences and Endow’s approval question. Do not substitute the original
  paper’s opposing question for Endow. [The mapping](lean.md#the-mapping) explains the changes.
- Add a reversal exactly where Tables 1 and 2 of the original paper code opposite framing
  as `Yes`, `Rather yes` or `Somewhat` (case-insensitive). Qualified positive answers are
  included; `No` and `Not really` are excluded. For both sets use the original paper’s
  code, rather than the task’s adaptation codes. Thus Tax and Bishops have reversals;
  RobinHood does not. A reversal changes only the final question and has its own short
  `reversed_suggested_action` clause. Appropriateness of opposite actions need not mirror.

The original outside texts end with an appropriateness-rating question, rather than a
binary choice. Their final questions are converted to “Do you …?” about the same action.
The judge’s evaluative question and response scale are specified separately in the arms
note; these binary questions identify the action and do not replace that rating scale.
Where an English vignette already omits names and dates, its heading is withheld and its
wording is paraphrased; `removed` records the heading rather than inventing absent names.

## Files and mechanical validation

Each JSON object contains `id`, `set`, `source`, `suggested_action`, `kept`, `removed` and
`rewrite`, with `notes` for review concerns and the two reversal fields when applicable.
Source citations identify the authors, source location, CC BY 4.0 licence and modifications.
`kept` includes the stated and unspecified aspects of self-benefit, group overlap and agency.
`removed` lists withheld source strings; it is not a proof that every identifying cue is gone.

[`pilot/source-word-counts.json`](../../pilot/source-word-counts.json) commits counts only.
For both sources, counts include the complete vignette and question, but no heading, answer
labels, outcomes, citations or section headings. PDF line breaks are normalised to spaces;
words are whitespace-separated, so a hyphenated expression counts as one word. The line-wrapped
“co-conspirators” is joined in the original text before counting. No full source vignette
is included in this repository addition.

Run the offline validator from the repository root:

```sh
PYTHONPATH=src uv run python -m zoon.pilot.rewrites
```

It checks required fields and types, non-empty fact lists, source-count coverage, exact task
and outside ID coverage against the committed tables, unique IDs, set membership, removal
strings (case-insensitive with Unicode normalisation), four-digit year-like numbers, both
length bounds, paper-based reversal presence, and unchanged reversal bodies. Four-digit
altitudes are spelled out without changing their numerical values. It reads no source PDF,
network resource, participant file or analysis result. Tests corrupt records to exercise
these guards, including both length limits and both versions.

## Inventory

Counts include the final question. A dash means no reversal. Kept and removed are counts of
list entries, not a count of all distinct facts.

| ID | Set | Source words | Rewrite words | Reversed words | Kept | Removed |
|---|---|---:|---:|---:|---:|---:|
| Conjoined Twins | outside | 167 | 134 | — | 6 | 1 |
| Cannibalism on Lifeboat | outside | 168 | 128 | — | 6 | 3 |
| Doodlebugs | outside | 173 | 134 | — | 6 | 4 |
| Plague Spread | outside | 160 | 123 | — | 6 | 3 |
| Euthanasia (in Ghetto) | outside | 129 | 104 | — | 6 | 4 |
| Immunization (Smallpox) | outside | 184 | 151 | — | 6 | 4 |
| 911 | outside | 149 | 116 | — | 6 | 5 |
| Coma (Mother) | outside | 145 | 123 | — | 6 | 2 |
| Organ Law | outside | 227 | 179 | 179 | 6 | 2 |
| Stillbirths | outside | 133 | 101 | — | 6 | 3 |
| Septuplets | outside | 141 | 109 | — | 6 | 2 |
| Atom Bomb | outside | 132 | 106 | — | 6 | 5 |
| Mare Nostrum | outside | 206 | 171 | 171 | 6 | 5 |
| Aid for Syria | outside | 174 | 136 | 138 | 6 | 3 |
| Rescue Torture | outside | 132 | 114 | — | 6 | 1 |
| War Torture | outside | 160 | 120 | — | 6 | 6 |
| Rugby Cannibalism | outside | 165 | 139 | — | 6 | 4 |
| Organ Market | outside | 203 | 162 | — | 6 | 1 |
| Ashley Treatment | outside | 196 | 155 | 157 | 6 | 2 |
| Test Tube Sibling | outside | 177 | 139 | 141 | 6 | 2 |
| Lion Hunt | outside | 173 | 144 | — | 6 | 1 |
| Bishops | outside | 152 | 118 | 119 | 6 | 8 |
| Trail of Tears | outside | 163 | 126 | 126 | 6 | 8 |
| Hogesa (Hooligan Demo) | outside | 91 | 72 | 72 | 6 | 3 |
| Mass Rape | outside | 160 | 120 | 123 | 6 | 2 |
| Endow | task | 120 | 98 | 98 | 6 | 1 |
| Ladder | task | 199 | 162 | — | 6 | 4 |
| Lifeboat | task | 227 | 174 | — | 6 | 1 |
| Medicine | task | 163 | 130 | 132 | 6 | 1 |
| Operations | task | 206 | 167 | — | 7 | 1 |
| RAF | task | 113 | 103 | 105 | 6 | 3 |
| Ransom | task | 139 | 116 | 118 | 6 | 2 |
| RobinHood | task | 173 | 136 | — | 6 | 2 |
| Rwanda | task | 148 | 140 | 141 | 6 | 3 |
| Suicide | task | 171 | 147 | 143 | 6 | 2 |
| Tax | task | 144 | 114 | 116 | 6 | 1 |
| Tyran | task | 127 | 108 | — | 6 | 2 |
| Vet | task | 161 | 125 | 125 | 6 | 1 |

## Notes requiring author and maintainer attention

- **Conjoined Twins**: The English vignette already omits names, places and dates; its case heading is withheld. The shared bladder and unequal organ function remain because they explain the dependency, although they may identify the case.
- **Cannibalism on Lifeboat**: The year, destination and named ocean are removed, but the yacht, four crew and seventeen-year-old cabin boy remain. Age, rank and group size bear on the choice and also make this case unusually recognisable.
- **Doodlebugs**: Countries, the war’s name and the city are removed. North/south geometry, class differences and military targets remain because they explain how deception reallocates risk; together they may still identify the incident.
- **Plague Spread**: The named disease becomes a contagious, deadly epidemic. Its transmission mechanism and predicted consequences are retained, but disease-specific expectations could change. The priest’s role and 800 inhabitants remain despite their recognition risk.
- **Euthanasia (in Ghetto)**: The named ghetto, ethnicity, year and occupying regime become a generic occupied ethnic ghetto. Deportation and forced-labour exclusion remain, but removing the specific persecutors and persecuted group loses historical context that could affect judgment.
- **Immunization (Smallpox)**: The two named diseases become a harmless disease and a dangerous disease. Their stated severity, contagiousness, sequence and uncertain immunity are retained. Disease names may carry relevant prior knowledge; the gardener relationship and age may still identify the case.
- **911**: Routes, cities and the named building are generic. The four hijackers, two prior aircraft and passenger resistance are retained because they establish evidence, agency and risk; this structure is especially likely to identify the event.
- **Coma (Mother)**: The riding accident becomes an unspecified accident; its resulting condition and seven-year duration remain. The accident’s kind might affect perceived realism, although the source gives no causal role for it in the decision.
- **Stillbirths**: The named camp, year and guards’ organisation are removed. Concentration-camp imprisonment and the guards’ orders remain essential to the constrained choice, but can still cue the historical persecution.
- **Atom Bomb**: The named regimes, countries and year become a hostile dictatorship and a country of residence. The resident physicist, first-bomb rumour and deterrence argument remain because they define the choice, but strongly suggest the historical case.
- **Mare Nostrum**: Named regions, institutions and programme are replaced by a sea crossing and a multinational rescue effort. The two published counts remain as scale and evidence; the specific migration context and legal institutions are lost.
- **Aid for Syria**: The country and named international organisation become generic. The organisation’s specific mandate and the conflict’s identities are lost; regime control and the source’s suspicions remain.
- **Rescue Torture**: The English vignette already omits names, places and dates; only its heading is withheld. The banker’s son, ransom collection and two-day custody remain because status and urgency could matter, although they may identify the case.
- **War Torture**: Nationalities, year and the convention’s name are removed. Its binding protection of prisoners is retained, but readers lose the named treaty and wartime context, which could affect how they interpret the prohibition.
- **Rugby Cannibalism**: The named route, year and sport are removed; a sports team remains. Counts, altitude, temperature and the end of the search are preserved because they describe survival constraints, but make recognition particularly likely. Numbers that would look like years are spelled out without changing their values.
- **Ashley Treatment**: Nationality and the case heading’s personal name are removed. Diagnosis, developmental level and procedures remain because they explain the care constraint and bodily harm, although their combination may identify the child’s case.
- **Test Tube Sibling**: The named blood disorder becomes a deadly blood disease and the court’s nationality is removed. Donor requirements and the conditional recovery probability remain, but disease-specific knowledge and the legal setting are lost.
- **Bishops**: Named religions, ancestry, country, occupier and year become a religion and a persecuted heritage. The protected subgroup’s membership in both categories remains, but removing these particular identities and the archbishop’s denomination may change the perceived duty or agreement.
- **Trail of Tears**: Named people, settler race, country, year and direction become an indigenous people, settlers and relocation within a country. Constitutional protection and the favourable court ruling remain. Removing specific political identities and geography loses relevant historical context.
- **Hogesa (Hooligan Demo)**: The named extremist movement, country and targeted religious identity are removed. Right-wing affiliation, opposition to religious extremism and past violence remain; generalising the religion may alter how the group’s aims are perceived.
- **Mass Rape**: Place and period are removed. The source says at least one is innocent yet describes many others as “probably all” guilty; that unresolved wording is retained without silently assigning guilt. The age and eight-person group remain possible recognition cues.
- **Endow**: The source vignette already omits names, places and dates; only its case heading is withheld. The distinctive offender-named foundation and 25,000 euro sum remain because changing either could change the trade-off. The task’s approval question is retained.
- **Lifeboat**: The year is removed, but the iceberg, 41 occupants and ten proposed deaths are preserved because they describe the hazard and trade-off. Together they may still identify the incident. The task’s second-boat captain and timescale of days are retained.
- **Medicine**: Nationality is removed while the age, rare immune disease and cost remain. Those facts may still identify the case. The task’s independent estimate of more deaths is retained, rather than replacing it with the original paper’s wording.
- **Operations**: The task changes knee to lung surgery but retains one reference to knee reimbursement. That inconsistency is preserved for author review rather than silently corrected. The named month becomes the final month of the year; the annual deadline, 50/48 threshold and two operations remain.
- **RAF**: The national setting and association’s formal name are replaced with generic equivalents. The source’s far-left affiliation and the hostage’s employers’ association role remain because political identity and social status could matter; they may also cue recognition.
- **Ransom**: The named desert becomes a desert of unspecified location. The task says ten hostages and also one prior death; this possible counting ambiguity is preserved rather than resolved.
- **RobinHood**: The paper codes opposite framing “No”, although the task’s table says “Yes”; no reversal is included under the paper-based rule. National setting and the case heading are removed; the bank role and fraud mechanism remain.
- **Rwanda**: Named ethnic groups become an attacking group and a targeted group. Their historical relationship is lost. The task threatens only the targeted ethnicity inside, not everyone inside; the rewrite does not infer your ethnicity or change that narrower threat.
- **Suicide**: The national legislature becomes a generic upper chamber. The six-month criterion and 139 million dollar estimate are retained as substantive eligibility and resource facts, although the combination may identify the proposal.
- **Tax**: The paper calls opposite framing “somewhat” reasonable, although the task’s table says “No”; a reversal is included under the paper-based rule. The task’s added possibility of encouraging future cybercrime is retained.
- **Tyran**: The ruler’s nationality and period are removed. Political and ethnic persecution, the family power structure and the number of conspirators remain; these may still identify the incident.

## Attribution

The 13 task drafts adapt Maier, Cheung and Lieder, *dilemmas and outcomes.pdf*, in the
materials for *Metacognitive learning from consequences of past choices shapes moral
decision-making* (2025), [OSF project 4up5z](https://osf.io/4up5z/),
[PDF](https://osf.io/download/sgf5y/),
[paper](https://doi.org/10.1038/s41562-025-02271-w).
The materials are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Their vignettes were themselves adapted from Körner and Deutsch.

The 25 outside drafts adapt the English vignettes in Körner and Deutsch (2022),
*Deontology and Utilitarianism in Real Life: A Set of Moral Dilemmas Based on Historic
Events*, *Personality and Social Psychology Bulletin*, 49(10), 1511–1528,
[article](https://doi.org/10.1177/01461672221103058),
[open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10478346/),
[materials](https://osf.io/cg5tq/), [English appendix PDF](https://osf.io/download/ephu6/).
The article and its appendix are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The reversal codes come from the paper’s Tables 1 and 2, as transcribed in
[`korner_deutsch_2022.csv`](../../src/zoon/pilot/korner_deutsch_2022.csv).

These files are adaptations: identifying details have been removed or generalised,
wording has been changed, questions use yes/no form, and selected questions have opposite
action variants. These changes are not endorsed by the source authors. The CC BY 4.0
attribution and licence apply to these adapted stimulus materials, separately from the
repository’s software licence.
