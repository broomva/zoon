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
identifiers and generalising incidental details reduces those cues but does not establish that recall is absent. Event
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

The author must compare each draft, including each list of preserved, removed and generalised facts,
with its specified source. The maintainer must then read each version before the stimuli
are fixed and committed for judge calls. The mechanical checks below do not replace those
steps. In particular, they do not determine semantic equivalence, whether two questions
are true opposites, or whether an omitted cue matters morally.

## Rules applied

Keep the moral choice the same. Preserve who does what to whom, numbers of people on each
side, the kind of harm, and every qualification of an outcome: “probably”, “most likely”,
“sooner or later”, conditional recovery and other hedges must keep their force. Preserve
whether the decider acts directly, benefits personally, or harms people who are also among
those benefiting. Keep the role that carries moral responsibility, such as a professional’s
duty of care, authority to approve an action, or a bystander’s opportunity to intervene.
Keep the suggested action and its final yes/no form, including which option is the
cost-benefit choice. Add no facts, motives or emotional colour.

Remove names, dates, places and other explicit identifiers. Also generalise details that
help identify an incident without setting the trade-off: political colour, exact offices,
institution types, disease subtypes, accident causes, precise ages where childhood suffices,
particular drugs or procedures, and incidental durations and measurements. Preserve the
same kind and severity of harm. An incidental count of collaborators, trial sites, prior
aircraft or transactions can become “a few”, “several” or “tens of thousands”; counts of
people on the competing sides cannot change. People counts used as evidence about future
killing remain in RAF as well. Where a detail is morally necessary, retain it and explain
why below, even if it makes the case recognisable.

Exact sums that only indicate scale become a substantial or large sum, while the stated
consequence of spending or losing that money stays. If a sum’s size itself sets a choice,
it must stay, rounded without changing the comparison. Here, the sources already state the
opportunity costs or sufficiency: other patients’ reduced care and likely extra deaths,
better treatment for many others, help to crime victims, or enough conservation land.
No currency conversion or new monetary threshold is introduced.

Use second person. Both versions must stay within **70–130%** of the source vignette plus
question. This replaces the previous 75–125% validator bound to allow about 30% variation
when incidental details are generalised. Length never justifies dropping a morally relevant
fact. Counts use whitespace-separated words, including the final question.

Use the task’s adapted version for its 13 dilemmas, including its role changes, additional
consequences and Endow’s approval question. [The mapping](lean.md#the-mapping) explains those
adaptations. Operations now consistently describes **a type of operation**: both the lung
references and the inherited single “knee surgeries” slip become the same general operation.
The 50/48 annual reimbursement rule, two unnecessary operations, two patients and deaths
sooner or later remain. No surgical complication or risk absent from the vignette is added.

Add a reversal exactly where Tables 1 and 2 of the original paper code opposite framing as
`Yes`, `Rather yes` or `Somewhat` (case-insensitive); exclude `No` and `Not really`. Use the
paper’s code for both sets, so Tax and Bishops have reversals and RobinHood does not. Here
the bodies, including every sentence naming an action, are **identical** between versions;
only the final question changes. This is stricter than allowing action-naming sentences to
change. Opposite actions need not have mirror-image appropriateness ratings.

The outside sources’ appropriateness questions become “Do you …?” about the same action.
These binary questions identify the action; the judge’s separate evaluative question and
response scale remain as specified in the arms note. An already anonymous source’s heading
is withheld and recorded in `removed`, without inventing absent names.

## Files and mechanical validation

Each JSON object has `id`, `set`, `source`, `suggested_action`, `kept`, `removed`, `generalised`,
`rewrite` and `notes`, plus the two reversal fields where applicable. `generalised` is a list
of objects with `from` (a short phrase as the source states it), `to` (replacement or an
explicit omission), and `why` (why the detail does not set the trade-off). It records every
generalisation in this revision, including those applied to wording already paraphrased in
the previous draft. Earlier removals remain in `removed`. Neither list is a guarantee that
recognition is impossible. Empty `generalised` lists require notes beginning “No further
generalisation:” and explaining why nothing incidental remains.

[`pilot/source-word-counts.json`](../../pilot/source-word-counts.json) contains only counts.
They include each complete source vignette and question, excluding headings, answers,
outcomes, citations and section headings. PDF line breaks are normalised to spaces; a
hyphenated expression counts as one word, and line-wrapped “co-conspirators” is joined.
No source vignette is committed.

[`pilot/people-counts.json`](../../pilot/people-counts.json) records the people on each side
as given by the specified source, including singular victims and donors, lower bounds,
and the source magnitudes “thousands” and “millions”. Each entry names a side, its count and
one or more phrases expressing that count in the rewrite. Singular references such as
“the boy” express one person. Three remaining crew in Cannibalism on Lifeboat is explicitly
labelled as implied by four minus the boy. No other unstated total is computed. Unnumbered
sides are described separately; an empty count list means unspecified counts or animal
stakes, never zero people harmed. Incidental attacker and collaborator counts are excluded;
RAF’s historical release and reoffending counts remain as numerical evidence for the risk.
The 50/48 surgery counts and six-month eligibility condition remain in the text but are not
people counts.

Run the offline validator from the repository root:

```sh
PYTHONPATH=src uv run python -m zoon.pilot.rewrites
```

It checks required fields and types, source citations and licence, fact lists, the structure
and explanation of generalisations, exact task/outside coverage against the committed
source tables, and coverage and structure of both count tables. Neither version may contain
a `removed` string, a generalised `from` phrase, or a four-digit year-like number. Text
comparisons use Unicode NFKC normalisation, case folding and collapsed whitespace. Both
versions must meet the 70–130% length bound. Every recorded people-side count must appear as
one of its contextual phrases, with word boundaries so that 140 cannot stand in for 40.
Finally it checks paper-based reversal presence, different final yes/no questions and exact
body equality. Corruption tests exercise each new guard, every recorded side in both
applicable versions, and the existing year/removal checks separately from generalisation
checks. Documentation tests compare the inventory, notes and full task generalisation lists
with the committed files.

These checks read no PDF, network resource, participant file or analysis result. They can
catch missing phrases, not establish correct allocation, semantic equivalence, preserved
moral judgment or successful de-identification. Author comparison and maintainer reading
remain pending. **No human has rated these rewrites.**

## Inventory

Counts include the final question. A dash means no reversal. Kept, removed and generalised
count list entries, rather than all distinct facts.

| ID | Set | Source words | Rewrite words | Reversed words | Kept | Removed | Generalised |
|---|---|---:|---:|---:|---:|---:|---:|
| Conjoined Twins | outside | 167 | 124 | — | 6 | 1 | 2 |
| Cannibalism on Lifeboat | outside | 168 | 131 | — | 6 | 3 | 2 |
| Doodlebugs | outside | 173 | 140 | — | 6 | 4 | 4 |
| Plague Spread | outside | 160 | 124 | — | 6 | 3 | 1 |
| Euthanasia (in Ghetto) | outside | 129 | 108 | — | 6 | 4 | 2 |
| Immunization (Smallpox) | outside | 184 | 151 | — | 6 | 4 | 2 |
| 911 | outside | 149 | 111 | — | 6 | 5 | 4 |
| Coma (Mother) | outside | 145 | 124 | — | 6 | 2 | 2 |
| Organ Law | outside | 227 | 177 | 177 | 6 | 2 | 2 |
| Stillbirths | outside | 133 | 102 | — | 6 | 3 | 1 |
| Septuplets | outside | 141 | 108 | — | 6 | 2 | 0 |
| Atom Bomb | outside | 132 | 112 | — | 6 | 5 | 2 |
| Mare Nostrum | outside | 206 | 169 | 169 | 6 | 5 | 2 |
| Aid for Syria | outside | 174 | 136 | 138 | 6 | 3 | 1 |
| Rescue Torture | outside | 132 | 113 | — | 6 | 1 | 1 |
| War Torture | outside | 160 | 119 | — | 6 | 6 | 1 |
| Rugby Cannibalism | outside | 165 | 122 | — | 6 | 4 | 3 |
| Organ Market | outside | 203 | 162 | — | 6 | 1 | 1 |
| Ashley Treatment | outside | 196 | 156 | 158 | 6 | 2 | 4 |
| Test Tube Sibling | outside | 177 | 137 | 139 | 6 | 2 | 1 |
| Lion Hunt | outside | 173 | 145 | — | 6 | 1 | 2 |
| Bishops | outside | 152 | 120 | 121 | 6 | 8 | 1 |
| Trail of Tears | outside | 163 | 127 | 127 | 6 | 8 | 1 |
| Hogesa (Hooligan Demo) | outside | 91 | 71 | 71 | 6 | 3 | 2 |
| Mass Rape | outside | 160 | 124 | 127 | 6 | 2 | 2 |
| Endow | task | 120 | 98 | 98 | 6 | 1 | 1 |
| Ladder | task | 199 | 161 | — | 6 | 4 | 1 |
| Lifeboat | task | 227 | 170 | — | 6 | 1 | 1 |
| Medicine | task | 163 | 129 | 131 | 6 | 1 | 3 |
| Operations | task | 206 | 173 | — | 6 | 1 | 2 |
| RAF | task | 113 | 98 | 100 | 6 | 3 | 4 |
| Ransom | task | 139 | 114 | 116 | 6 | 2 | 5 |
| RobinHood | task | 173 | 137 | — | 6 | 2 | 0 |
| Rwanda | task | 148 | 139 | 140 | 6 | 3 | 1 |
| Suicide | task | 171 | 141 | 137 | 6 | 2 | 3 |
| Tax | task | 144 | 112 | 114 | 6 | 1 | 3 |
| Tyran | task | 127 | 109 | — | 6 | 2 | 1 |
| Vet | task | 161 | 125 | 125 | 6 | 1 | 0 |

## Full generalisation record for the 13 task dilemmas

The phrases on the left are short source details, not excerpts of complete vignettes.

### Endow

- `25,000 euros` → `a substantial donation` — the exact amount only signals the scale of help.

### Ladder

- `rope ladder` → `ladder` — the exact ladder material does not change blocking or pushing.

### Lifeboat

- `colliding with an iceberg` → `sank` — the cause of sinking does not change the lifeboat choice.

### Medicine

- `7-year-old boy` → `young boy` — childhood and dependence on family funding remain.
- `very rare immunological disease` → `rare, fatal disease` — regular medicine and certain death without it retain the medical stakes.
- `approximately 200,000 euros per year` → `a very large annual sum` — the stated loss of other care and likely additional deaths define the opportunity cost.

### Operations

- `lung surgeries` → `a type of operation` — the organ treated does not set unnecessary surgery versus access to emergency care.
- `knee surgeries` → `operations of this type` — this inconsistent organ reference concerns the same reimbursement threshold.

### RAF

- `far-left terrorist group` → `armed group` — political colour and label do not change hostage-taking or lethal attacks.
- `President of the Federation of the German Employers’ Association` → `prominent public figure` — the exact office does not change the hostage’s life at stake.
- `former terrorists` → `imprisoned members` — the same prisoners are released without an additional political label.
- `terrorist attacks` → `attacks` — future lethal conduct remains explicit without the group label.

### Ransom

- `Sahara Desert` → `location omitted` — the terrain does not change the failed rescue or lethal threat.
- `group of terrorists` → `an armed group` — kidnapping, weapons and future killing retain the conduct at stake.
- `half a year` → `months` — prolonged failed rescue and the one prior death remain.
- `several million euros` → `a large ransom` — the stated weapons funding and likely future deaths define the cost.
- `future terrorism` → `future attacks` — weapons funding and subsequent killing remain the consequence.

### RobinHood

No further generalisation: the bank role, temporary undetectable transfer, credit rating, wealthy depositors, indebted families and betrayed trust define the fraud and its beneficiaries; no incidental identifying detail remains. The paper codes opposite framing “No”, so no reversal is included despite the task’s different code.

### Rwanda

- `development aid team` → `aid project` — responsibility for employees and their families remains.

### Suicide

- `Canadian Senate` → `legislator` — law-making authority remains without a specific chamber.
- `tranquillizers` → `medication` — a doctor’s fatal overdose remains the direct harm.
- `139 million dollars` → `a large sum` — the stated improvement of treatment for many others defines the benefit.

### Tax

- `state minister of finance of a German federal state` → `public official responsible for tax revenue` — authority over buying evidence and public revenue remains.
- `up to 50,000 transactions` → `tens of thousands of transactions` — these are transaction counts rather than numbers of people at stake.
- `millions of euros` → `large sums` — the lost public revenue remains substantial without a currency cue.

### Tyran

- `six co-conspirators` → `a few fellow conspirators` — the number of collaborators does not count victims or beneficiaries.

### Vet

No further generalisation: veterinary duty, specialist expertise, directly infecting animals with highly contagious deadly disease, laboratory suffering, delayed medicines and probably more future animal lives saved all set the professional conflict, harms and benefit. No incidental identifying detail remains; no number of animals is specified.

## Details deliberately retained and review concerns

These details remain because they set the trade-off. Broader wording can still change
how a dilemma is judged; the concerns below are unresolved without human review and ratings.

- **Conjoined Twins**: Deliberately retained: two joined girls, one sustaining the other, and the shared-organ allocation; these determine the one-versus-two survival choice and victim-beneficiary overlap. Anatomical generalisation may change perceived medical realism.
- **Cannibalism on Lifeboat**: Deliberately retained: four crew, the remaining three, the youngest member’s minority and incapacity, and death without killing; these set the group sizes, vulnerability and survival choice. Replacing exact age and rank may affect perceived vulnerability or hierarchy.
- **Doodlebugs**: Deliberately retained: poorer residents, lower population density and vital military targets; they determine the distribution of deaths and military risk. Relative geometry remains necessary to explain how deception redirects strikes.
- **Plague Spread**: Deliberately retained: 800 villagers, children, contagion and increased local contact under isolation; these set the exposed population and the mechanism of increased local deaths versus reduced spread. A broader religious role may change perceived pastoral duty.
- **Euthanasia (in Ghetto)**: Deliberately retained: occupied ethnic ghetto, deportation and exclusion from hard labour; these explain why you feel sure the children will eventually be killed. The physician’s direct lethal act and the children’s probable suffering remain. Generalising the drug may change assumptions about sedation.
- **Immunization (Smallpox)**: Deliberately retained: the doctor-employee relationship, child subject, parental offer, sequential infections and uncertain immunity; these set authority, vulnerability, direct risk and possible benefit. Broader age wording may change perceived vulnerability.
- **911**: Deliberately retained: hijacking, the claimed bomb, evidence of prior crashes into buildings, passenger resistance and the uninhabited crash alternative; these determine the inferred threat, probable retaliation and deaths aboard versus on the ground. This causal structure remains recognisable. No moral inference depends on the terrorist label; hijacking and threatened killings remain.
- **Coma (Mother)**: Deliberately retained: mother-child relationship, no conscious perception, permanent loss of higher function, no hope of improvement and direct suffocation; these distinguish relief of relatives’ suffering from relief of the mother’s own conscious suffering.
- **Organ Law**: Deliberately retained: opt-out donation, cessation of brain activity, persistent cardiac arrest despite resuscitation and rare survival after that criterion; these define the risk of killing donors who might survive versus saving many more recipients.
- **Stillbirths**: Deliberately retained: concentration-camp imprisonment, nursing duty, guards’ orders, usually spared mothers after stillbirth and most likely survival after poisoning; these set the constrained choice and its uncertainty.
- **Septuplets**: No further generalisation: seven fetuses, three abortions and four remaining fetuses define the trade-off; severe disability, death, extremely high risk and parental authorisation define the harms, uncertainty and agency. No incidental identifying detail remains beyond those constraints.
- **Atom Bomb**: Deliberately retained: rumoured first development, the researcher’s contribution, assumed deterrence and the deadliest weapon of mass destruction in history; these set agency, uncertainty and severity. Removing the nuclear mechanism may change perceived plausibility of deterrence.
- **Mare Nostrum**: Deliberately retained: 150,000 rescued, 3,000 dead, unsafe crossings and the traffickers’ reliance on rescue; these quantify people affected and explain immediate rescue versus future deterrence. The probably-fewer future crossings and current versus future populations remain distinct.
- **Aid for Syria**: Deliberately retained: regime control of every receiving organisation, strong suspicion of diversion to war, unintentional support and millions denied aid on refusal; these define the competing harms and uncertainty.
- **Rescue Torture**: Deliberately retained: more than two days in custody, death from thirst feared soon, confessed kidnapping, illegal threats or physical violence and most likely disclosure; these establish urgency, the legal violation and uncertain life-saving benefit.
- **War Torture**: Deliberately retained: wartime urgency, most prisoners probably lacking information, systematic torture of all prisoners, binding prisoner protections and probably thousands saved; these determine severity, duty and uncertainty.
- **Rugby Cannibalism**: Deliberately retained: 15 of 40 dead initially, 19 survivors, no food, the abandoned search and eating already dead teammates; these set the populations, survival options and harm to the dead without killing for food. Broad altitude and temperature wording may change perceived survival realism.
- **Organ Market**: Deliberately retained: kidney failure, dialysis, the minor heart attack’s effect on eligibility, one kidney left to the donor, low medical risk and probably decades of life for both; these set the medical alternatives, survival benefit and donor harm.
- **Ashley Treatment**: Deliberately retained: removal of uterus and breast tissue, inability to sit or walk or consent, and medically unnecessary growth suppression; these specify irreversible bodily harm and the care constraint. Broader age, diagnosis and hormone wording may change perceived vulnerability or medical realism.
- **Test Tube Sibling**: Deliberately retained: embryo selection, umbilical-cord stem cells and above-90% recovery conditional on successful pregnancy and transplantation; these define creating a donor child, the procedure’s harm and its benefit. No injury to the donor child is invented.
- **Lion Hunt**: Deliberately retained: killing threatened animals and selling their furs, insufficient alternatives and sufficient proceeds to protect the remaining population; these define animal harm, overlap and the conservation benefit. Generalising the species may change judgments linked to attachment to lions.
- **Bishops**: Deliberately retained: co-religionists of the persecuted heritage, the silent agreement and the moral duty to speak; these determine group overlap and the risk imposed by fulfilling that duty. The broader religious office may change perceived responsibility.
- **Trail of Tears**: Deliberately retained: indigenous identity, traditional territory, constitutional rights, favourable court ruling and an unwinnable war if resistance continues; these set the violated rights, belonging and certain versus uncertain harms.
- **Hogesa (Hooligan Demo)**: Deliberately retained: opposition to religious extremism, repeated attacks on immigrants and police, and inalienable speech and assembly rights; these set the aim, threatened victims and rights conflict. Removing political affiliation may change perceptions of the group despite retaining its conduct.
- **Mass Rape**: Deliberately retained: eight men, at least one innocent, many others “probably all” guilty, unusable evidence and silence; these determine wrongful custody versus releasing rapists. The source’s unresolved guilt wording is retained. Broader age wording may change perceived vulnerability.
- **Endow**: Deliberately retained: the offender-named foundation, child kidnapping and murder, unclear remorse versus reputation repair, and the victim family’s objection; these set disrespect of the victim versus help to other victims. The task’s approval question remains. Broad donation wording may change perceived benefit size.
- **Ladder**: Deliberately retained: your position already on deck, the unresponsive blocking man, fear or cold, jostling and probable death from your direct push; these determine agency, absent self-rescue benefit and the alternative drownings.
- **Lifeboat**: Deliberately retained: 41 occupants, roughly ten adults to sacrifice, freezing death, the full second boat and your command role there; these fix group sizes, certainty of harm, lack of self-benefit and indirect agency. Most likely sinking within days and a good chance of rescue remain.
- **Medicine**: Deliberately retained: regular dosing, certain death without medicine, probably death within a few years, reduced care elsewhere and the independent estimate of likely more deaths; these define the one-versus-more resource choice. Broader cost, age and disease wording may change perceived urgency or benefit size.
- **Operations**: Deliberately retained: annual 50/48 reimbursement threshold, two patients, rural emergency access, possible non-reimbursement and deaths sooner or later; these define the institutional constraint, bodily harm and downstream deaths. The inherited “knee surgeries” slip amid lung surgery is corrected to one general type of operation throughout. Broad operation wording may change perceived surgical severity; no particular complication or risk was stated.
- **RAF**: Deliberately retained: one hostage, 11 demanded prisoners, five previously released and four subsequent attackers; the first two set the exchange and the latter two quantify the evidence for future killing. Several innocent victims and likely future attacks remain. No moral inference depends on the terrorist label. Broader hostage status may change judgments about public responsibility.
- **Ransom**: Deliberately retained: ten hostages, one prior death, probably more than ten future lives saved by refusal, weapons purchases and increased future kidnapping; these define the people at stake and causal trade-off. The source’s ten-hostage/one-prior-death ambiguity is retained. No moral inference depends on the terrorist label. Broader ransom size and duration may change perceived urgency or cost.
- **RobinHood**: No further generalisation: the bank role, temporary undetectable transfer, credit rating, wealthy depositors, indebted families and betrayed trust define the fraud and its beneficiaries; no incidental identifying detail remains. The paper codes opposite framing “No”, so no reversal is included despite the task’s different code.
- **Rwanda**: Deliberately retained: ethnic persecution, one demanded employee, the family and employees under your care, and the threat specifically to the targeted ethnicity inside; these determine responsibility, group overlap and risk. Protection probably causes an attack; handover means his death and the others’ safety. Your ethnicity remains unspecified. The aid setting remains because it explains the duty of care.
- **Suicide**: Deliberately retained: six-month eligibility, terminal illness, voluntary choice and a doctor’s fatal overdose; these define eligibility, consent and agency. Many others’ improved treatment remains the benefit. Broader medication and savings wording may change perceived medical realism or benefit size.
- **Tax**: Deliberately retained: illegal hacking, probable reliability, rewarding theft, possible future cybercrime and large tax receipts and fines; these set the evidence, uncertainty and cost-benefit choice. The original paper’s “Somewhat” reversal code governs despite the task’s “No”.
- **Tyran**: Deliberately retained: thousands already killed, political and ethnic persecution, the family power structure, your army role and the chance of a new beginning without civil war; these define severity, agency and an uncertain benefit. Only the ruler is to be killed.
- **Vet**: No further generalisation: veterinary duty, specialist expertise, directly infecting animals with highly contagious deadly disease, laboratory suffering, delayed medicines and probably more future animal lives saved all set the professional conflict, harms and benefit. No incidental identifying detail remains; no number of animals is specified.

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
