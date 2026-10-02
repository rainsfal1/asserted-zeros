# v2 pool pre-registration — the clean instrument, predicted before it exists

*Written and committed 2026-08-16, BEFORE `experiments/v2_pool.py` (or any v2 render/loop
harness) exists and before any file under `results/v2_pool/` is produced. The commit order
is the argument, exactly as in `docs/setup-ablation-preregistration.md`,
`docs/model-variation-preregistration.md` and `docs/decision-ablations-preregistration.md`.
Nothing below is edited after the harness lands; outcomes go in
`results/v2_pool/REPORT.md` beside these predictions, right or wrong. Where a later
measurement supersedes a statement here, the superseding note lives in the report, never in
this file.*

**Why v2 exists.** Audit waves 1–2 (`research/experiment-audit.md`,
`research/audit-wave-2.md`) found the v1 instrument defective at four layers, each now fixed
at the root (commits `e565476`, `d89dadd`, `151e3f7`, `d29356f`): the conformal score was a
top-20 truncation artifact (F-11); the channel's noise was a function of the seed alone
(2 clips for all of S3, 1 for M3) and entered after the codec; the re-ask loop's `rephrase`
toggled back to the original wording from k=3 while logging itself as a rephrase; and the
bank held two scripted wordings per cell, flooring Σᵢαᵢ at 2.75 (vacuous). **v1 stays frozen
as the labelled record. v2 is the instrument the paper's claims will stand on**, and this
document commits its design and its predictions before a single record is rendered.

---

## 0 · Disclosure — everything already seen before this was written

A pre-registration that hides prior looks is worse than none. Everything below has been
examined, and several predictions are directly informed by it — each prediction labels its
scouting status.

**The Powergrading corpus and M6's committed results, in full.** 698 students × 10 gold
items, three graders, majority gold; gold pass rate **94.7%** (661/698) under USCIS ≥ 6/10;
only **17.9%** of students within ±1 of the boundary. M6's sweep (releases, errors,
coverage) at seven Σα points; the corrected floor **0.042735 = 10/234** with the release
cliff exactly on it; `floor_by_distinct_cal` 0.1654; the κ table (Q3 = 0.574, Q20 = 0.449;
`LOW_KAPPA = {3, 20}`); the 69.4% train↔test lexical-overlap measurement; the 31.6% mean
pairwise seed-overlap measurement.

**The wording-diversity table** over normalized responses per (item, gold): correct
10/75/437/90/69/129/324/34/208/202 and incorrect 37/37/126/93/26/54/96/41/125/19 for items
1–8, 13, 20 respectively (measured 2026-08-16 over the whole corpus, before any split).

**The v1 defect measurements**: the F-11 concentration table (support-1 rates
52.3%/80.1%/51.4%/79.5%; zero-score shares 90.6%/89.7%; threshold 0.0 at α ≥ 0.1); the
channel's 2-of-930 / 1-of-930 noise-file collapse and the 0.55 s tiled clip; the rephrase
toggle trace; B1's builder A/B (0/30 vs 30/30 feasible cells at 0% risk) and α-spending
price (101 vs 5 wrong per 5,000 at α = 0.02); M1.5's 175/1600; the binding-decomposition
cells; every number in both audit documents.

**Not seen at any point:** any v2-rendered audio or transcript; any per-option score on any
real model output (the per-option path exists and is unit-tested on mocks only); any
end-to-end-calibrated threshold anywhere; any loop run over Powergrading data; any
repeat-vs-rephrase measurement on real wording diversity.

## 1 · Established structurally (not predictions)

- **`verdict_margin` ≥ 1**, the USCIS rule is a counting rule, and its O(k) reachability
  shortcut is exact (774,198-case verification, audit wave 2).
- **The per-option path cannot truncate or fabricate** — by construction and unit test
  (`per_option_scores` raises on zero evidence and on post-mask semantics).
- **`degrade_v2` gives every record its own noise realization** — by construction and test.
- **`rephrase` under `reask_sessions_v2` cannot re-draw an elicited wording** — by
  construction and test; `pool_exhausted` is reachable.
- **Incorrect-gold rephrase pools are thin**: some (item, gold=incorrect) cells hold few
  distinct wordings within a one-third student split (item 20: 19 corpus-wide). Exhaustion
  WILL fire there; that is the instrument being honest, not a failure.

## 2 · The design, pinned

**Pool.** The **first 350 students** by FNV-1a hash of student id (deterministic, seed-free)
× the 10 gold items × **4 channel draws** per response: levels `(tel, snr0, snr_m5,
snr_m10)`, one draw each. **One voice per student**, assigned round-robin over the 8
edge-tts accent voices by student-hash order. TTS of the student's typed answer verbatim
(the typed→spoken modality transfer is a stated limitation, §4). Channel = `degrade_v2`
(per-record noise, noise-before-codec, active-speech SNR). ASR = faster-whisper large-v3,
CPU int8, per-worker singleton, resume-by-id caching — the m3 harness pattern.

**Record ids** keep the 5-field arity `voice_of` requires:
`q{item}:{student}:{VOICE}:{level}:d{draw}` with draw ∈ {1..4} fixed per level.

**Extraction.** Judge and evaluator (the seated v1 pair, unchanged models), blind as
always, via **`canonical_per_option`** over single-token options `("a", "b")` mapped to
{correct, incorrect}, with the item's question and the USCIS key in the prompt's meaning
slots. Per-option raw logprobs persisted (`raw_logprob`, `mass_unnormalized`) — the
diagnostic v1 threw away.

**Split.** `split_students` (FNV-1a, deterministic), same three-way shape as M6 for
comparability: **train third held in reserve** (no component is fitted in v2; it exists for
any future fitted-comparison arm), **cal third** for conformal calibration, **test third**
for evaluation. n ≈ 116/117/117. Split-conformal floor at k = 10: **10/118 ≈ 0.0847** —
clears Σα = 0.10, with the thinner margin than M6's 0.0427 stated rather than hidden.
**Re-ask draws never cross splits**: a test session's rephrase pool is test-split responses
only; the calibration protocol's re-asks draw cal-split responses only. (A cross-split draw
would let calibration responses enter test evidence — leakage by the back door.)

**Sessions.** One session per test student: the 10 items' first elicitations are the
student's own 4-draw records at a per-session level mix fixed by hash (so levels are
balanced across sessions, not chosen by outcome). `repeat` = a fresh channel draw of the
same response (up to 3 remain); `rephrase` = another same-split student's response with the
same gold for that item, excluding wordings already elicited (population rephrase — a
different object from a self-rephrase, stated as such); `ladder` = repeat then rephrase.

**The two defenses, head-to-head, same pool, same grid:**
- **D1 — geometric α-spending** (`spend=True`, α/2ᵏ), the v1 defense.
- **D2 — end-to-end protocol calibration**: the identical re-ask procedure (same policy,
  same budget, same stopping rule) is run on every calibration unit, and the conformal
  score calibrated is the **finally-selected elicitation's** score. One unsplit α; no union
  bound over looks; no 2ᵏ price (Quach et al., ICLR 2024; the audit's T4 Tier-1
  recommendation, deferred since pass 2 and built here).

**Attack arms** (reproductions on the clean instrument): the naive stop-at-first-singleton
loop (`spend=False` + re-ask — attack 1); the quarantined legacy builder computed on v2
scores (attack 2) — **this requires adding the v2 harness to the quarantine's frozen
importer allowlist**, which the ratchet permits in the same diff with justification; that
edit ships with the harness commit and is disclosed here in advance; the unspent-budget
default (attack 3) is attack 1's spend axis.

**Grid.** Σα ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50} (M6's grid), 5 session-draw
seeds, budget K = 3, all three policies plus `none`.

## 3 · Predictions

Each names its quantity and its falsifier; each will be scored by a function computing that
quantity and nothing else.

**P1 — Attack 1 reproduces: the optional-stopping exploit is a property of the mechanism,
not of v1's broken instrument.** *(Scouting: informed by M1.5's 175/1600 and B1's re-ask
phase — on v1 data only.)*
Under the naive loop (`spend=False`, policy `rephrase`, K = 3, stop at first certification)
the wrong-release rate among released **strictly exceeds** the no-loop (`none`) rate on the
same sessions, at every Σα ∈ {0.05, 0.10}, in ≥ 4 of 5 seeds.
*Falsified if* the excess is absent or reversed at either Σα in 2+ seeds.

**P2 — Attack 3 reproduces: spending pays.** *(Scouting: informed by B1's 101 → 5.)*
At the same operating points, `spend=True` releases **strictly fewer** wrong verdicts than
`spend=False` under the looped policies, at every Σα ∈ {0.02, 0.05, 0.10}, pooled over
seeds; and never more at any swept Σα.
*Falsified if* spending releases more wrong verdicts at any swept point.

**P3 — Attack 2 reproduces: ignorance semantics decide feasibility.** *(Scouting: informed
by B1's 0/30 vs 30/30 — v1 scores only.)*
The legacy builder (empty→argmin) has **no feasible operating point at 0% risk tolerance**
in any seed on this pool, while `conformal_set` (empty→FULL) has ≥ 1 in every seed.
*Falsified if* the legacy builder attains any 0%-risk point, or the normative builder
attains none.

**P4 — The per-option score is a real dial.** *(Scouting: informed by the F-11
concentration table — v1's failure mode, not v2 data.)*
The share of calibration scores **exactly 0.0** is **below 0.20** (v1: ~0.90), and the
calibrated threshold at Σα ∈ {0.05, 0.10} is **strictly positive** in every split seed.
*Falsified if* the zero-spike reaches 0.20, or any calibrated threshold is exactly 0.0.
*Reported regardless:* the full score histogram; `mass_unnormalized` distribution (the
off-option residual v1 could not express); threshold reproducibility across 20
student-disjoint split seeds — the training-conditional dispersion that was v1's standing
negative, re-measured on a real dial.

**P5 — D2 (end-to-end calibration) dominates D1 (spending) on release at matched risk.**
*(Scouting: structural only — the 2ᵏ price argument; no end-to-end threshold has ever been
computed.)*
At matched realized release (banded, [target, target+0.05] — the project's signature-error
rule) D2's wrong-per-1000 is **no higher** than D1's at every attainable target, AND D2
releases **strictly more** than D1 at equal Σα ∈ {0.05, 0.10} in ≥ 4 of 5 seeds.
*Falsified if* D2 shows higher risk at any matched target, or its release advantage is
absent in 2+ seeds.

**P6 — Repeat and rephrase differ beyond noise, measurably, without the toggle confound.**
*(Scouting: v1's finding ⑤ — pool-dependent sign, measured on the broken draw. No
directional claim is made here, deliberately: v1's evidence is that the direction is
pool-dependent, so predicting a direction would be theater.)*
At Σα = 0.05 with spending, |wrong-release(repeat) − wrong-release(rephrase)| exceeds the
5-seed noise band on this pool, and `pool_exhausted` fires under `rephrase` for
(item, gold=incorrect) cells with < K available wordings.
*Falsified if* the policies are within noise, or exhaustion fails to fire where the pool
arithmetic says it must.

## 4 · Reported regardless, and the standing limitations

- **Both defenses' full frontiers** with realized release printed per cell and the band
  shown; per-seed dispersion everywhere; n_eff (distinct responses carrying errors) beside
  every pooled denominator.
- **The always-answer baseline and the trivial "always PASS" baseline** (94.7% gold pass
  rate makes it release 100% at 5.3% error — the comparator M6 lacked, audit wave 2
  finding 14).
- **Per-item coverage**, not only the pooled average.
- The `mode_effective` audit: every round's label vs the policy that requested it.
- **Limitations, stated now**: TTS speech (real language, synthetic voice — the L2-ARCTIC
  anchor remains the only real-speech evidence); typed→spoken modality transfer;
  population-rephrase ≠ self-rephrase (the 25-recording study is the path to the real
  object); one ASR family (T1's arm still unexercised); the corpus's 94.7% pass-rate skew
  (boundary sessions are scarce: ~18% of students); grader-majority gold (coverage w.r.t.
  P_vote, Stutz et al.).

## 5 · Amendment protocol

Amendments before the harness exists: additions-only, dated, above this line — the B1/B2
convention. After the harness lands, this file is frozen; supersessions live in
`results/v2_pool/REPORT.md`.

---

## 6 · Amendment, 2026-08-16 — audit wave 3, before the harness (additions only)

*§§0–5 are byte-identical; `git diff c7f8455 -- docs/v2-pool-preregistration.md` shows
insertions only. Still BEFORE any v2 harness file exists. Wave 3 (two independent
adversarial reviewers plus a byte-identity executor — `research/audit-wave-3.md`) audited
this document against the code and found the §2/§3 design defective in ways that would have
silently decided several predictions by arithmetic. Every correction is below, each naming
its finding. The wave also verified the backward-looking claims clean: all six frozen
harnesses reproduce byte-identically at the commit this amendment lands on.*

### 6.1 · The floors, corrected — spending doubles the floor (wave-3 finding 1)

§2 disclosed only the plain floor 10/118 ≈ 0.0847. Under **D1 the first elicitation runs at
αᵢ/2** (`alpha_for_elicitations`, budget reserved up front), so D1's effective floor is
**20/118 ≈ 0.1695**. Exact, verified against `calibrate_threshold`:

| Σα (grid) | plain / D2 threshold finite? | D1 (spend) threshold finite? |
|---|---|---|
| 0.02, 0.05 | no — `inf`, releases nothing | no |
| 0.10, 0.15 | **yes** | no |
| 0.20 – 0.50 | yes | **yes** |

**Predictions re-pointed to live operating points** (clauses otherwise unchanged):
- **P1** (naive loop vs none, both unspent): scored at **Σα ∈ {0.10, 0.15}** (was 0.05/0.10).
- **P2** (spend vs unspent): scored at **Σα ∈ {0.20, 0.30, 0.50}** — the points where both
  arms can release; "and never more at any swept Σα" now reads "at any swept Σα where the
  unspent arm releases".
- **P4** (thresholds strictly positive): scored at **Σα ∈ {0.10, 0.20}**.
- **P5** (D1 vs D2): equal-Σα comparison at **Σα ∈ {0.20, 0.30}**; the matched-release
  comparison additionally reports every target D1 cannot attain as **absent, never a tie**.
The degenerate points stay on the grid and are reported (the release cliffs are themselves
evidence, as M6's was), but no prediction is scored on them.

### 6.2 · D2 defined operationally — the circularity removed (wave-3 finding 9)

§2's "calibrate the finally-selected elicitation's score" left which threshold governs the
calibration-side re-asks unstated — circular as written. **D2 is hereby pinned as
Learn-then-Test over one session-level threshold:** for each candidate t on the committed
14-point grid, run the identical loop (policy, budget K = 3, stopping rule) with threshold t
on every **cal** session; measure the realized P(release ∧ wrong) against the target Σα with
a binomial (Hoeffding) p-value; Bonferroni over the 14 candidates; select the largest t
whose p-value clears α_LTT = 0.05. Evaluate on **test** only. One selected threshold, no
union bound over looks, no 2ᵏ price; validity is LTT's (Angelopoulos et al., *AoAS* 2025),
with Quach et al. (ICLR 2024) as the construction this instantiates. D2's floor is the
plain 10/118.

### 6.3 · The session engine and the keying — the design was unimplementable as stated (wave-3 finding 2)

- `certify`/`reask_ranking` are hardwired to the v1 11-item scorer; `reask_sessions_v2`
  therefore **cannot run a v2 session today**. The harness commit will add a
  decision-function-parameterized engine (the USCIS reachability function, already
  brute-force-verified in the frozen M6 harness over 774,198 cases, ported into `src/` with
  its own tests — the frozen copy in `m6_powergrading.py` is untouched).
- **Keying, decided:** rephrase-exclusion keys on the **normalized wording**
  (`m6.normalize`) — that is what "already elicited" must mean; `repeat` is implemented as
  *another channel draw of the same (student, item) response*, NOT via the v1
  `_variant_index` machinery, which was built for the uniform-multiplicity bank pools and
  whose guard would (correctly) reject this pool's 10-vs-437 wording skew. §1's claim that
  the exclusion property was "established by construction and test" is **downgraded**: it
  was established on uniform synthetic banks; the v2 engine re-proves it on this pool's
  keying in its own tests before any run.
- **The cross-split rule gains an enforcement mechanism** (wave-3 finding 4): the engine
  takes the allowed draw pool explicitly and asserts every drawn record's student is inside
  the session's split — a loud failure, not caller discipline.

### 6.4 · The binary prompt (wave-3 finding 3, found independently by both reviewers)

`CANONICAL_PROMPT` renders a dangling empty "(c)" slot and instructs "reply with one of:
a, b, c" even when only {a, b} are live — leaking pre-mask mass to a phantom option and
corrupting `mass_unnormalized`, the diagnostic this pool exists to measure. The harness
uses a **binary prompt variant** (two slots, instruction naming exactly the live options),
added beside the v1 template (which stays byte-frozen). The prompt text will be committed
with the harness, before extraction runs.

### 6.5 · P6's confounds, disclosed (wave-3 finding 10)

Under this design `repeat` changes the **channel level** (the four draws are one per
level) and `rephrase` changes the **voice** with probability 7/8 (round-robin per student).
The repeat-vs-rephrase contrast is therefore confounded with level and voice shifts.
Mitigation, pre-committed: P6 is scored on the pooled contrast as registered, AND the
report carries the per-level-transition and same-voice-subset breakdowns so the confound is
measurable rather than assumed away. P6's exhaustion clause is corrected: exhaustion is
forced only when the loop directs ≥ (available wordings) rephrase attempts at one item —
the scoring function counts per-item attempts, not the global budget.

### 6.6 · Remaining pins (wave-3 finding 9)

- **Calibration row unit:** one record per (student, item) — the record at the
  hash-assigned per-(student, item) level, the same level-mix rule the test sessions use.
  n = 117 rows per item; the §2 floor figures attach to exactly this.
- **Split seed:** split seed **0** is primary for every scored clause; the 20-seed
  dispersion is reported alongside (unchanged).
- **Distinct-wording floor** for the cal third is reported beside the row floor (M6's
  analogue was 0.1654 at n=233; v2's will be worse and is reported, not hidden).
- **The quarantine-importer contradiction** (wave-3 finding 6) is resolved in favour of
  this document: the attack-2 arm imports the quarantine via the frozen allowlist, grown in
  the harness diff with justification; the ratchet's and F-3 row's "v2 never imports it"
  prose will be corrected to name this one disclosed exception in the same commit.
- **`verdict_margin` does not port** to USCIS totals (wave-3 finding 7): the decision
  ablations' B-ii arm on the v2 pool uses a margin function for the ≥6/10 rule
  (|correct-count − 6| based), committed with that harness; the v1 arms keep VM-21's.

---

## 7 · Amendment, 2026-08-17 — audit wave 4 + the live serving finding (additions only)

*§§0–6 byte-identical; still before ANY extraction has run (the live serving session
produced only the 12-record diagnostic below — no extraction file exists). Three
corrections, each caught before it could bias a scored result.*

### 7.1 · P6's operating point — the §6.1 defect class, for the prediction §6.1 missed

P6 was registered at Σα = 0.05 *with spending* — below D1's own floor (§6.1: spend-k1 is
`inf` until Σα = 0.20), so both policies release nothing and the clause is auto-falsified by
arithmetic. **P6 is re-pointed to Σα = 0.20** (the smallest live spend point); its clauses
are otherwise unchanged.

### 7.2 · The extraction transport — the live finding, and the pivot

The pre-registered transport ("one forced guided call per option") does not exist on this
serving stack: `guided_choice=[single option]` is **inert** — forcing "b" *emits* "a"
(verified twice against Qwen3-32B-AWQ; `research/v2-extraction-transport-note.md`). The
per-option guard caught this at call 1. The same diagnostic showed the measured-transport
alternative: under `CANONICAL_PROMPT_BINARY`, variant-aggregated top-k mass gives
P(a)+P(b) = 1.0000 — every probability measured, none asserted.

**Amended transport:** one call per record, `top_logprobs` at the server maximum (the
server is restarted with `--max-logprobs 100`), per-option mass = variant-aggregated top-k
(`token.strip("(). ").lower()`), with two hard guards: (i) an option with **no variant
present raises** — absence is never recorded as zero (the F-11 fix preserved as a guard
rather than an assumption); (ii) `mass_unnormalized` persisted per record. The probe
(3-option domain) uses the same transport and the same guards. `PerOptionExtraction`'s
schema is unchanged.

### 7.3 · Repeat's draw order, pinned

The engine drew repeat re-elicitations in ascending draw order — d1 (`tel`) first — a
*directional* bias toward the clean channel inside the repeat arm (§6.5 disclosed level
shifts generically but did not pin the order). **Pinned: hash-shuffled order per
(student, item)** — deterministic, seed-free, level-unbiased. Engine + tests updated in the
same commit as this amendment.

### 7.4 · Small measured disclosures

The two unspeakable responses (§ manifest) land one in the **cal** third (its item's
calibration stream is 116, floor 1/117 for that item) and one in the **test** third (that
session is dropped: n_test = 116). All floors are measured from actuals by the harness, so
these shift reported figures in the third decimal at most; disclosed here so §6.1's
"n = 117" carries its caveat.

---

## 8 · Amendment, 2026-08-17 — audit wave 5 (additions only)

*§§0–7 byte-identical. Still before ANY extraction exists: the transport §7.2 pins has not
been implemented yet, so no score, threshold, or verdict in this document has ever been
computed on real extractions. Everything below is therefore a pre-data correction. Where a
number appears it is either **arithmetic** (independent of the data) or a **synthetic dry
run** — a fake extraction cache over the real 13,992-record pool, used to exercise code
paths — and each is labelled as such. No dry-run number is evidence about the instrument.*

### 8.1 · D2's real feasibility floor — §6.1's table and §6.2's floor claim corrected

§6.1's table says D2's threshold is finite at Σα ∈ {0.10, 0.15}, and §6.2 says "D2's floor
is the plain 10/118". **Both are wrong, by the arithmetic of the test §6.2 itself
registers.** The Hoeffding p-value at realized cal risk `r` is `exp(−2n(σ−r)²)`, and
Bonferroni requires it to clear `0.05/14 = 0.003571`. With the cal split's `n = 116`
sessions (117 cal students less the §7.4 drop), even a **perfect** cal run (`r = 0`) needs

  σ ≥ √( ln(280) / (2·116) ) = √(5.63479/232) = **0.1558**

so **no candidate can be selected at any Σα ≤ 0.15**, and Σα = 0.20 additionally requires
`r ≤ 0.044`. D2's feasibility floor is therefore **Σα ≈ 0.156**, not the plain 10/118
floor: LTT's price is paid in the *evidence needed to certify a threshold*, not in the
threshold's own construction. A synthetic dry run agrees (0 passers at every σ ≤ 0.15).
P5's scored points {0.20, 0.30} are unaffected. Absent D2 cells are reported as absent —
never silently skipped, never a tie.

### 8.2 · D2's selection direction — "largest passer" is degenerate, and is replaced

§6.2 registered "select the largest t whose p-value clears α_LTT". At the top of the grid
every option's score falls under the threshold, so every item set becomes FULL, nothing
certifies, and release *and* risk both collapse to zero — which means **the largest
candidate always passes, vacuously, by releasing nothing at all**. The rule as written
makes D2 a near-no-op and would have decided P5 by construction rather than by evidence.

*(Reasoning corrected the same day, before any data: this amendment first justified the
change by "risk is monotone non-increasing in t". Checking that claim against the dry run
refuted the monotonicity half — release is **unimodal** in t, not monotone: 0.000 at the
bottom of the grid, where no option clears the threshold and the empty set becomes FULL by
the §2 rule; a peak of 1.000 at t ∈ [0.3, 0.5], where exactly one option survives; and
0.000 again at t ≥ 0.9. The correction strengthens the case — both ENDS of the grid are
degenerate, and "largest" lands on one of them — but the original sentence was wrong and is
not left standing.)*

**Amended:** among the candidates that clear the same Bonferroni-corrected test, D2 selects
the **most-releasing** one (ties to the smaller t), and the full pass set is reported with
each candidate's p-value and realized cal release. Validity is unchanged and is LTT's own:
the correction controls the whole family of 14 candidates, so *any* selection among the
passers is valid — utility-directed selection is what LTT exists to license. Synthetic dry
run, for the size of what this fixes: at every live σ the registered rule picks t = 0.95 at
**0.000** cal release, the amended rule t = 0.3 at **1.000**.

### 8.3 · D2 is calibrated under the policy it is evaluated under

§2 requires "the **identical** re-ask procedure (same policy, same budget, same stopping
rule)" on every calibration unit. The harness selected one threshold under `rephrase` and
evaluated it under all four policies, so for three of them the calibration protocol was not
the test protocol and LTT's guarantee did not attach. **Pinned: D2's threshold is selected
per policy**, and a D2 cell exists only for a policy whose own selection succeeded.
(Synthetic dry run: `none` selects 0.5 where the looped policies select 0.3 — the mismatch
was not hypothetical.)

### 8.4 · Scorer-vs-clause corrections (no clause changes)

Three scorers did not compute their registered clauses; all are now brought into
compliance, which changes code, not commitments:

- **P2** was scored on `rephrase` alone and omitted the "never more at any swept Σα where
  the unspent arm releases" conjunct. Both looped-policy coverage and the sweep are restored.
- **P5**'s matched-release conjunct — the *first half of the registered conjunction* — was
  never computed at all; only the equal-Σα seed comparison was. The banded matched-release
  frontier is implemented, with §6.1's "absent, never a tie" pin enforced and the number of
  attained targets reported beside the verdict.
- **P6**'s exhaustion clause counted `pool_exhausted` over **every** rule and policy and
  asked only "> 0" — the global-budget reading §6.5 explicitly corrected. It is now scored
  under `rephrase` only, per item, against that item's own wording supply, and a *spurious*
  exhaustion (the loop quitting while wordings remained) counts as a violation.
  §6.5's per-level-transition and same-voice breakdowns are now actually rendered.

### 8.5 · The extraction transport carries an identity guard

§7.2's guards are joined by a third, forced by the same live finding: **the emitted token
must be the option that was forced.** Without it, an inert constraint makes every pass
return the same token's logprob, and the assembled distribution is a fabrication that looks
entirely plausible — a near-uniform split with a believable residual — whenever that token's
probability is below ~0.5, i.e. silently, in exactly the borderline band that decides
calibration. It raises now, with the diagnosis in the message.

### 8.6 · Two engine terminal behaviors, disclosed rather than changed

Measured, and left as they are because changing the loop's shape after the pool is rendered
would be a post-hoc design change: (i) `ladder` does not fall back — once past its single
repeat rung, an item with no fresh wordings ends the session even if unused repeat draws
remain; (ii) a session ends when the **top-ranked** item's pool is empty, without trying
other positive-gain items (the rule inherited verbatim from the v1 loop). Both inflate
`pool_exhausted` relative to a per-item reading, which is why §8.4's exhaustion audit keys
on the exhausted item — sessions now record `stop_item`, so the reading is exact rather
than inferred.

### 8.7 · §4's "reported regardless" is now actually reported

P4's threshold reproducibility across 20 student-disjoint split seeds, the score histogram,
the `mass_unnormalized` distribution, the full frontier with realized release and per-seed
dispersion, `n` beside every pooled rate, and the two no-abstention baselines (always-answer
at the point estimate; always-PASS) were pre-committed in §4 and rendered nowhere. They are
computed and rendered now. A report that showed only six HELD/FALSIFIED verdicts would let
the strongest cell speak for the instrument — which is the failure mode §4 exists to prevent.

