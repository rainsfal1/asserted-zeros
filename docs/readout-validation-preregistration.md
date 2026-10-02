# Readout validation pre-registration — checking the fix against something other than itself

*Written and committed 2026-08-18, BEFORE any harness for these measurements exists and
before any file under `results/readout_validation/` or `results/s3m3_measured/` is produced.
Frozen on harness landing; supersessions live in the REPORTs, never here. Amendments are
additions-only and dated.*

**Purpose.** The workshop paper claims a remedy: measure every option's probability or
refuse. The standing reviewer objection is that the remedy is **checked only against
itself** — `canonical_measured` reads a top-100 window under `guided_choice`, aggregates
token variants with `_norm_token`, renormalizes over options, and nothing outside that code
path has ever confirmed the result. This document registers three measurements that check
it from outside, plus one that answers an attribution question the paper currently gets
wrong.

## 0 · Disclosure — what has already been seen

Full disclosure matters more here than usual, because these are post-hoc experiments on a
paper already written.

**Seen:** every committed cache (`results/s3`, `results/m3`, `results/v2_pool`,
`results/per_option_probe`); all eight audit waves; the probe's REPORT (Q1/Q3/Q4 held, Q2
falsified); the v2 REPORT (P4 held, P1/P2/P3/P5/P6 falsified); the paper in its current
state, including the claim that 0 of 13,992 records carry an exactly-zero score and that
per-extractor minimum probabilities are 7.8e-9 and 3.9e-9. It is also known that
`raw_logprob` carries no information beyond `dist` and `mass_unnormalized` (verified:
`max |log(dist·mass) − raw_logprob| = 0.000e+00` over all 13,992 rows).

**Not seen:** any full-vocabulary distribution from either model, ever; any rank-level
reconstruction of top-*k* truncation; any comparison between the old normalizer
(`alt.token.strip().lower()`) and `_norm_token` on a shared token window; any offline
forward pass of either model in this project.

**Known bias to guard against.** Three consecutive revisions of this paper contained
claims the author asserted beyond the evidence, each caught by an independent pass. The
criteria below are therefore written to be decidable by arithmetic, not by judgement, and
the falsification conditions are stated before the harness exists.

## 1 · The attribution defect this document also settles

`experiments/per_option_probe.py:6` states "the scoring transport is the only changed
variable." **That is false.** The old readout normalized with `alt.token.strip().lower()`
(`extraction.py:138`); the corrected one uses `_norm_token`, which additionally strips
`"(). \t\n"` (`extraction.py:424`). The old path cannot see `(a)`, `a.`, or `A)`. Some
share of the headline 90.6% may therefore be a **normalization** artifact rather than a
**truncation** one, and the paper attributes it wholly to truncation.

This is registered as a measurement to be reported regardless of outcome.

## 2 · Design, pinned

### 2.1 F4 — independent full-vocabulary readout validation

- **Transport.** An offline forward pass, no HTTP server, no `guided_choice`. Preferred
  implementation is `transformers`; if the AWQ checkpoints cannot be loaded without adding
  a dependency that risks the imperatively-installed GPU stack (this environment pins that stack
  imperatively, and `autoawq` is not installed), the fallback is vLLM's offline `LLM`. **The
  independence claim is transport-dependent and must be stated accordingly:**
  - `transformers`: independent of vLLM entirely.
  - vLLM offline: independent of the HTTP server, the OpenAI-compatible layer, guided
    decoding, and the top-*k* window — **but shares vLLM's model execution**. It validates
    the *readout*, not the engine, and the paper must say exactly that.
- **Models.** Both extractors: `Qwen/Qwen3-32B-AWQ` and
  `gaunernst/gemma-3-27b-it-qat-autoawq`. Sequential, one resident at a time.
- **Positive control, mandatory and blocking.** Before any distribution is compared, assert
  the offline prompt's token IDs are identical to those the served path produces for the
  same record (same chat template, `enable_thinking=False` for Qwen3). A distribution
  difference is uninformative unless the input is proven identical. If the control fails,
  the comparison does not run and the failure is reported.
- **Sample.** 300 records per extractor, stratified on the corrected readout's own
  **minimum option probability**, in decile bands of `log10(min_p)`, oversampling the
  bottom band (which is where the paper's claim lives and where a uniform sample would
  contain nothing). Deterministic FNV-1a hash order of record id, seed-free. Drawn from
  `results/v2_pool` (binary domain) and, after §2.2 runs, `results/s3m3_measured`
  (3-option domain) so both arities are covered.

**The statistic.** For each sampled record, from the full vocabulary:

- `p_full(o)` = Σ over **all** vocabulary tokens `t` with `_norm_token(t) == o` of `P(t)`
- `M_full` = Σ_o `p_full(o)` — the true option mass, compared against `mass_unnormalized`
- `dist_full(o)` = `p_full(o) / M_full` — compared against the recorded `dist`
- `s_full` = `1 − dist_full(gold)` — compared against the recorded nonconformity score

**Two-level decomposition, registered so a disagreement is interpretable.** If the offline
and served values disagree, the cause is ambiguous between (a) the top-100 window missing
option mass and (b) the pre-mask assumption being wrong. To separate them, also compute
`M_window` — the option mass the offline full distribution would have yielded had it been
truncated to its own top 100. Then:

- `M_window ≈ mass_unnormalized` but `M_full > M_window` ⇒ the readout logic is correct and
  the window genuinely misses mass. **This is the case that falsifies the remedy.**
- `M_window ≠ mass_unnormalized` ⇒ the discrepancy lives in the server / guided-decoding
  layer, i.e. the pre-mask assumption, which is a different finding.

### 2.2 F3 + F2 — one re-extraction of s3 + m3

- **Records.** All 2,618 (`s3` 770 + `m3` 1,848), both extractors, 5,236 calls. Same
  models that produced the committed v1 extractions, confirmed from
  `results/{s3,m3}/serve-*.log`. Same prompt (`CANONICAL_PROMPT`), same 3-option domain, so
  records, options, models and conditions are all held fixed and the readout is the only
  changed variable — this time actually.
- **Code change.** `PerOptionExtraction` gains a field carrying the full top-*k* token list;
  `variant_aggregated_scores` returns it instead of discarding it. Additive; the 250
  committed tests must stay green.
- **Persisted per record:** the full top-100 `(token, logprob)` list, the per-option
  aggregated mass under **both** normalizers, `mass_unnormalized`, `dist`, and the score.

### 2.3 Sample-size honesty

The probe already established the *direction* on 200 stratified records per pool. The
marginal value of 2,618 unstratified records is an honest **marginal distribution** for
Figure 1 — which a stratified sample cannot produce — plus the token windows F2 and the
attribution analysis need. It is not expected to reverse anything, and this document says
so in advance so that "it confirmed the probe" cannot later be presented as a new result.

## 3 · Predictions

**V1 — The variant set is complete.** *(Tokenizer-only, no GPU.)* Enumerating every
vocabulary token of each model, `_norm_token` maps into `{a,b,c}` (resp. `{a,b}`) exactly
the tokens a human would call a spelling of that option, and misses none that carries
non-negligible mass.
*Falsified if* any vocabulary token that a reader would call an option spelling is missed by
`_norm_token`, **and** that token carries ≥1e-6 mass on any sampled record.
*Reported regardless:* the full list of matched tokens per model, and every near-miss.

**V2 — The window does not miss option mass.** `M_full / mass_unnormalized` has median
within `[0.99, 1.01]`, and the 99th percentile of `|1 − M_full/mass_unnormalized|` is below
`0.05`.
*Falsified if* the median ratio exceeds `1.05` — i.e. the top-100 window systematically
misses ≥5% of the true option mass, which would mean **the corrected readout has its own
smaller version of the defect this paper is about.**
*Reported regardless:* the full ratio distribution, and the `M_window` decomposition of §2.1.

**V3 — Scores agree.** On ≥95% of sampled records the true-option nonconformity scores from
the two paths agree within a factor of 2 (ratio ∈ `[0.5, 2]`), compared on the log scale
because the scores span 1e-9 to 1 and absolute tolerances are meaningless at the low end.
*Falsified if* fewer than 95% fall in that band.
*Reported regardless:* median ratio and the 5th/95th percentiles. The bar is deliberately
loose — two deterministic forward passes of the same weights should agree far more tightly —
so the reported distribution, not the pass/fail, is the informative output.

**V4 — No zeros.** The independent readout finds **0** exactly-zero true-option scores on
the sample.
*Falsified if* any record scores exactly 0.0.

**V5 — Normalization is a small share of the artifact.** Re-aggregating the persisted
windows under the old normalizer versus `_norm_token`, the share of the committed
asserted-zero rows explained by normalization width rather than by truncation is **below
5%**.
*Falsified if* it is 5% or above, in which case §2 and §4 must re-attribute the headline
number and the probe's "only changed variable" claim must be corrected in the paper, not
just in the code.
*Reported regardless:* the exact split, per corpus and per extractor.

**V6 — Rank-level truncation reproduces the mechanism.** Simulating "the option fell out of
a top-20 window **by rank**" on the persisted token lists reproduces the qualitative
structure §5 currently obtains from a probability-floor simulation: thresholds driven to
endpoints where the window bites, and untouched items where it does not.
*Falsified if* the rank-level reconstruction shows a materially different mechanism — in
which case §5's mechanism paragraph is rewritten to match the rank-level result, which is
the real one, rather than the floor being defended.

## 4 · What each outcome does to the paper

| Outcome | Consequence |
|---|---|
| V2, V3, V4 hold | The remedy is corroborated by a path sharing no readout code. The paper deletes the hedge, and claims **nothing more** — corroboration does not license a stronger remedy claim. |
| V2 falsified | The corrected readout also misses mass. Report the magnitude, narrow §3's claim to what survives, and treat it as the third time this paper publishes a refutation of its own claim. **Author reports the number to the user before any prose changes.** |
| V3 falsified but V2 holds | The disagreement is in aggregation or renormalization, not coverage. Investigate before writing. |
| Positive control fails | No comparison is reported at all. A prompt mismatch makes the whole measurement uninformative, and reporting it anyway would be exactly the error this paper documents. |
| V5 falsified | §2/§4 re-attribute the 90.6% between truncation and normalization; the probe's control claim is corrected in the paper. |
| V6 falsified | §5's mechanism is rewritten to the rank-level result. If it opens a question this paper cannot close, it moves to SaTML rather than being half-answered. |

## 5 · Run order and the serving constraint

One model resident at a time on the 46 GB L20. `VLLM_USE_FLASHINFER_SAMPLER=0` and
`CUDA_HOME=<venv>/lib/python3.11/site-packages/nvidia/cu13` on every launch. The zombie
vLLM on port 8000 (user `sines`) is not touched.

1. V1 — tokenizer-only, no GPU, runs first.
2. F4 evaluator: stop the served evaluator, offline load, measure. **Report to the user.**
3. F4 judge: offline load, measure.
4. Serve judge → §2.2 re-extraction (judge).
5. Serve evaluator → §2.2 re-extraction (evaluator).
6. Offline: V5, V6, and the new Figure 1.

Nothing in §2.2 is invalidated by any F4 outcome — it is raw measurement — so the ordering
costs nothing if F4 goes badly.

---

## Amendment 1 — 2026-08-18, additions only, before any F4 measurement

*Made after inspecting the committed `mass_unnormalized` distribution (a quantity already
public in the paper), and before any full-vocabulary value exists.*

§2.1's sample stratifies on `log10(min option probability)` alone. That targets V3 and V4
well but targets **V2 badly**, and V2 is the prediction that can falsify the remedy. The
committed distributions show why:

| extractor | `mass_unnormalized` min | median | rows < 0.9 |
|---|---|---|---|
| judge | 0.9999 | 1.0000 | 0 |
| evaluator | **0.4379** | 0.9997 | **36** |

The judge's options already carry ~99.99% of all next-token mass, so there is almost no
room for missed option mass. The evaluator has a real off-option tail — and *missed option
variants are indistinguishable from genuine off-option mass in the committed data*. A
sample drawn on `min_p` alone would contain few or none of those 36 rows.

**Added stratum, pinned now:** on top of the decile design, the sample includes the **50
lowest `mass_unnormalized` records per extractor** (all 36 of the evaluator's sub-0.9 rows
fall inside this). These are the records where a missed variant could hide, so V2 is tested
where it can actually fail rather than where it is guaranteed to pass.

**Consequence for reporting, pinned now:** because this stratum is deliberately adversarial,
V2's pass/fail is evaluated on the **decile sample only**, and the low-mass stratum is
reported separately as a targeted probe. Pooling them would let an oversampled tail decide a
prediction that is about the corpus.

**Added control, pinned now:** while the evaluator is still served, re-extract ~20 committed
records through the served path and confirm the values are bit-identical to the cache. If
the server has drifted since the extraction ran, the whole offline/served comparison is
confounded, and no F4 number is reportable.

---

## Amendment 2 — 2026-08-18, after the drift control fired, before any offline measurement

*The control added in amendment 1 failed, and this records what was done about it. No
full-vocabulary value exists yet; the offline stage has not been run.*

**What happened.** Re-extracting committed records through a freshly launched evaluator does
**not** reproduce the cache exactly. Measured on 200 records
(`results/readout_validation/drift_200.json`):

| | |
|---|---|
| bit-identical to cache | **154/200 (77%)** |
| \|Δ\| > 1e-6 | 13/200 |
| \|Δ\| > 1e-4 | 4/200 |
| \|Δ\| > 1e-2 | 1/200 |
| median / p95 / max | 0.00 / 7.8e-6 / **1.27e-2** |

**It is drift, not nondeterminism, and that was tested rather than assumed.** On the
worst record, six identical repeat calls to the *running* server returned
`a = 0.18242552` every time, against a cached `a = 0.14804720` — stable within the instance,
different across instances. Within-instance repeats agree; across-instance they need not.
The likely cause is startup-time kernel and batching selection in vLLM.

**Consequence for F4, pinned now.** The registered comparison (offline vs the committed
cache) would confound the readout difference under test with a ~1e-2 instance-level noise
floor. F4 is therefore **re-baselined**: the offline full-vocabulary values are compared
against **freshly re-extracted served values from the same server instance**, captured
before that instance is stopped. The committed cache is no longer the comparator.

- V2/V3/V4 thresholds are unchanged; only the comparator changes.
- The cache-vs-fresh drift is **not** discarded. It is reported as a finding in its own
  right (below), because it bears on every exactly-reproducible claim the paper makes.

**New prediction, registered now, before the measurement.**

**V7 — The paper's headline claims survive the drift.** No perturbation of the size measured
above can create an exactly-zero score from a positive one, nor move the 90.6%/89.7%
asserted-zero shares materially, because both sit orders of magnitude away from the noise.
But **per-item calibration thresholds and release counts in §5 are single-instance
quantities**, since a score moving by 1e-2 can cross a threshold of that order.
*Falsified if* re-running §5's per-item measurement on freshly extracted scores changes the
qualitative structure (which items go inert, which collapse to zero release).
*Reported regardless:* the exact drift distribution above, and §5 must state that its counts
come from one server instance as well as one split.

---

## Amendment 3 — 2026-08-18, transport decided, before any offline measurement

§2.1 registered `transformers` as preferred and vLLM offline as the fallback. **The
fallback is taken**, and the reason is recorded so the choice is not read as convenient:
transformers cannot load these AWQ checkpoints without `gptqmodel`, which in turn needs a
`pcre` module absent from the package registry. `gptqmodel` was installed `--no-deps`
(leaving torch/vllm/xgrammar/transformers/triton byte-identical, verified before and after)
and then uninstalled when its own import chain failed. Chasing further leaf dependencies
was judged a worse risk to the imperatively-installed GPU stack than accepting the weaker
claim, which §2.1 already wrote down in advance.

**The independence claim, as it must now be stated in the paper:** the check is independent
of the HTTP server, the OpenAI-compatible layer, `guided_choice`, the top-100 window, and
the variant-aggregation code — but it **shares vLLM's model execution**. It validates the
*readout*, not the engine.

**How "full vocabulary" is realised, precisely.** vLLM's reported log-probabilities are
absolute values from the full softmax, not renormalized over a window. So requesting a wide
window (2,048) and summing the option-normalizing tokens inside it yields the **exact**
full-vocabulary option mass, provided every option-normalizing token id appears within that
window. That condition is **checked per record and reported**; where it fails, the record is
excluded and counted, rather than silently contributing a truncated sum — which would be
this paper's own defect committed inside its own validation.
