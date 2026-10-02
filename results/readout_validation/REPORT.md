# Readout validation — results

Design and predictions: `docs/readout-validation-preregistration.md` (committed 2026-08-18,
before any harness here existed). Results are reported below whether or not they favour the
paper.

---

## V1 — Is the variant set complete? *(tokenizer-only, no GPU)*

`experiments/readout_v1_variant_set.py` → `v1_variant_set.json`. Enumerates the entire
vocabulary of each model and asks which tokens each normalizer maps onto an option letter.

| model | vocab | `_norm_token` matches | old normalizer matches | candidate misses |
|---|---|---|---|---|
| judge — Qwen3-32B-AWQ | 151,669 | **33** | 18 | 103 |
| evaluator — gemma-3-27b-it-qat-autoawq | 262,145 | **12** | 12 | 8 |

### The normalization-width variable, now scoped exactly (bears on V5)

The corrected readout's normalizer is strictly wider than the old one, but **only on the
judge**, and by exactly 15 token forms — five per option:

```
'a': (A  (a  )a  .A  .a        'b': (B  (b  )b  .B  .b        'c': (C  (c  )c  .C  .c
```

On the **evaluator the two normalizers are identical** — both match exactly
`{A, a, " A", " a"}` per option, 12 tokens. So any normalization contribution to the
evaluator's committed asserted-zero rate is **exactly zero, by construction**, and the V5
attribution question reduces to the judge alone.

This does not yet say how much mass those 15 forms carry. That is the GPU measurement.

### Candidate misses — what neither normalizer catches

- **Judge (103):** all punctuation-prefixed letters — `"a`, `#a`, `$a`, `%A`, `&A`, `'a`,
  `*A`, `+a`, `,b` … A token like `"a` is a plausible emission if the model starts a quoted
  answer, so these are not dismissible a priori.
- **Evaluator (8):** `<a>`, `</a>`, `<b>`, `</b>` and four combining-diacritic artifacts
  (`́a`, `̉a`, `́c`, `̣c`). None is a plausible spelling of an option in this prompt.

**V1 is not yet decided.** Per the pre-registration it is falsified only if a missed token
*also carries ≥1e-6 mass on a real record*, which the full-vocabulary pass measures. The
candidate-miss set above is the list that pass must check.

### One correction to this measurement, disclosed

The first version of the candidate detector stripped `[^0-9A-Za-z]`, which deletes non-ASCII
**letters** as well as punctuation. It therefore counted ` Aç`, ` Bà`, ` Có`, ` đưa` and 230
other word fragments in other languages as spellings of a/b/c — 237 false positives that
would have buried any real miss. The detector is now unicode-aware (`str.isalnum()`), and
the numbers above are from the corrected version.

---

## V2/V3/V4 — full-vocabulary check, evaluator (gemma-3-27b-it-qat-autoawq)

Transport: **vLLM offline** (prereg amendment 3 — `transformers` cannot load these AWQ
checkpoints without `gptqmodel`, whose own import chain is broken here). Independent of the
HTTP server, the OpenAI layer, the top-100 window and the aggregation code; **shares vLLM's
model execution**. It validates the *readout*, not the engine, and the paper must say so.

Comparator: the same-instance served baseline, not the committed cache (amendment 2).

**Positive control passed:** all 347 prompts tokenize identically to the server's own
`/tokenize` output.

**Measurement is exact, not approximate.** Absent option tokens are *bounded*, never dropped
to zero — the paper's own defect must not appear inside its validation. Worst missing-mass
bound: **1.45e-13 relative**.

### Decile sample (n=300) — the sample the predictions are scored on

| prediction | threshold | measured | |
|---|---|---|---|
| **V1** variant set complete | miss-token mass < 1e-6 | max **6.41e-12** | **held** |
| **V2** window misses no option mass | median ratio ∈ [0.99, 1.01], p99 dev < 0.05 | median **1.000000**, range [0.9842, 1.0279], p99 dev **0.0129** | **held** |
| **V3** scores agree | ≥95% within factor 2 | **100%**, median ratio **1.000000092** | **held** |
| **V4** no zeros | 0 exact zeros | **0** | **held** |

### Low-mass stratum (n=47) — adversarial, reported apart as pinned

Median mass ratio 1.0002, but range [0.903, 1.160] and p99 deviation 0.16. Agreement is an
order of magnitude looser where the options carry least of the distribution. Scores still
agree (100% within factor 2, median 0.99999999). Reported, not pooled.

**So the remedy is corroborated** by a path sharing no readout code — and corroborated only
to the extent amendment 3 allows: the readout is validated, the engine is not.

---

## Two findings that were not predicted

### The decoding constraint was silently dropped, and the v2 corpus is unconstrained

`guided_choice` **does not exist in vLLM 0.25.1** — it appears in zero files in the package;
`SamplingParams` carries `structured_outputs` instead. It was passed via `extra_body` and
ignored. So all 13,992 × 2 v2 extractions ran with no constraint.

The old corpora were fine: all 5,236 s3/m3 rows have `emitted ∈ {a,b,c}` and `diverged = 0`,
so the constraint was honored on the older stack and lost in an upgrade.

**Blast radius, measured rather than argued.** Masking only subtracts a constant from
surviving log-softmax entries, so log-odds — and therefore every `dist`, score, threshold,
set and release count — are invariant. And the leading token is an option anyway on
**346 of 347** records (the one exception leads with `'incorrect'`). What is wrong is the
paper's *description* of its transport (`main.tex:132`, `:242`), and C7's attribution of the
failure to single-option grammars specifically when the parameter was ignored at every arity.

### The option rank — the quantity the paper says it cannot report

§3 states: *"What we cannot report is the rank of the lowest-ranked option variant: the raw
top-k list was not persisted."* The offline pass persists it.

| | |
|---|---|
| median option rank | **2** |
| p95 / max | **4 / 4** |
| ranks beyond a top-20 window | **0 of 694** |
| ranks beyond top-100 | **0 of 694** |

**This bears directly on §5.** On the binary corpus an option never leaves a top-20 window,
so a *rank-level* top-20 truncation there would be **completely inert**. The τ-floor
simulation is therefore not a stand-in for rank truncation on this corpus: it models the
*consequence* (an option recorded as 0) rather than the *cause* (an option falling out of the
window). With only two options the artifact cannot arise by rank at all — it took three.
§5 already labels itself a controlled simulation and not a rank-level reconstruction, which
was prescient; it now needs to say why.

---

## V2/V3/V4 — full-vocabulary check, judge (Qwen3-32B-AWQ)

Same registered protocol, same criteria, unchanged. Comparator is the same-instance served
baseline (amendment 2). Positive control passed on all 347 prompts — and needed a fix first:
`Extractor` auto-sets `enable_thinking=False` for Qwen3, so the `/tokenize` control had to
pass the same `chat_template_kwargs` or it would have failed for a reason unrelated to the
readout. Gemma has no such switch, which is why the evaluator run never exposed it.

Worst missing-mass bound **5.33e-11** relative; absent option tokens bounded, never zeroed.

### Decile sample (n=299) — scored

| prediction | threshold | judge | evaluator, for comparison | |
|---|---|---|---|---|
| **V1** variant set | miss mass < 1e-6 | **1.91e-7** | 6.41e-12 | held |
| **V2** option mass | median ∈ [0.99,1.01], p99 dev < 0.05 | median **1.000000000**, p99 dev **6.6e-7** | median 1.000000, p99 dev 0.0129 | held |
| **V3** scores | ≥95% within factor 2 | **100%**, median **1.000000000** | 100%, median 1.000000092 | held |
| **V4** zeros | 0 | **0** | 0 | held |

The judge agrees an order of magnitude more tightly than the evaluator on both mass and
scores (p05/p95 of the score ratio: 0.938/1.056 against 0.779/1.455).

### Low-mass stratum (n=48) — reported apart, never pooled

Mass ratio median 1.00000004, range [0.999998, 1.000003], p99 dev 3.5e-6. Scores 100% within
a factor of 2. Unlike the evaluator — whose adversarial stratum loosened to [0.90, 1.16] —
the judge's stays tight, because its options carry essentially all the next-token mass.

### Disagreement cases, manually inspected

Worst score ratios span **0.920 to 1.065**, all far inside the factor-of-2 band, on absolute
scores of order 1e-7 to 1e-3. They are consistent with cross-instance numerical variation
(C65), not with a readout defect.

The judge's miss-token mass is five orders larger than the evaluator's (1.91e-7 vs 6.41e-12)
because its candidate-miss set is 103 punctuation-prefixed tokens against 8. That is above
the paper's smallest reported option probabilities, so it was inspected directly rather than
waved past the 1e-6 threshold: in **0 of 347** records does the missed mass exceed the
smallest option's own mass, worst ratio **0.13**. Only the aggregate was persisted, not which
tokens carry it.

### Unpredicted: variant aggregation is load-bearing, and the prompt is why

The judge's leading token is a **parenthesized** option (`(a`, `(b`) on **220 of 347 records
(63.4%)**, and on those the parenthesized spelling alone carries a median **82.7%** of that
option's mass (range 50.8–100%). A normalizer matching only bare letters would have measured
the residue and renormalized it.

This also explains why V5's normalizer route is **zero** on s3/m3. The binary prompt lists
options as "(a) correct / (b) incorrect"; the 3-option prompt says only "Reply with exactly
one of: a, b, c" — and there the judge emits bare letters (s3: 280/260/230, matching gold
exactly). The old normalizer sufficed on the corpora it actually ran on **because of the
prompt format**, not by luck, and would not have sufficed on the binary prompt.

### Assessment: do the judge-side mechanism claims still hold?

**Yes, and they are now better supported.**

- **C62's decomposition (99.6% writer / 0.4% window)** rests on a replay through the corrected
  readout, and all 31 window-route zeros are the judge's. That readout is now independently
  validated on the judge, more tightly than on the evaluator. The inputs are sound.
- **C69 (normalizer route zero)** holds and is now *explained* rather than merely observed —
  see above. Its scope tightens honestly: dead on the corpora where the old readout ran; it
  would not have been dead under the binary prompt.
- **C68's rank claim** extends to the judge unchanged: median 2, p95 3, max 4, 0/694 beyond
  top-20, and 0/347 records whose leading token is not an option.

No judge-side claim was weakened. The "checked only against itself" objection is closed for
both extractors, subject to the standing transport caveat: vLLM-offline validates the
readout, not the engine.

---

## Scorecard: V5, V6, V7 (the registered predictions not yet scored above)

### V5 — Normalization is a small share of the artifact: **HELD**

Registered threshold: the share of committed asserted-zero rows explained by normalization
width rather than truncation is below 5%. Measured, after the wave-10 window bug was fixed
(the first attribution tested the old normalizer over the top-100 when v1 only saw the
top-20, which hid this route entirely): **direct 97/7,861 = 1.2%**, **counterfactual
256/7,861 = 3.3%**. Both readings sit under the registered 5%. Held — but the route is real,
not absent, and the paper's §2 table now carries both columns.

### V6 — Rank-level truncation reproduces the mechanism: **FALSIFIED**

`experiments/readout_v6_ranksweep.py` → `v6_ranksweep.json`. A pure rank cut on the retained
s3/m3 windows (corrected normalizer, full precision, no writer), swept over k:

| k | zeros produced | share of the 15,708 option slots |
|---|---|---|
| 5 | 3,171 | 20.2% |
| 10 | 617 | 3.9% |
| **20 (deployed)** | **31** | **0.2%** |
| 50 | 0 | 0% |
| 100 | 0 | 0% |

The committed artifact is 7,861 zeros (50% of slots). Rank truncation cannot reproduce it at
the deployed k — it produces 0.4% of it — and cannot reach the ~90% score atom at any swept
k. On the binary corpus it is fully inert (best-variant rank max 4, 0/1,388 beyond top-20,
both extractors). Cross-check: k=20 yields exactly the attribution's 31.

The registered expectation was written when the rank story was believed; the falsification
is the decomposition seen from the other side. The prereg's own consequence clause — rewrite
§5's mechanism to the rank-level result rather than defending the floor — was executed in
waves 9–10, before this scoring, which is disclosed rather than hidden: the paper's current
§5 already describes the writer as the mechanism and the τ-floor as its faithful model.

One shape worth keeping: the sweep shows the window route is *k-sensitive* — at k=5 it would
have manufactured a real artifact (20% of slots). The deployed k=20 happened to be wide
enough that the writer preempted everything. A pipeline with a narrower window would meet
the same failure through the other door.

### V7 — §5's structure survives a fresh server instance: **HELD**

Scored as registered: a full fresh-instance extraction of the binary corpus (13,992 × 2,
`results/v2_instance2/`, frozen caches untouched), then §5's per-item measurement re-run on
it. The comparison harness (`experiments/readout_v7_instance.py`) carries a faithfulness
guard — it must reproduce `per_item_honest.json` exactly from the frozen caches before it
may score the fresh instance. It did (10 items × 2 gates × 3 fields).

**Result: every per-item threshold bit-identical. Inert set {13,4,5,6} unchanged.
Zero-release set {20,3} unchanged. Wrong-verdict totals 57/44 unchanged.** The only
movement anywhere is item 1's measured release count, 104 → 103 (one test record crossing a
3.2e-8 threshold). 0 exact zeros in the 27,984 fresh rows.

Fresh-vs-committed agreement: judge 99.62% bit-identical (max |Δ| 7.5e-3), evaluator
**99.98%** (max 4.3e-9).

**A refinement of C65 falls out.** C65's drift control — 154/200 bit-identical, worst
1.27e-2 — ran against the *day-old running instance*, the same process that wrote the cache.
An actual relaunch reproduces the cache at 99.98%/99.62%. So the instability C65 measured is
**within-lifetime state drift, not launch-time nondeterminism**: a clean instance is nearly
deterministic; a long-running one wanders from its own past outputs. C65's numbers stand;
its interpretation is corrected (C80), and App. D's description in the paper is updated.

The scorecard is complete: **V1 held · V2 held · V3 held · V4 held · V5 held ·
V6 falsified (informatively) · V7 held.** Every registered prediction is scored and
reported, including the one that failed.
