# The canonical extraction object (S2)

*Amendment 2026-08-15 — one claim in this document was FALSE and is corrected in place;
the superseded sentence is preserved verbatim below so the history is not wiped. The
experiment audit (`research/experiment-audit.md`, architecture finding **F-11**) measured
that the single-call renormalization is **not** equivalent to per-option scoring whenever an
option falls outside the returned top-k, which happens on 52.3%/51.4% of judge rows and
80.1%/79.5% of evaluator rows [E-114][E-115]. §"What the distribution actually is" is new;
rule 1 gains a precondition; the stale `build_set` reference is corrected to
`conformal_set`.*

**One definition, used identically by every family:** a single
`/v1/chat/completions` call with `guided_choice=["a","b","c"]`; the
decision-token top-logprobs are renormalized over {a,b,c} (aggregating
leading-space token variants); the **extraction is the argmax of that
distribution**, and the distribution itself is the conformal score input for M2.
The constrained-decode *emitted* token is recorded as a diagnostic only.
Implementation: `Extractor.canonical()` in `src/verdict_cert/extraction.py`.

## Why (the S1 confound)

S1 ran extraction with `guided_choice=["a","b","c","unclear"]` and took the
emitted token as the answer. Mistral then produced rows like: emitted
`unclear` while its own recovered distribution put P(a)=0.95. The cause is a
tokenizer asymmetry: `" unclear"` is a single token for Mistral (a live
competitor at the decision point) but multi-token for Gemma (invisible in any
top-k, never emitted). An asymmetric competitor makes abstention behavior a
property of the tokenizer, not the model's belief — and it is a confound for
S3: it would make Mistral look spuriously decorrelated from the other families
and could mis-seat the judge/evaluator pair.

## The two design rules that follow

1. **`unclear` is not a competing token.** Abstention/uncertainty is a property
   of the distribution's *shape* (low max-prob, high entropy) and is handled by
   conformal thresholding in M2 (`conformal_set` over nonconformity scores — the
   normative builder; the legacy `build_set` this rule originally named is
   deprecated, architecture finding F-3), never by letting a fourth token compete
   asymmetrically.
   ⚠ **Precondition, added 2026-08-15 (F-11):** the shape argument holds only while the
   option set survives the top-k cut. When it does not, the shape is flattened to a point
   mass and uncertainty becomes unrepresentable at this layer — so on the majority of rows
   this rule's mechanism is defeated before conformal thresholding ever sees the score.
2. **The emitted token is never the answer.** The decision is the argmax of the
   renormalized {a,b,c} distribution. Emitted≠argmax divergences are logged as
   diagnostics (they track the constrained-decode anomaly observed in S1).

## What the distribution actually is (F-11)

**It is the renormalized mass over the options present in the decision token's top-k — not
the model's distribution over {a,b,c}.** An option absent from the returned top-k is recorded
as mass **0.0: asserted, not measured.** When only one option survives, renormalization
returns exactly **1.0** — a certainty the model never expressed — and a nonconformity score
of exactly **0.0**. Measured: only one option survives on **52.3%/51.4%** of judge rows and
**80.1%/79.5%** of evaluator rows (s3/m3) [E-114][E-115], which drives **90.6%/89.7%** of
true-option scores to exactly zero under `combine_min` [E-116][E-117]. Consequences and the
full derivation: `research/experiment-audit.md`; disposition: architecture §9 finding F-11.

The correct statement of the equivalence is **conditional**:

> Since a/b/c are single tokens for every family in play, the single-call distribution equals
> per-option sequence scoring **exactly when all three options fall inside the returned
> top-k**, at one call instead of three and with no assumptions about forced-path logprob
> semantics. Outside that condition it does not, and the gap is one-sided: the surviving
> option is credited with the missing mass.

Both directions are pinned by `tests/test_extraction_canonical.py`
(`test_equivalence_holds_only_when_every_option_survives_the_cut` and
`test_truncation_manufactures_certainty`).

> **Superseded 2026-08-15, preserved verbatim.** The original claim was:
> *"Since a/b/c are single tokens for every family in play, the single-call distribution is
> mathematically identical to per-option sequence scoring, at one call instead of three and
> with no assumptions about forced-path logprob semantics."*
> Single-tokenness is necessary but **not sufficient** — the options must also survive the
> top-k cut. This sentence is the reason the top-20 design was believed safe, and it is
> banned from returning by `tests/test_doc_coherence.py`.

## Seating note (filled by S3)

S3 measures pairwise error correlation on channel-corrupted answers using this
canonical object for all three families and seats the least-correlated pair.
See `results/s3/REPORT.md`.
