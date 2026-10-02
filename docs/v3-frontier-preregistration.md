# V3 — the certification frontier: pre-registration

*Status: FROZEN on commit. Amendments are dated additions only; §§1–8 are never edited in
place. Written and committed BEFORE the harness exists — the commit order is the evidence.*

**Date:** 2026-08-17.

## 0 · Why this experiment

V2 answered "does the deployed operating point certify?" with a measured no: release is
0.000 in every arm at every registered Σα, because per-item conformal thresholds sit at
0.999998 — the (1−αᵢ) quantile of a calibration stream in which a large minority of rows
score ≈ 1.0 (`research/v2-outcome-note.md`). That is a boundary, and it is worth reporting.
It is not yet an answer to the question a reader will ask next.

This experiment asks the general question instead:

> **Under what measurable operating conditions can an uncertainty-based decision gate
> provide useful certification?**

Not "does our system work". The object of study is the *operating region*, and an empty
region is as publishable as a large one — provided it is mapped rather than asserted.

## 1 · What is fixed before anything runs

- **Data:** `results/v2_pool/` — 13,992 rendered records and BOTH §7.2 extraction caches,
  already committed and byte-pinned (`records.sha256`). **No new extraction, no
  re-rendering, no model calls.** This experiment is pure offline arithmetic over evidence
  that already exists, which is what makes it immune to fishing at the elicitation layer.
- **Split:** `m6.split_students(students, 0)` → train 116 / cal 117 / test 117, the same
  seed-0 split every v2 number used. Train is UNUSED (nothing is fitted). Two students lack
  a full item set (§7.4 of the v2 prereg), so n_test = 116 sessions.
- **Score:** nonconformity `s(o) = 1 − p(o)`, with `p` the measured per-option probability
  and the two extractors combined by `combine_min` — identical to v2. No re-scoring.
- **Gate:** per-item set `{o : s(o) ≤ t_item}`, empty → FULL; release iff the reachable
  verdict set is a singleton (`uscis.certify`). **No re-ask loop** — the loop is v2's
  object; this is the static frontier.

## 2 · Axis A — decision complexity

`k ∈ {1, 2, 3, 5, 10}`. Items are ordered by FNV-1a hash of their id (seed-free). For each
k, **five** subsets are evaluated: `subset_j = ordered[(j·k) mod 10 …]`, k consecutive with
wraparound, j = 0…4 (for k = 10 the five coincide and one is reported). Evaluating five
subsets rather than one is the anti-fishing device: a lucky triple cannot carry the claim.

Pass mark scales as `pm = max(1, round(6k/10))` — the rule the frozen M6 sensitivity arm
already used, adopted here so the comparison is to precedent rather than to a fresh choice.

## 3 · Axis B — extraction error

Calibration and evaluation are stratified by channel level (Mondrian conformal): **`tel`,
`snr0`, `snr_m5`, `snr_m10`**, plus **`mixed`** — v2's hash-assigned stream — as the
registered reference. Calibration rows and test rows always come from the SAME stratum.

**Stratifying costs no sample size here**, and that is what makes this axis clean: every
student answers every item at every level, so n = 117 calibration rows per (item, stratum),
exactly as in v2's mixed stream. The feasibility floor `k/(n+1)` is therefore IDENTICAL
across strata, and any movement in the frontier is attributable to ε alone.

**ε is defined threshold-free:** per (item, stratum), the fraction of calibration rows whose
true option is not the argmax of the min-combined distribution. Reported per item and as the
stratum mean ε̄.

## 4 · The sweep, and what is reported regardless

Per-item budget `αᵢ ∈ {0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40}`; the implied
total is `Σα = k·αᵢ` and is reported beside every cell. **Any cell with Σα > 1 is printed
with a NOT-A-GUARANTEE marker** and may never be quoted as a guarantee; such cells exist
only to show the curve's shape.

Reported for every (k, subset, stratum, αᵢ) cell whatever the outcome: certified release
rate; **wrong-release probability P(release ∧ wrong), unconditional** (the project's
corrected guarantee — not error-among-released, which is also reported beside it); the
per-item thresholds and how many are degenerate (≥ the stratum's max calibration score);
mean uncertainty-set size; ε̄; and n beside every rate.

Three baselines accompany every risk-coverage curve:

1. **always-answer** — release every session with the point-estimate verdict, no abstention.
2. **confidence threshold** — release iff `min over items of max_o p(o) ≥ c`, with
   `c ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999}`; verdict is the point
   estimate. No calibration, no guarantee. **c is swept, never tuned on test.**
3. **conformal gate** — this system.

## 5 · Predictions, with falsifiers

**Q1 — the union bound is the price of conjunction.** Within a stratum, the smallest αᵢ
reaching release ≥ 0.50 is weakly increasing in k, so the required **Σα grows at least
linearly** in k. *Falsified if* Σα_required is non-monotone in k in ≥ 2 strata, or if any
stratum has Σα_required(k=10) < Σα_required(k=1).

**Q2 — ε orders the frontier.** Within each k, the required αᵢ is ordered
`tel ≤ snr0 ≤ snr_m5 ≤ snr_m10`, matching the measured ε̄ ordering. *Falsified if* that
ordering inverts for ≥ 2 of the five k values.

**Q3 — a useful region exists.** At least one (k, stratum) cell attains **release ≥ 0.50 at
Σα ≤ 0.10** with realized wrong-release ≤ Σα. *Falsified if* no cell in the grid does.
*(Genuinely open: ε̄_tel ≈ 0.054 already exceeds αᵢ = 0.05, so this may hold at k = 1 and
fail by k = 3. A falsification here is the stronger negative result, not a failure.)*

**Q4 — validity holds where it is claimed.** In every cell with Σα ≤ 1, realized
P(release ∧ wrong) ≤ Σα + binomial 95% slack at n = 116. *Falsified if* any cell exceeds it.

**Q5 — conformal buys the guarantee, not the frontier.** At matched release (banded
[target, target+0.05]) the conformal gate's wrong-release rate is within ±20% relative of
the swept confidence baseline. *Falsified if* either dominates the other by more than that
at ≥ 2 matched targets. *(We predict the two frontiers coincide: both threshold the same
scores. The conformal gate's contribution is a distribution-free guarantee fixed in advance,
not a sharper curve — and saying so before measuring is the honest position.)*

## 6 · Figures

- **Figure 1** — release rate vs ε̄, one line per k, at fixed αᵢ (0.05 and 0.10 panels).
- **Figure 2** — required Σα (for release ≥ 0.50) vs k, one line per stratum, with the
  k·αᵢ reference line.
- **Figure 3** — risk-coverage: wrong-release probability vs release rate, comparing
  always-answer, confidence threshold and conformal gate, per stratum.

## 7 · Forbidden — the ways this could be rescued artificially

Named in advance so a reader can check they did not happen: choosing item subsets by
outcome; tuning the confidence baseline's `c` on test; reporting the best stratum only;
quoting a Σα > 1 cell as a guarantee; selecting k after seeing the curves; changing the
score, split or combiner; re-extracting anything. If a design change becomes necessary it
is a dated §9 amendment, additions-only, with the reason stated before the new numbers.

## 8 · Success criterion

**The experiment succeeds if the region is mapped, whichever shape it has.** A large useful
region and an empty one are both reportable; what would make it a failure is an unmapped
boundary, a curve without its baselines, or a guarantee quoted outside the region where it
holds.

---

## 9 · Amendment, 2026-08-17 — two registered tests were the wrong instrument (additions only)

*§§0–8 byte-identical. Written after the run, so the registered verdicts stand exactly as
§5 defines them: **Q1 HELD, Q2 HELD, Q3 HELD, Q4 FALSIFIED, Q5 FALSIFIED.** Nothing below
changes a verdict; it records what the registered tests could not distinguish, because a
pre-registration can commit to a bad test and the honest response is to say so.*

### 9.1 · Q4's slack omitted the dominant term

Q4 compared realized wrong-release against `Σα + binomial 95% slack at n = 116`. That slack
is the sampling error of the **test** set. The dominant term at this size is the
**calibration** draw: split-conformal's realized miscoverage, conditional on one calibration
set, is `Beta(n+1−l, l)` with `l = ⌈(n+1)(1−α)⌉` (Vovk 2012), whose 95th percentile is
**2.55× α at α = 0.01, 2.01× at 0.02, 1.54× at 0.05, 1.42× at 0.10** for n = 116. Of 12
marginal breaches, **11 fall inside that band and one survives** — within multiplicity
expectation for the number of cells scored.

The finding this exposes is not "the guarantee failed" but **how wide the honest per-split
guarantee is**: at the α where this method is useful, a practitioner's realized risk is up
to 2.5× the nominal budget on a 116-row calibration split. Bimodal LLM confidence makes
that maximal — the quantile lands on a near-vertical segment of the score CDF, so a small
split difference moves realized coverage directly, with no margin to absorb it. Measured on
the one breaching cell: calibration error 16.4% vs test error 26.7% for the same item.

### 9.2 · Q5's ±20% relative band is undefined near the resolution limit

At n = 116 one session is 0.0086, so comparing 0.0000 against 0.0086 registers as an
infinite relative difference. The registered test therefore counted noise as disagreement.
In absolute terms over the same 91 matched-release bands: **conformal lower in 54,
confidence lower in 22, tied in 15, median absolute gap 0.0517.**

So the frontiers do not coincide, the registered prediction was wrong, and it was wrong in
the method's favour: per-item calibrated thresholds adapt to item difficulty, which one
global confidence cut cannot do. This is reported as a *falsified prediction with a
measured direction*, not converted into a success.

### 9.3 · One reading the prereg left open

§2 pinned five item subsets per k as an anti-fishing device but did not pin how to
aggregate them. Scored on the **mean** release across subsets, with per-subset min–max
reported beside every number.
