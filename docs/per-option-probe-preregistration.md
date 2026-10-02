# Per-option probe pre-registration — bounding v1's truncation, predicted first

*Written and committed 2026-08-16, BEFORE `experiments/per_option_probe.py` exists and
before any file under `results/per_option_probe/` is produced. Frozen on harness landing;
supersessions live in the probe's REPORT, never here.*

**Purpose.** F-11 established that v1's recorded "probabilities" are renormalized mass over
options surviving a top-20 cut (support-1 on 52.3%/80.1% of s3 judge/evaluator rows). The
v2 pool measures per-option scores on *Powergrading*; **only this probe can bound the
artifact's magnitude on the S3/M3 pools the committed v1 results stand on.** It re-extracts
a subsample with `canonical_per_option` and compares against the committed truncated
distributions, record by record.

**Run window.** The probe shares the week-2 serving session (judge, then evaluator — one
model resident at a time per ops rules) so no extra model loads are spent on it. It runs
BEFORE the v2 extraction in each serving turn.

## 0 · Disclosure

Seen: everything in `research/experiment-audit.md` (the concentration and zero-score
tables, the threshold table), both audit waves, and the committed `extract-*.jsonl` files
themselves. **Not seen:** any per-option score on any real model output (mock-tested only);
any TV distance between a truncated and per-option distribution anywhere.

## 1 · Design, pinned

- **Sample:** 200 records per pool (s3, m3), stratified: 100 uniformly at random from
  support-1 rows, 50 from support-2, 50 from support-3, by FNV-1a hash order of record id
  (seed-free, deterministic). Both extractors on every sampled record.
- **Extraction:** `canonical_per_option` with the v1 prompt (`CANONICAL_PROMPT`, options
  a/b/c — the SAME prompt the committed extractions used, so the only changed variable is
  the scoring transport).
- **Outputs per record:** per-option `raw_logprob`, `mass_unnormalized`, the renormalized
  dist, TV distance to the committed truncated dist, and both nonconformity scores for the
  gold option.

## 2 · Predictions

**Q1 — Truncation understated the truth's nonconformity.** On support-1 rows where the
surviving option is gold (committed score exactly 0.0), the per-option gold score is
**> 0 in ≥ 95%** of rows. *Falsified if* more than 5% remain exactly 0.
**Q2 — The artifact is large where F-11 says it is.** Median TV distance on support-1 rows
**≥ 0.05**, and support-1 median TV **> support-3 median TV**. *Falsified if* either fails.
**Q3 — Decisions mostly survive; scores do not.** The argmax option changes on **< 10%** of
rows (the choice was robust; the calibration input was not). *Falsified if* ≥ 10% flip.
**Q4 — The threshold dial comes alive.** Recomputing `calibrate_threshold` on the probe's
gold scores (per pool, both extractors pooled via `combine_min` on probe rows only): the
α = 0.1 threshold is **strictly positive**. *Falsified if* it is exactly 0. *(Labelled: a
200-row threshold is an illustration of the dial, not a calibration claim.)*

## 3 · Reported regardless

Full TV histograms by support stratum and by extractor; the share of rows where
`mass_unnormalized < 0.5` (option tokens not dominating the next-token mass — the residual
v1 could not express); vLLM version + sampler env recorded in the output header; and every
row's raw per-option logprobs, so the analysis is re-derivable without re-serving.
