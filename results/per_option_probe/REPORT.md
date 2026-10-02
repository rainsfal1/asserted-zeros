# Per-option probe — REPORT

Scores the pre-registered Q1–Q4 (`docs/per-option-probe-preregistration.md`).
Stratum defined by the judge's committed distribution (harness pin, §doc).

Each question carries its registered verdict below — the prereg commits to reporting outcomes "right or wrong", which a table of measurements without verdicts does not do (audit wave 5).

## s3/judge (n=200)

- **Q1 — HELD**: gold score leaves exactly-zero on 100/100 support-1-gold rows (registered: > 0 in ≥ 95%)
- **Q2 — FALSIFIED**: median TV support-1 0.0000 vs support-3 0.0001 (registered: ≥ 0.05 and support-1 > support-3)
- **Q3 — HELD**: argmax flips 0/200 (registered: < 10%)
- mass_unnormalized < 0.5: 0/200
- UNSCOREABLE (an option absent even at top-100, so refused rather than zeroed): 0

## s3/evaluator (n=200)

- **Q1 — HELD**: gold score leaves exactly-zero on 156/156 support-1-gold rows (registered: > 0 in ≥ 95%)
- **Q2 — FALSIFIED**: median TV support-1 0.0000 vs support-3 0.0000 (registered: ≥ 0.05 and support-1 > support-3)
- **Q3 — HELD**: argmax flips 0/200 (registered: < 10%)
- mass_unnormalized < 0.5: 0/200
- UNSCOREABLE (an option absent even at top-100, so refused rather than zeroed): 0

**Q4 — HELD** (s3, illustration at n=200): α=0.1 threshold = 0.0013478436208598232 (registered: strictly positive), exactly-zero share = 0.000

## m3/judge (n=200)

- **Q1 — HELD**: gold score leaves exactly-zero on 99/99 support-1-gold rows (registered: > 0 in ≥ 95%)
- **Q2 — FALSIFIED**: median TV support-1 0.0000 vs support-3 0.0000 (registered: ≥ 0.05 and support-1 > support-3)
- **Q3 — HELD**: argmax flips 0/200 (registered: < 10%)
- mass_unnormalized < 0.5: 0/200
- UNSCOREABLE (an option absent even at top-100, so refused rather than zeroed): 0

## m3/evaluator (n=200)

- **Q1 — HELD**: gold score leaves exactly-zero on 162/162 support-1-gold rows (registered: > 0 in ≥ 95%)
- **Q2 — FALSIFIED**: median TV support-1 0.0000 vs support-3 0.0000 (registered: ≥ 0.05 and support-1 > support-3)
- **Q3 — HELD**: argmax flips 0/200 (registered: < 10%)
- mass_unnormalized < 0.5: 0/200
- UNSCOREABLE (an option absent even at top-100, so refused rather than zeroed): 0

**Q4 — HELD** (m3, illustration at n=200): α=0.1 threshold = 0.00017423629631152515 (registered: strictly positive), exactly-zero share = 0.000

