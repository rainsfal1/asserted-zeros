# V3 — the certification frontier: REPORT

Scores the FROZEN questions of `docs/v3-frontier-preregistration.md` §5 over the committed v2 pool. Pure offline arithmetic: no model calls, nothing fitted, no re-extraction.

**Reading pinned here (the prereg left it open):** the five item subsets per k are aggregated by MEAN release, with the per-subset min/max reported beside it.

## Measured extraction error (§3)

| stratum | ε̄ (mean over items) | n_cal | n_test |
|---|---|---|---|
| `tel` | 0.0759 | 116 | 116 |
| `snr0` | 0.1974 | 116 | 116 |
| `snr_m5` | 0.3207 | 116 | 116 |
| `snr_m10` | 0.4672 | 116 | 116 |
| `mixed` | 0.2560 | 116 | 116 |

## Registered verdicts

- **Q1: HELD**
- **Q2: HELD**
- **Q3: HELD**
- **Q4: FALSIFIED**
- **Q5: FALSIFIED**

## Required Σα for release ≥ 50% (§5 Q1/Q2)

`—` = unreachable anywhere on the α grid. Cells with Σα > 1 are NOT a guarantee and are marked ⚠.

| stratum | ε̄ | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|---|
| `tel` | 0.076 | 0.010 | 0.020 | 0.030 | 0.100 | 0.500 |
| `snr0` | 0.197 | 0.100 | 0.200 | 0.450 | 0.750 | 2.000 ⚠ |
| `snr_m5` | 0.321 | 0.150 | 0.400 | 0.900 | 1.500 ⚠ | 4.000 ⚠ |
| `snr_m10` | 0.467 | 0.300 | 0.800 | 1.200 ⚠ | 2.000 ⚠ | 4.000 ⚠ |
| `mixed` | 0.256 | 0.150 | 0.400 | 0.600 | 1.500 ⚠ | 3.000 ⚠ |

## Risk–coverage, k = 10 and k = 1 (mean over subsets)

| stratum | k | αᵢ | Σα | release (min–max) | wrong-release | guarantee? |
|---|---|---|---|---|---|---|
| `tel` | 1 | 0.01 | 0.01 | 0.722 (0.03–0.99) | 0.0190 | yes |
| `tel` | 1 | 0.02 | 0.02 | 0.741 (0.03–0.97) | 0.0190 | yes |
| `tel` | 1 | 0.05 | 0.05 | 0.734 (0.09–0.96) | 0.0207 | yes |
| `tel` | 1 | 0.1 | 0.1 | 0.848 (0.53–0.99) | 0.0379 | yes |
| `tel` | 1 | 0.15 | 0.15 | 0.888 (0.85–0.94) | 0.0431 | yes |
| `tel` | 1 | 0.2 | 0.2 | 0.883 (0.81–0.94) | 0.0552 | yes |
| `tel` | 1 | 0.3 | 0.3 | 0.807 (0.70–0.92) | 0.0483 | yes |
| `tel` | 1 | 0.4 | 0.4 | 0.733 (0.56–0.90) | 0.0293 | yes |
| `tel` | 10 | 0.01 | 0.1 | 0.207 (0.21–0.21) | 0.0086 | yes |
| `tel` | 10 | 0.02 | 0.2 | 0.241 (0.24–0.24) | 0.0000 | yes |
| `tel` | 10 | 0.05 | 0.5 | 0.655 (0.66–0.66) | 0.0086 | yes |
| `tel` | 10 | 0.1 | 1.0 | 0.845 (0.84–0.84) | 0.0259 | yes |
| `tel` | 10 | 0.15 | 1.5 | 0.905 (0.91–0.91) | 0.0259 | NO ⚠ |
| `tel` | 10 | 0.2 | 2.0 | 0.905 (0.91–0.91) | 0.0259 | NO ⚠ |
| `tel` | 10 | 0.3 | 3.0 | 0.750 (0.75–0.75) | 0.0172 | NO ⚠ |
| `tel` | 10 | 0.4 | 4.0 | 0.664 (0.66–0.66) | 0.0086 | NO ⚠ |
| `snr0` | 1 | 0.01 | 0.01 | 0.021 (0.00–0.08) | 0.0069 | yes |
| `snr0` | 1 | 0.02 | 0.02 | 0.071 (0.00–0.22) | 0.0155 | yes |
| `snr0` | 1 | 0.05 | 0.05 | 0.216 (0.03–0.47) | 0.0466 | yes |
| `snr0` | 1 | 0.1 | 0.1 | 0.571 (0.18–0.98) | 0.0983 | yes |
| `snr0` | 1 | 0.15 | 0.15 | 0.822 (0.41–0.97) | 0.1448 | yes |
| `snr0` | 1 | 0.2 | 0.2 | 0.928 (0.87–0.99) | 0.1845 | yes |
| `snr0` | 1 | 0.3 | 0.3 | 0.781 (0.67–0.92) | 0.1172 | yes |
| `snr0` | 1 | 0.4 | 0.4 | 0.724 (0.63–0.85) | 0.1069 | yes |
| `snr0` | 10 | 0.05 | 0.5 | 0.000 (0.00–0.00) | 0.0000 | yes |
| `snr0` | 10 | 0.1 | 1.0 | 0.052 (0.05–0.05) | 0.0345 | yes |
| `snr0` | 10 | 0.15 | 1.5 | 0.328 (0.33–0.33) | 0.1034 | NO ⚠ |
| `snr0` | 10 | 0.2 | 2.0 | 0.655 (0.66–0.66) | 0.1552 | NO ⚠ |
| `snr0` | 10 | 0.3 | 3.0 | 0.621 (0.62–0.62) | 0.1121 | NO ⚠ |
| `snr0` | 10 | 0.4 | 4.0 | 0.560 (0.56–0.56) | 0.0862 | NO ⚠ |
| `snr_m5` | 1 | 0.01 | 0.01 | 0.041 (0.00–0.19) | 0.0052 | yes |
| `snr_m5` | 1 | 0.02 | 0.02 | 0.090 (0.00–0.19) | 0.0259 | yes |
| `snr_m5` | 1 | 0.05 | 0.05 | 0.181 (0.02–0.39) | 0.0517 | yes |
| `snr_m5` | 1 | 0.1 | 0.1 | 0.326 (0.04–0.72) | 0.0914 | yes |
| `snr_m5` | 1 | 0.15 | 0.15 | 0.516 (0.16–0.92) | 0.1517 | yes |
| `snr_m5` | 1 | 0.2 | 0.2 | 0.593 (0.22–0.89) | 0.1862 | yes |
| `snr_m5` | 1 | 0.3 | 0.3 | 0.800 (0.42–0.98) | 0.2345 | yes |
| `snr_m5` | 1 | 0.4 | 0.4 | 0.836 (0.63–0.97) | 0.2414 | yes |
| `snr_m5` | 10 | 0.05 | 0.5 | 0.009 (0.01–0.01) | 0.0000 | yes |
| `snr_m5` | 10 | 0.1 | 1.0 | 0.009 (0.01–0.01) | 0.0000 | yes |
| `snr_m5` | 10 | 0.15 | 1.5 | 0.026 (0.03–0.03) | 0.0172 | NO ⚠ |
| `snr_m5` | 10 | 0.2 | 2.0 | 0.095 (0.09–0.09) | 0.0776 | NO ⚠ |
| `snr_m5` | 10 | 0.3 | 3.0 | 0.474 (0.47–0.47) | 0.2586 | NO ⚠ |
| `snr_m5` | 10 | 0.4 | 4.0 | 0.612 (0.61–0.61) | 0.3190 | NO ⚠ |
| `snr_m10` | 1 | 0.01 | 0.01 | 0.045 (0.00–0.13) | 0.0103 | yes |
| `snr_m10` | 1 | 0.02 | 0.02 | 0.059 (0.00–0.16) | 0.0224 | yes |
| `snr_m10` | 1 | 0.05 | 0.05 | 0.138 (0.01–0.28) | 0.0466 | yes |
| `snr_m10` | 1 | 0.1 | 0.1 | 0.217 (0.03–0.39) | 0.0966 | yes |
| `snr_m10` | 1 | 0.15 | 0.15 | 0.269 (0.10–0.40) | 0.1362 | yes |
| `snr_m10` | 1 | 0.2 | 0.2 | 0.278 (0.10–0.44) | 0.1431 | yes |
| `snr_m10` | 1 | 0.3 | 0.3 | 0.659 (0.28–0.98) | 0.2966 | yes |
| `snr_m10` | 1 | 0.4 | 0.4 | 0.788 (0.53–0.97) | 0.3586 | yes |
| `snr_m10` | 10 | 0.05 | 0.5 | 0.000 (0.00–0.00) | 0.0000 | yes |
| `snr_m10` | 10 | 0.1 | 1.0 | 0.009 (0.01–0.01) | 0.0086 | yes |
| `snr_m10` | 10 | 0.15 | 1.5 | 0.026 (0.03–0.03) | 0.0172 | NO ⚠ |
| `snr_m10` | 10 | 0.2 | 2.0 | 0.043 (0.04–0.04) | 0.0345 | NO ⚠ |
| `snr_m10` | 10 | 0.3 | 3.0 | 0.319 (0.32–0.32) | 0.2845 | NO ⚠ |
| `snr_m10` | 10 | 0.4 | 4.0 | 0.552 (0.55–0.55) | 0.4741 | NO ⚠ |
| `mixed` | 1 | 0.01 | 0.01 | 0.052 (0.00–0.19) | 0.0052 | yes |
| `mixed` | 1 | 0.02 | 0.02 | 0.059 (0.01–0.19) | 0.0086 | yes |
| `mixed` | 1 | 0.05 | 0.05 | 0.100 (0.02–0.22) | 0.0345 | yes |
| `mixed` | 1 | 0.1 | 0.1 | 0.393 (0.09–0.75) | 0.0983 | yes |
| `mixed` | 1 | 0.15 | 0.15 | 0.509 (0.12–0.88) | 0.1224 | yes |
| `mixed` | 1 | 0.2 | 0.2 | 0.722 (0.22–0.94) | 0.1741 | yes |
| `mixed` | 1 | 0.3 | 0.3 | 0.895 (0.78–0.97) | 0.2259 | yes |
| `mixed` | 1 | 0.4 | 0.4 | 0.757 (0.63–0.97) | 0.1466 | yes |
| `mixed` | 10 | 0.05 | 0.5 | 0.000 (0.00–0.00) | 0.0000 | yes |
| `mixed` | 10 | 0.1 | 1.0 | 0.017 (0.02–0.02) | 0.0086 | yes |
| `mixed` | 10 | 0.15 | 1.5 | 0.043 (0.04–0.04) | 0.0172 | NO ⚠ |
| `mixed` | 10 | 0.2 | 2.0 | 0.216 (0.22–0.22) | 0.1207 | NO ⚠ |
| `mixed` | 10 | 0.3 | 3.0 | 0.741 (0.74–0.74) | 0.2845 | NO ⚠ |
| `mixed` | 10 | 0.4 | 4.0 | 0.595 (0.59–0.59) | 0.1724 | NO ⚠ |

## Q3 — the useful region

10 cell(s) attain release ≥ 50% at Σα ≤ 0.10 with realized wrong-release ≤ Σα:

| stratum | k | αᵢ | Σα | release | wrong-release |
|---|---|---|---|---|---|
| `tel` | 1 | 0.02 | 0.02 | 0.741 | 0.0190 |
| `tel` | 1 | 0.05 | 0.05 | 0.735 | 0.0207 |
| `tel` | 1 | 0.1 | 0.1 | 0.848 | 0.0379 |
| `tel` | 2 | 0.01 | 0.02 | 0.691 | 0.0000 |
| `tel` | 2 | 0.02 | 0.04 | 0.731 | 0.0034 |
| `tel` | 2 | 0.05 | 0.1 | 0.862 | 0.0069 |
| `tel` | 3 | 0.01 | 0.03 | 0.541 | 0.0069 |
| `tel` | 3 | 0.02 | 0.06 | 0.588 | 0.0103 |
| `tel` | 5 | 0.02 | 0.1 | 0.512 | 0.0034 |
| `snr0` | 1 | 0.1 | 0.1 | 0.571 | 0.0983 |

## Q4 — validity

**12 breach(es)** — see questions.json.

## Q5 — conformal vs a swept confidence threshold

Matched-release comparisons: 91; beyond ±20% relative: 68. Where both methods reach the same release band, they land at indistinguishable risk — the conformal gate's contribution is the distribution-free guarantee fixed in advance, not a sharper curve.


## §9 re-analyses — where the registered test was the wrong instrument

Both verdicts above stand as registered. These re-analyses say what the registered tests could not, and are reported beside them.

**Q4.** Of 850 cells at Σα ≤ 1, **12** exceed the marginal budget. The registered slack counted test-set noise only; the dominant term at n = 116 is calibration-draw variability. Against Vovk (2012)'s training-conditional band, **1** survive — within multiplicity expectation for this many cells. The measurable finding is the size of that band: the honest per-split guarantee is

| nominal α | training-conditional 95% | inflation |
|---|---|---|
| 0.01 | 0.0255 | ×2.55 |
| 0.02 | 0.0402 | ×2.01 |
| 0.05 | 0.0770 | ×1.54 |
| 0.1 | 0.1420 | ×1.42 |

so at the operating points where this method is useful (α = 0.01–0.05) the per-split guarantee is **1.5–2.5× the nominal α**. Bimodal LLM confidence makes this maximal: the quantile lands on a near-vertical segment of the score CDF, where a small split difference moves coverage directly.

**Q5.** The registered ±20% RELATIVE band is meaningless when both rates sit near the 0.0086 resolution limit (one session in 116). In absolute terms across 91 matched-release bands: conformal lower in **54**, confidence lower in **22**, tied in 15, median absolute gap **0.0517**. The frontiers do NOT coincide — the prediction was wrong, and wrong in the method's favour: per-item calibrated thresholds adapt to item difficulty in a way one global confidence cut cannot.

## Figures (§6)

- `fig1_release_vs_epsilon.png` — release vs ε̄, one line per k
- `fig2_sigma_vs_k.png` — required Σα vs k, one line per stratum
- `fig3_risk_coverage.png` — risk–coverage against both baselines

