# v2 pool — REPORT

**Scores the §4/§6/§7/§8-AMENDED predictions** of `docs/v2-pool-preregistration.md` (§6.1 + §7.1 operating points; §8.2's D2 selection; split seed 0) — not superseded text.

Floors (measured): plain 0.084818, spend-k1 0.169636 — §6.1's disclosure, confirmed.
Calibration zero-score share: 0.0 (v1 was ~0.90 — F-11's fix, measured).

## Baselines without the certificate (§4)

- Test sessions: **n = 116** (every pooled rate below has this n per seed, 580 session-runs pooled over 5 seeds).
- Always answer (point estimate, no abstention): **44/116 wrong** (379.3 per 1000).
- Always release PASS: **116/116 wrong** (1000.0 per 1000).

## P1: FALSIFIED
```json
{
 "0.1": {
  "seed_wins": 0,
  "naive": [
   null,
   null,
   null,
   null,
   null
  ],
  "none": [
   null,
   null,
   null,
   null,
   null
  ]
 },
 "0.15": {
  "seed_wins": 0,
  "naive": [
   null,
   null,
   null,
   null,
   null
  ],
  "none": [
   null,
   null,
   null,
   null,
   null
  ]
 }
}
```

## P2: FALSIFIED
```json
{
 "0.2": {
  "repeat": {
   "spend": 0,
   "nospend": 0
  },
  "rephrase": {
   "spend": 0,
   "nospend": 0
  },
  "ladder": {
   "spend": 0,
   "nospend": 0
  }
 },
 "0.3": {
  "repeat": {
   "spend": 0,
   "nospend": 0
  },
  "rephrase": {
   "spend": 0,
   "nospend": 0
  },
  "ladder": {
   "spend": 0,
   "nospend": 0
  }
 },
 "0.5": {
  "repeat": {
   "spend": 0,
   "nospend": 0
  },
  "rephrase": {
   "spend": 0,
   "nospend": 0
  },
  "ladder": {
   "spend": 0,
   "nospend": 0
  }
 },
 "never_more_sweep": {}
}
```

## P3: FALSIFIED
*Scored on the `|none` arms (builder-only contrast, the clean analogue of B1's A/B). Policy `none` consumes no rng, so the five session seeds are byte-identical there and the clause's "in any/every seed" is a single trial, not five.*

```json
{
 "legacy_feasible_seeds": 0,
 "conformal_feasible_seeds": 0
}
```

## P4: HELD
```json
{
 "zero_score_share": 0.0,
 "thresholds_positive": true
}
```

## P5: FALSIFIED
```json
{
 "0.2": {
  "seed_wins": 0,
  "d2_risk": null,
  "d1_risk": null
 },
 "0.3": {
  "seed_wins": 0,
  "d2_risk": null,
  "d1_risk": null
 },
 "matched_release": {
  "0.0": {
   "status": "attained",
   "d1_wrong_per_1000": 0.0,
   "d2_wrong_per_1000": 0.0,
   "d2_no_higher": true
  },
  "0.05": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.1": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.15": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.2": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.25": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.3": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.35": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.4": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.45": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.5": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.55": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.6": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.65": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.7": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.75": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.8": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.85": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.9": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 0
  },
  "0.95": {
   "status": "absent",
   "d1_points": 0,
   "d2_points": 1
  }
 },
 "n_targets_attained": 1,
 "matched_ok": true,
 "wins_ok": false
}
```

## P6: FALSIFIED
```json
{
 "mean_abs_diff": 0.0,
 "within_arm_spread": 0,
 "exhaustion_rephrase": {
  "fired": 0,
  "violations": 0,
  "by_item": {}
 },
 "confound_breakdown": {
  "repeat": {
   "level_transitions": {},
   "same_voice": 0,
   "n_rounds": 0
  },
  "rephrase": {
   "level_transitions": {},
   "same_voice": 0,
   "n_rounds": 0
  }
 }
}
```

## Threshold reproducibility, 20 student-disjoint splits (§4, P4)

The training-conditional dispersion v1 could only note as a standing negative, measured here on a real dial. `no finite t` counts splits where the item's calibration stream cannot reach the level at all.

| Σα | item | seed-0 t | min | max | splits with no finite t |
|---|---|---|---|---|---|
| 0.1 | 1 | 0.9999987731869059 | 0.9999981304077065 | 0.9999989361434368 | 0/20 |
| 0.1 | 13 | 0.9999958869892864 | 0.9999937235756122 | 0.9999985026397246 | 0/20 |
| 0.1 | 2 | 0.9999963051693918 | 0.9999940965901797 | 0.9999963051693918 | 0/20 |
| 0.1 | 20 | 0.9999993231091846 | 0.9999988391202898 | 0.99999942636387 | 0/20 |
| 0.1 | 3 | 0.9999991732151323 | 0.9999987593173882 | 0.9999991732151323 | 0/20 |
| 0.1 | 4 | 0.9999991382410165 | 0.9999967149773021 | 0.9999991382410165 | 0/20 |
| 0.1 | 5 | 0.9999991525341646 | 0.9999991246750956 | 0.9999991525341646 | 0/20 |
| 0.1 | 6 | 0.9999979243468743 | 0.9999974538049287 | 0.999998544767185 | 0/20 |
| 0.1 | 7 | 0.9999969500078238 | 0.9999960340522112 | 0.9999983438488687 | 0/20 |
| 0.1 | 8 | 0.9999983092700725 | 0.999994033216522 | 0.9999983092700725 | 0/20 |
| 0.2 | 1 | 0.9999979180826993 | 0.9999978158634829 | 0.9999989295746974 | 0/20 |
| 0.2 | 13 | 0.9999952750641108 | 0.9999881484725474 | 0.9999958869892864 | 0/20 |
| 0.2 | 2 | 0.9999962480405175 | 0.9999906587351265 | 0.9999962480405175 | 0/20 |
| 0.2 | 20 | 0.9999988522299633 | 0.9999986841298893 | 0.9999993231091846 | 0/20 |
| 0.2 | 3 | 0.9999991692770716 | 0.9999976964962897 | 0.9999991692770716 | 0/20 |
| 0.2 | 4 | 0.9999988944596209 | 0.9999963219139026 | 0.9999988944596209 | 0/20 |
| 0.2 | 5 | 0.9999991251108153 | 0.9999991246750956 | 0.9999991525341646 | 0/20 |
| 0.2 | 6 | 0.9999974538049287 | 0.9999974538049287 | 0.9999979243468743 | 0/20 |
| 0.2 | 7 | 0.9999957652058696 | 0.9999950143625906 | 0.9999983438488687 | 0/20 |
| 0.2 | 8 | 0.9999977425784922 | 0.999994033216522 | 0.9999979742446768 | 0/20 |

## Score histogram and the off-option residual (§4)

```json
{
 "score_histogram": {
  "[0.0,1e-09)": 0,
  "[1e-09,0.05)": 873,
  "[0.05,0.1)": 5,
  "[0.1,0.2)": 5,
  "[0.2,0.4)": 3,
  "[0.4,0.6)": 2,
  "[0.6,0.8)": 5,
  "[0.8,0.95)": 8,
  "[0.95,1.0001)": 268
 },
 "mass_unnormalized_quantiles": {
  "p5": 0.9969,
  "p25": 0.9997,
  "p50": 1.0,
  "p75": 1.0,
  "p95": 1.0
 }
}
```

## Frontier — every arm, every Σα (§4: reported regardless)

Release and wrong-per-1000 are pooled over seeds; the seed range shows dispersion. `wrong/1000` counts wrong releases per 1000 SESSIONS (P(release ∧ wrong)), not per release.

| Σα | arm | n (pooled) | release | seed range | wrong/1000 | wrong among released | re-asks |
|---|---|---|---|---|---|---|---|
| 0.02 | `d1_spend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `d1_spend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `d1_spend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `d1_spend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `legacy_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `legacy_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `legacy_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `legacy_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `plain_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `plain_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `plain_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.02 | `plain_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `d1_spend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `d1_spend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `d1_spend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `d1_spend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `legacy_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `legacy_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `legacy_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `legacy_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `plain_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `plain_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `plain_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.05 | `plain_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `d1_spend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `d1_spend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `d1_spend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `d1_spend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `legacy_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `legacy_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `legacy_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `legacy_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `plain_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `plain_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `plain_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.1 | `plain_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `d1_spend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `d1_spend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `d1_spend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `d1_spend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `legacy_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `legacy_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `legacy_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `legacy_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `plain_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `plain_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `plain_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.15 | `plain_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d1_spend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d1_spend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d1_spend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d1_spend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d2_ltt|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d2_ltt|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d2_ltt|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `d2_ltt|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `legacy_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `legacy_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `legacy_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `legacy_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `plain_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `plain_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `plain_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.2 | `plain_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d1_spend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d1_spend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d1_spend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d1_spend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d2_ltt|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d2_ltt|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d2_ltt|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `d2_ltt|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `legacy_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `legacy_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `legacy_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `legacy_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `plain_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `plain_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `plain_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.3 | `plain_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.5 | `d1_spend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.5 | `d1_spend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.5 | `d1_spend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.5 | `d1_spend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.5 | `d2_ltt|ladder` | 580 | 1.000 | 1.000–1.000 | 363.8 | 0.3638 | 51 |
| 0.5 | `d2_ltt|none` | 580 | 0.957 | 0.957–0.957 | 362.1 | 0.3784 | 0 |
| 0.5 | `d2_ltt|repeat` | 580 | 1.000 | 1.000–1.000 | 370.7 | 0.3707 | 55 |
| 0.5 | `d2_ltt|rephrase` | 580 | 1.000 | 1.000–1.000 | 370.7 | 0.3707 | 41 |
| 0.5 | `legacy_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 15 |
| 0.5 | `legacy_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.5 | `legacy_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 15 |
| 0.5 | `legacy_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 15 |
| 0.5 | `plain_nospend|ladder` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 15 |
| 0.5 | `plain_nospend|none` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 0 |
| 0.5 | `plain_nospend|repeat` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 15 |
| 0.5 | `plain_nospend|rephrase` | 580 | 0.000 | 0.000–0.000 | 0.0 | — | 15 |

## D2 (Learn-then-Test) selection, per policy (§6.2 + §8.2)

`t` is the most-releasing passer (§8.2); `passers` is how many of the 14 candidates cleared Bonferroni at all. `t = None` means D2 is infeasible at that Σα — §8.1's feasibility floor, not a bug.

| Σα | policy | t | passers | cal release at t | n_cal |
|---|---|---|---|---|---|
| 0.02 | none | None | 0 | — | 116 |
| 0.02 | repeat | None | 0 | — | 116 |
| 0.02 | rephrase | None | 0 | — | 116 |
| 0.02 | ladder | None | 0 | — | 116 |
| 0.05 | none | None | 0 | — | 116 |
| 0.05 | repeat | None | 0 | — | 116 |
| 0.05 | rephrase | None | 0 | — | 116 |
| 0.05 | ladder | None | 0 | — | 116 |
| 0.1 | none | None | 0 | — | 116 |
| 0.1 | repeat | None | 0 | — | 116 |
| 0.1 | rephrase | None | 0 | — | 116 |
| 0.1 | ladder | None | 0 | — | 116 |
| 0.15 | none | None | 0 | — | 116 |
| 0.15 | repeat | None | 0 | — | 116 |
| 0.15 | rephrase | None | 0 | — | 116 |
| 0.15 | ladder | None | 0 | — | 116 |
| 0.2 | none | 0.0 | 2 | 0.000 | 116 |
| 0.2 | repeat | 0.0 | 2 | 0.000 | 116 |
| 0.2 | rephrase | 0.0 | 2 | 0.000 | 116 |
| 0.2 | ladder | 0.0 | 2 | 0.000 | 116 |
| 0.3 | none | 0.0 | 2 | 0.000 | 116 |
| 0.3 | repeat | 0.0 | 2 | 0.000 | 116 |
| 0.3 | rephrase | 0.0 | 2 | 0.000 | 116 |
| 0.3 | ladder | 0.0 | 2 | 0.000 | 116 |
| 0.5 | none | 0.02 | 14 | 0.966 | 116 |
| 0.5 | repeat | 0.005 | 14 | 0.991 | 116 |
| 0.5 | rephrase | 0.005 | 14 | 1.000 | 116 |
| 0.5 | ladder | 0.005 | 14 | 1.000 | 116 |

## §6.5 confounds and exhaustion, measured

`repeat` moves the channel LEVEL; `rephrase` moves the VOICE with probability 7/8. Both are measured here rather than assumed away.

| Σα | arm | exhausted | spurious | same-voice re-asks | level transitions |
|---|---|---|---|---|---|
| 0.02 | `d1_spend|ladder` | 0 | 0 | 0/0 | — |
| 0.02 | `d1_spend|repeat` | 0 | 0 | 0/0 | — |
| 0.02 | `d1_spend|rephrase` | 0 | 0 | 0/0 | — |
| 0.05 | `d1_spend|ladder` | 0 | 0 | 0/0 | — |
| 0.05 | `d1_spend|repeat` | 0 | 0 | 0/0 | — |
| 0.05 | `d1_spend|rephrase` | 0 | 0 | 0/0 | — |
| 0.1 | `d1_spend|ladder` | 0 | 0 | 0/0 | — |
| 0.1 | `d1_spend|repeat` | 0 | 0 | 0/0 | — |
| 0.1 | `d1_spend|rephrase` | 0 | 0 | 0/0 | — |
| 0.15 | `d1_spend|ladder` | 0 | 0 | 0/0 | — |
| 0.15 | `d1_spend|repeat` | 0 | 0 | 0/0 | — |
| 0.15 | `d1_spend|rephrase` | 0 | 0 | 0/0 | — |
| 0.2 | `d1_spend|ladder` | 0 | 0 | 0/0 | — |
| 0.2 | `d1_spend|repeat` | 0 | 0 | 0/0 | — |
| 0.2 | `d1_spend|rephrase` | 0 | 0 | 0/0 | — |
| 0.3 | `d1_spend|ladder` | 0 | 0 | 0/0 | — |
| 0.3 | `d1_spend|repeat` | 0 | 0 | 0/0 | — |
| 0.3 | `d1_spend|rephrase` | 0 | 0 | 0/0 | — |
| 0.5 | `d1_spend|ladder` | 0 | 0 | 0/0 | — |
| 0.5 | `d1_spend|repeat` | 0 | 0 | 0/0 | — |
| 0.5 | `d1_spend|rephrase` | 0 | 0 | 0/0 | — |

## Wording supply per (item, gold) in the test split (§6.5 arithmetic)

```json
{
 "1|1": 5,
 "13|1": 47,
 "2|1": 17,
 "20|1": 62,
 "3|1": 83,
 "4|1": 26,
 "6|1": 29,
 "5|1": 22,
 "7|1": 68,
 "8|0": 21,
 "4|0": 23,
 "6|0": 12,
 "3|0": 21,
 "13|0": 25,
 "8|1": 7,
 "2|0": 13,
 "5|0": 8,
 "1|0": 6,
 "7|0": 8,
 "20|0": 2
}
```

