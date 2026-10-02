# Asserted Zeros

Code and data for **"Asserted Zeros: When a Valid Guarantee Certifies an
Unmeasured Quantity"**, Zayyan Ahmed. NeurIPS 2026 Workshop on Reliable Agent
Development (*Who Verifies the Agents?*), poster.

[Paper on OpenReview](https://openreview.net/forum?id=vAaj5fqMYR) · arXiv: *forthcoming*

A distribution-free guarantee certifies whatever score it is handed, including
one that was never measured. This repository holds the harnesses that produced
each result, the committed score caches every figure is computed from, and the
pre-registrations committed before their harnesses existed. Every number in the
paper can be recomputed from it on a CPU.

Licensed CC BY 4.0 (`LICENSE`). The three corpora the harnesses read are cited
but **not** redistributed; see `LICENSE` for their terms.

## Citation

```bibtex
@inproceedings{ahmed2026asserted,
  title     = {Asserted Zeros: When a Valid Guarantee Certifies an Unmeasured Quantity},
  author    = {Ahmed, Zayyan},
  booktitle = {NeurIPS 2026 Workshop on Reliable Agent Development},
  year      = {2026},
  note      = {Poster},
  url       = {https://openreview.net/forum?id=vAaj5fqMYR}
}
```

---

## Reproducing, in three commands

No GPU is needed. The score caches are already here, so every number and figure
in the paper is recomputed from committed data by CPU arithmetic.

```
pip install -r requirements-analysis.txt     # numpy + matplotlib, pinned
export PYTHONPATH=src

python paper/workshop/fig_scores.py               # Figure 1
python paper/workshop/sweep_split_tau_ladder.py   # Appendix F, the tau-ladder
python experiments/readout_v7_instance.py         # fresh-instance replication
```

`requirements-gpu.l20.lock` pins the serving stack instead, and is needed only
to regenerate the caches from audio — which is not required to check anything
below.

**These scripts overwrite their own outputs in place** (the `*.json` beside them,
and `figs/*.pdf`). That is how they were run originally, and the values are
reproduced identically, but if you want the shipped artifacts untouched, copy
the archive first.

Each figure script asserts the numbers it draws and exits non-zero if the
artifact disagrees: `fig_scores.py` checks the 90.6% share and the 0 corrected
zeros; `fig_stepfunction.py` checks ten items, 57 → 44 wrong verdicts, and 2/10
zero-release. `sweep_split_tau_ladder.py` and `readout_v7_instance.py` each run a
faithfulness guard first — they re-derive the committed JSON from the caches and
refuse to score anything new if they cannot reproduce it exactly.

---

## Where each result lives

| Paper | Claim | Artifact |
|---|---|---|
| §2 | 90.6% / 89.7% of graded records score exactly 0.0 | `results/s3/`, `results/m3/` — derivation readable in `paper/workshop/fig_scores.py` |
| §2 | which factor erased each zero (Table 1) | `paper/workshop/sweep_truncation.py` |
| §3 | corrected readout: 0 exact zeros on the same 660 records | `results/s3m3_measured/`, `experiments/s3m3_measured.py` |
| §3 | independent full-vocabulary reconstruction | `experiments/readout_v4_fullvocab.py`, `results/readout_validation/` |
| §4 | the pre-registered probe, all ten predictions scored | `experiments/per_option_probe.py`, `results/per_option_probe/` |
| §5 | per-item cost at the gate | `paper/workshop/sweep_per_item.py`, `results/v2_pool/` |
| §5 | the twenty-resplit stability treatment | `paper/workshop/sweep_split_stability.py` |
| App. C | the certification frontier | `experiments/v3_frontier.py`, `results/v3_frontier/` |
| App. F | the τ-ladder, including the writer's own 5e-6 | `paper/workshop/sweep_split_tau_ladder.py`, `split_stability_tau.json` |
| — | fresh-instance replication of §5's structure | `experiments/readout_v7_instance.py`, `results/v2_instance2/` |

**On `results/v2_instance2/`:** its `records.jsonl` is byte-identical to
`results/v2_pool/records.jsonl`, which is intended — the *stimuli* are held fixed
so that only the extraction varies. The `extract-*.jsonl` files differ; that
difference is the measurement. See `experiments/readout_v7_instance.py`.

---

## Pre-registrations

Committed before the harnesses they describe existed, and scored clause by
clause in §4 and Appendix C -- including the five predictions that failed and the
one that was substantively falsified. Reporting the failures is the evidence the
registrations were binding rather than decorative.

Each states its own commit date and what did not yet exist when it was written,
in its opening lines.

| File | Registers | Paper |
|---|---|---|
| `per-option-probe-preregistration.md` | the probe bounding what the defect distorted | §4 |
| `v2-pool-preregistration.md` | the corrected corpus, predicted before it existed | §3, §5 |
| `readout-validation-preregistration.md` | checking the fix against something other than itself | §3 |
| `v3-frontier-preregistration.md` | the certification frontier | App. C |

`docs/canonical-extraction.md` is the odd one out: a design contract for the
readout the repair implements, not a pre-registration. It carries a dated
amendment correcting one of its own claims, with the superseded sentence kept
verbatim.

---

## Layout

```
paper/workshop/   the paper, its figure and sweep scripts, and their artifacts
src/              the library: the decision rule, the certificate, extraction.
                  The scoring key is src/verdict_cert/scorer.py
experiments/      the harnesses behind each section
results/          committed score caches; every number is computed from these
items/            the instrument: item wording and the answer bank
docs/             the pre-registrations
```

---

## What is not here

This is a reproduction package for one paper, extracted from a larger research
codebase. That project's design documents, test suite and internal reviews are
not included; a few files refer to them by path, and those references will not
resolve. Nothing needed to check the paper depends on them.

The deployed screening service the scorer is transcribed from is described but
not named.
