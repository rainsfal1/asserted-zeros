"""What the original readout was worth: re-impose truncation on measured scores.

POST-HOC and NOT pre-registered. This is a controlled simulation of the readout
mechanism, not a reconstruction of the original pipeline: real top-k truncation
drops an option when it falls outside the decision token's window, which depends
on the full token distribution we no longer hold. Here we re-impose the same
EQUATION -- an option whose measured probability is below a floor tau is recorded
as 0.0 and the rest renormalized -- and sweep tau over the corrected corpus.

Everything else is held fixed: same records, same split, same calibration, same
decision rule. The only variable is how much of the tail the readout can see.

    ../../.venv/bin/python sweep_truncation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
POOL = REPO / "results" / "v2_pool"
TAGS = ("judge", "evaluator")
OPT = {"a": 1, "b": 0}
TAUS = (0.0, 1e-6, 1e-4, 1e-2, 0.1)
#: Swept jointly with tau, because the first run showed the two INTERACT: with
#: n=116 calibration rows the (1-alpha_i) quantile sits at the maximum score
#: whenever alpha_i is small, so the zeros truncation creates are simply never
#: reached. The collapse needs alpha_i above the non-zero fraction.
ALPHAS_I = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40)
K_ITEMS = 10

import sys  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from verdict_cert import uscis  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402
import importlib.util as ilu  # noqa: E402

_spec = ilu.spec_from_file_location("m6f", REPO / "experiments" / "m6_powergrading.py")
m6 = ilu.module_from_spec(_spec)
_spec.loader.exec_module(m6)

_v3 = ilu.spec_from_file_location("v3", REPO / "experiments" / "v3_frontier.py")
v3 = ilu.module_from_spec(_v3)
_v3.loader.exec_module(v3)


def truncate(p: dict[int, float], tau: float) -> dict[int, float]:
    """The original readout's equation: below the window, recorded as zero."""
    kept = {o: (v if v >= tau else 0.0) for o, v in p.items()}
    z = sum(kept.values())
    if z <= 0.0:                       # every option below the floor
        return dict(p)                 # nothing survives to renormalize
    return {o: v / z for o, v in kept.items()}


def main() -> int:
    recs = {json.loads(x)["id"]: json.loads(x)
            for x in (POOL / "records.jsonl").open() if x.strip()}
    ex = {t: {json.loads(x)["id"]: json.loads(x)["dist"]
              for x in (POOL / f"extract-{t}.jsonl").open() if x.strip()}
          for t in TAGS}
    students = sorted({r["student"] for r in recs.values()})
    _, cal, test = m6.split_students(students, 0)
    items = sorted({r["item"] for r in recs.values()})

    # one row per (student, item) at the hash-assigned level -- the v2 convention
    rows: dict[tuple, dict] = {}
    for r in recs.values():
        if r["level"] != v3.hash_level(r["student"], r["item"]):
            continue
        per_tag = [{OPT[o]: p for o, p in ex[t][r["id"]].items()} for t in TAGS]
        rows[(r["student"], r["item"])] = {"gold": r["gold"], "p": per_tag}

    def complete(pop):
        return [s for s in pop if all((s, it) in rows for it in items)]

    cal_ok, test_ok = complete(cal), complete(test)
    out = {"n_cal": len(cal_ok), "n_test": len(test_ok), "cells": [],
           "note": "post-hoc controlled simulation; not pre-registered"}
    grid = [(t, a) for t in TAUS for a in ALPHAS_I]

    for tau, a_i in grid:
        def score_min(cell):
            ps = [truncate(p, tau) for p in cell["p"]]
            return {o: min(1.0 - p[o] for p in ps) for o in (1, 0)}

        thr = {}
        zero_share = 0
        n_cal_rows = 0
        for it in items:
            xs = []
            for s in cal_ok:
                sc = score_min(rows[(s, it)])[rows[(s, it)]["gold"]]
                xs.append(sc)
                zero_share += (sc == 0.0)
                n_cal_rows += 1
            thr[it] = calibrate_threshold(xs, a_i)

        rel = wrong = 0
        for s in test_ok:
            sets, gold = {}, {}
            for it in items:
                sc = score_min(rows[(s, it)])
                keep = frozenset(o for o in (1, 0) if sc[o] <= thr[it])
                sets[it] = keep or frozenset({1, 0})
                gold[it] = rows[(s, it)]["gold"]
            cert = uscis.certify(sets)
            if cert.certified:
                rel += 1
                wrong += int(cert.verdict != uscis.verdict_of(gold))
        n = len(test_ok)
        finite = [t for t in thr.values() if not math.isinf(t)]
        out["cells"].append({
            "tau": tau, "alpha_i": a_i, "sigma_alpha": round(a_i * K_ITEMS, 3),
            "cal_zero_score_share": zero_share / n_cal_rows,
            "thr_median": sorted(finite)[len(finite) // 2] if finite else None,
            "release_rate": rel / n, "wrong_release_rate": wrong / n,
            "released": rel, "wrong": wrong, "n": n,
            "err_among_released": (wrong / rel) if rel else None,
        })
        c = out["cells"][-1]
        thr_s = "inf" if c["thr_median"] is None else f"{c['thr_median']:.4f}"
        print(f"  tau={tau:<7} a_i={a_i:<5} zeros={c['cal_zero_score_share']:.3f} "
              f"thr={thr_s:<9} release={rel / n:.3f} wrong/1000={1000 * wrong / n:.1f}")

    (OUT / "truncation_sweep.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT / 'truncation_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
