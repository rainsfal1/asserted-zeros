"""Does the readout defect change an ISSUED certificate at an honest budget?

The §3.1 sweep ran at k=10 on the mixed stream -- the one configuration in the
whole grid where nothing releases at any budget, so it could only ever show the
artifact failing to reach a gate that was inert anyway. The committed frontier
shows the CLEAN stratum releasing 51-86% at Sigma-alpha <= 0.10 with realized
risk inside budget. This runs the same truncation simulation there, where there
is something for the artifact to corrupt.

POST-HOC, not pre-registered, same controlled-simulation caveat as §3.1: a
probability floor is not a rank cut.

    ../../.venv/bin/python sweep_honest_budget.py
"""

from __future__ import annotations

import importlib.util as ilu
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from verdict_cert import uscis  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402

_v3 = ilu.spec_from_file_location("v3", REPO / "experiments" / "v3_frontier.py")
v3 = ilu.module_from_spec(_v3)
_v3.loader.exec_module(v3)

STRATUM = "tel"                       # the clean channel: it releases
KS = (1, 2, 3, 5)
TAUS = (0.0, 1e-6, 1e-4, 1e-2)
#: alpha_i values whose Sigma = k*alpha_i stays <= 0.10 -- an honest budget
ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.10)


def truncate(p, tau):
    kept = {o: (v if v >= tau else 0.0) for o, v in p.items()}
    z = sum(kept.values())
    return dict(p) if z <= 0.0 else {o: v / z for o, v in kept.items()}


POOL = REPO / "results" / "v2_pool"
TAGS = ("judge", "evaluator")
OPT = {"a": 1, "b": 0}


def load_per_extractor():
    """(student, item) -> per-extractor probability dicts, at STRATUM's level.

    v3.load() keeps only the min-combined score, which cannot be truncated;
    truncation acts on each extractor's distribution before combination.
    """
    recs = [json.loads(x) for x in (POOL / "records.jsonl").open() if x.strip()]
    ex = {t: {json.loads(x)["id"]: json.loads(x)["dist"]
              for x in (POOL / f"extract-{t}.jsonl").open() if x.strip()}
          for t in TAGS}
    rows, items = {}, sorted({r["item"] for r in recs})
    for r in recs:
        if r["level"] != STRATUM:
            continue
        rows[(r["student"], r["item"])] = {
            "gold": r["gold"],
            "p": [{OPT[o]: v for o, v in ex[t][r["id"]].items()} for t in TAGS]}
    return rows, items


def main() -> int:
    all_rows, items = load_per_extractor()
    students = sorted({s for (s, _i) in all_rows})
    _, cal, test = v3.m6.split_students(students, 0)
    complete = lambda pop: [s for s in pop if all((s, it) in all_rows for it in items)]  # noqa: E731
    cal_ok, test_ok = complete(cal), complete(test)
    cal_rows = test_rows = all_rows
    print(f"stratum={STRATUM} n_cal={len(cal_ok)} n_test={len(test_ok)}")

    out = {"stratum": STRATUM, "n_cal": len(cal_ok), "n_test": len(test_ok),
           "note": "post-hoc controlled simulation; not pre-registered", "cells": []}

    for k in KS:
        pm = v3.pass_mark_for(k)
        for subset in v3.subsets_for(k, items):
            for a_i in ALPHAS:
                sigma = round(k * a_i, 4)
                if sigma > 0.10 + 1e-9:
                    continue
                for tau in TAUS:
                    def sc(cell, tau=tau):
                        ps = [truncate(p, tau) for p in cell["p"]]
                        return {o: min(1.0 - p[o] for p in ps) for o in (1, 0)}

                    thr = {it: calibrate_threshold(
                        [sc(cal_rows[(s, it)])[cal_rows[(s, it)]["gold"]]
                         for s in cal_ok], a_i) for it in subset}
                    released = set()
                    wrong = 0
                    for s in test_ok:
                        sets, gold = {}, {}
                        for it in subset:
                            v = sc(test_rows[(s, it)])
                            keep = frozenset(o for o in (1, 0) if v[o] <= thr[it])
                            sets[it] = keep or frozenset({1, 0})
                            gold[it] = test_rows[(s, it)]["gold"]
                        c = uscis.certify(sets, pass_mark=pm)
                        if c.certified:
                            released.add(s)
                            wrong += int(c.verdict != uscis.verdict_of(gold, pass_mark=pm))
                    out["cells"].append({
                        "k": k, "pass_mark": pm, "items": list(subset),
                        "alpha_i": a_i, "sigma_alpha": sigma, "tau": tau,
                        "released": sorted(released), "n_released": len(released),
                        "wrong": wrong, "n": len(test_ok)})

    # the comparison that matters: same cell, tau=0 vs tau>0, session by session
    by = {}
    for c in out["cells"]:
        by[(c["k"], tuple(c["items"]), c["alpha_i"], c["tau"])] = c
    diffs = []
    for (k, its, a, tau), c in by.items():
        if tau == 0.0:
            continue
        base = by[(k, its, a, 0.0)]
        b, t = set(base["released"]), set(c["released"])
        if b != t:
            diffs.append({"k": k, "alpha_i": a, "sigma_alpha": c["sigma_alpha"],
                          "tau": tau, "items": list(its),
                          "released_measured": len(b), "released_truncated": len(t),
                          "gained": len(t - b), "lost": len(b - t),
                          "wrong_measured": base["wrong"], "wrong_truncated": c["wrong"]})
    out["differing_cells"] = diffs
    (HERE / "honest_budget_sweep.json").write_text(json.dumps(out, indent=1))

    print(f"\ncells: {len(out['cells'])}  |  cells where truncation CHANGES "
          f"the released set: {len(diffs)}")
    for d in sorted(diffs, key=lambda x: -abs(x["gained"] - x["lost"]))[:12]:
        print(f"  k={d['k']} Σα={d['sigma_alpha']:<6} τ={d['tau']:<7} "
              f"released {d['released_measured']}→{d['released_truncated']} "
              f"(+{d['gained']}/−{d['lost']})  wrong {d['wrong_measured']}→{d['wrong_truncated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
