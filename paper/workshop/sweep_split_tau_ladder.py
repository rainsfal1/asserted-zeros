"""Does section 5's structure survive at the defect's OWN resolution?

Section 5 runs its per-item measurement at tau = 1e-2. The writer that produced
the artifact rounded to five decimals, i.e. a floor at 5e-6 -- 2000x finer.
Appendix F already shows the single-split totals move at that scale (57 -> 59
rather than 57 -> 44, and no zero-release items), which inverts the aggregate
direction section 5 reports. What has never been run is the TWENTY-RESPLIT
treatment at 5e-6, so the three claims section 5 says survive resampling have
only ever been tested at the coarse floor.

This runs sweep_split_stability.py's measurement across the whole tau ladder,
changing nothing else -- same alpha, pass mark, decision rule, splits, seeds.

Faithfulness guard: the tau = 1e-2 rung must reproduce the committed
split_stability.json before any other rung is reported. A reimplementation that
cannot reproduce the committed artifact is not allowed to judge a new one.

Post-hoc, like the rest of the section 5 simulation. Reads committed artifacts;
no GPU, no inference.
"""

from __future__ import annotations

import importlib.util as ilu
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from verdict_cert import uscis  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402

_s = ilu.spec_from_file_location("sw", HERE / "sweep_honest_budget.py")
sw = ilu.module_from_spec(_s)
_s.loader.exec_module(sw)

ALPHA, PASS_MARK, B = 0.10, 1, 20
TAUS = (1e-6, 5e-6, 1e-4, 1e-2)          # 5e-6 is the writer's own resolution


def one_split(rows, items, seed, tau):
    """sweep_split_stability.one_split, with tau as an argument."""
    students = sorted({s for (s, _i) in rows})
    _, cal, test = sw.v3.m6.split_students(students, seed)
    ok = [s for s in cal if all((s, i) in rows for i in items)]
    te = [s for s in test if all((s, i) in rows for i in items)]

    def score(cell, t):
        ps = [sw.truncate(p, t) for p in cell["p"]]
        return {o: min(1.0 - p[o] for p in ps) for o in (1, 0)}

    per = {}
    for it in items:
        rec = {"cal_error": sum(
            1 for s in ok if score(rows[(s, it)], tau)[rows[(s, it)]["gold"]] > 0.5
        ) / len(ok)}
        for tag, t in (("measured", 0.0), ("truncated", tau)):
            thr = calibrate_threshold(
                [score(rows[(s, it)], t)[rows[(s, it)]["gold"]] for s in ok], ALPHA)
            rel = wrong = 0
            for s in te:
                v = score(rows[(s, it)], t)
                keep = frozenset(o for o in (1, 0) if v[o] <= thr)
                c = uscis.certify({it: keep or frozenset({1, 0})}, pass_mark=PASS_MARK)
                if c.certified:
                    rel += 1
                    wrong += int(c.verdict != uscis.verdict_of(
                        {it: rows[(s, it)]["gold"]}, pass_mark=PASS_MARK))
            rec[tag] = {"threshold": None if thr == float("inf") else thr,
                        "released": rel, "wrong": wrong}
        rec["inert"] = (rec["measured"]["threshold"] == rec["truncated"]["threshold"]
                        and rec["measured"]["released"] == rec["truncated"]["released"])
        per[it] = rec
    return {"seed": seed, "n_cal": len(ok), "n_test": len(te), "items": per}


def rung(rows, items, tau):
    splits = [one_split(rows, items, b, tau) for b in range(B)]

    def agg(f):
        v = [f(sp) for sp in splits]
        return {"min": min(v), "median": st.median(v), "max": max(v), "values": v}

    endpoints = reached = 0
    direction = {"rise": 0, "tie": 0, "fall": 0}
    zero_rel_splits = extra_wrong_splits = 0
    for sp in splits:
        limit = 10.0 / sp["n_cal"]
        if any(r["truncated"]["released"] == 0 for r in sp["items"].values()):
            zero_rel_splits += 1
        below = [r for r in sp["items"].values()
                 if not r["inert"] and r["cal_error"] <= limit]
        if any(r["truncated"]["wrong"] > r["measured"]["wrong"] for r in below):
            extra_wrong_splits += 1
        for r in sp["items"].values():
            if r["inert"]:
                continue
            reached += 1
            t = r["truncated"]["threshold"]
            endpoints += int(t is None or t in (0.0, 1.0) or t >= 0.999999 or t <= 1e-12)
            if r["cal_error"] <= limit:
                m, tr = r["measured"]["released"], r["truncated"]["released"]
                direction["rise" if tr > m else ("tie" if tr == m else "fall")] += 1

    return {
        "tau": tau, "B": B, "alpha": ALPHA, "pass_mark": PASS_MARK,
        "wrong_measured": agg(lambda s: sum(r["measured"]["wrong"] for r in s["items"].values())),
        "wrong_truncated": agg(lambda s: sum(r["truncated"]["wrong"] for r in s["items"].values())),
        "n_items_zero_release": agg(
            lambda s: sum(1 for r in s["items"].values() if r["truncated"]["released"] == 0)),
        "n_items_inert": agg(lambda s: sum(1 for r in s["items"].values() if r["inert"])),
        "truncated_worse_on_aggregate": sum(
            1 for s in splits
            if sum(r["truncated"]["wrong"] for r in s["items"].values())
            > sum(r["measured"]["wrong"] for r in s["items"].values())),
        # the three claims section 5 says survive resampling
        "endpoint_thresholds": {"reached": reached, "endpoint": endpoints},
        "splits_with_a_zero_release_item": zero_rel_splits,
        "splits_with_an_extra_wrong_below_step": extra_wrong_splits,
        "below_step_release_direction": direction,
    }


def main() -> int:
    rows, items = sw.load_per_extractor()

    # --- faithfulness guard -------------------------------------------------
    committed = json.loads((HERE / "split_stability.json").read_text())
    base = rung(rows, items, 1e-2)
    checks = [
        ("wrong_measured.median", base["wrong_measured"]["median"], committed["wrong_measured"]["median"]),
        ("wrong_truncated.median", base["wrong_truncated"]["median"], committed["wrong_truncated"]["median"]),
        ("endpoint reached", base["endpoint_thresholds"]["reached"], committed["endpoint_thresholds"]["reached"]),
        ("endpoint count", base["endpoint_thresholds"]["endpoint"], committed["endpoint_thresholds"]["endpoint"]),
        ("direction", base["below_step_release_direction"], committed["below_step_release_direction"]),
        ("aggregate worse", base["truncated_worse_on_aggregate"], committed["truncated_worse_on_aggregate"]),
    ]
    bad = [(n, g, w) for n, g, w in checks if g != w]
    if bad:
        print("FAITHFULNESS GUARD FAILED -- refusing to report other rungs:")
        for n, g, w in bad:
            print(f"   {n}: got {g!r}, committed {w!r}")
        return 1
    print(f"faithfulness guard: tau=1e-2 rung reproduces split_stability.json "
          f"({len(checks)} invariants)\n")

    out = {"note": "post-hoc; only tau varies. alpha, splits, seeds, rule fixed.",
           "rungs": {}}
    hdr = (f"{'tau':>9}{'wrong meas':>12}{'wrong trunc':>13}{'agg worse':>11}"
           f"{'endpoint':>11}{'zero-rel':>10}{'extra-wrong':>13}{'rise/tie/fall':>15}")
    print(hdr)
    print("-" * len(hdr))
    for tau in TAUS:
        r = base if tau == 1e-2 else rung(rows, items, tau)
        out["rungs"][f"{tau:.0e}"] = r
        d = r["below_step_release_direction"]
        print(f"{tau:>9.0e}"
              f"{r['wrong_measured']['median']:>12}"
              f"{r['wrong_truncated']['median']:>13}"
              f"{r['truncated_worse_on_aggregate']:>9}/{B}"
              f"{r['endpoint_thresholds']['endpoint']:>7}/{r['endpoint_thresholds']['reached']}"
              f"{r['splits_with_a_zero_release_item']:>8}/{B}"
              f"{r['splits_with_an_extra_wrong_below_step']:>11}/{B}"
              f"{d['rise']:>7}/{d['tie']}/{d['fall']}")
    (HERE / "split_stability_tau.json").write_text(json.dumps(out, indent=1))
    print("\nwrote split_stability_tau.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
