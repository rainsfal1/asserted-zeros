"""F1: is the step function a property of the data, or of one split?

Section 5 reports release and wrong-verdict counts from a single speaker-level
split at n_cal = 116. Section 6 says, citing Vovk (2012), that per-split
miscoverage at that n runs 1.4-2.5x nominal -- so a reviewer is right to ask how
much of the reported structure survives resampling. This repeats the whole
per-item measurement over B speaker-disjoint splits and reports the spread.

Same everything else as sweep_per_item.py (alpha, tau, pass mark, decision rule);
the ONLY thing that varies is the split seed. Post-hoc, not pre-registered.
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

ALPHA, TAU, PASS_MARK, B = 0.10, 1e-2, 1, 20


def one_split(rows, items, seed):
    students = sorted({s for (s, _i) in rows})
    _, cal, test = sw.v3.m6.split_students(students, seed)
    ok = [s for s in cal if all((s, i) in rows for i in items)]
    te = [s for s in test if all((s, i) in rows for i in items)]

    def score(cell, tau):
        ps = [sw.truncate(p, tau) for p in cell["p"]]
        return {o: min(1.0 - p[o] for p in ps) for o in (1, 0)}

    per = {}
    for it in items:
        rec = {"cal_error": sum(
            1 for s in ok if score(rows[(s, it)], TAU)[rows[(s, it)]["gold"]] > 0.5
        ) / len(ok)}
        for tag, tau in (("measured", 0.0), ("truncated", TAU)):
            thr = calibrate_threshold(
                [score(rows[(s, it)], tau)[rows[(s, it)]["gold"]] for s in ok], ALPHA)
            rel = wrong = 0
            for s in te:
                v = score(rows[(s, it)], tau)
                keep = frozenset(o for o in (1, 0) if v[o] <= thr)
                c = uscis.certify({it: keep or frozenset({1, 0})}, pass_mark=PASS_MARK)
                if c.certified:
                    rel += 1
                    wrong += int(c.verdict != uscis.verdict_of(
                        {it: rows[(s, it)]["gold"]}, pass_mark=PASS_MARK))
            rec[tag] = {"threshold": None if thr == float("inf") else thr,
                        "released": rel, "wrong": wrong}
        # did truncation touch this item at all, on THIS split?
        rec["inert"] = (rec["measured"]["threshold"] == rec["truncated"]["threshold"]
                        and rec["measured"]["released"] == rec["truncated"]["released"])
        per[it] = rec
    return {"seed": seed, "n_cal": len(ok), "n_test": len(te), "items": per}


def main() -> int:
    rows, items = sw.load_per_extractor()
    splits = [one_split(rows, items, b) for b in range(B)]

    def agg(f):
        v = [f(sp) for sp in splits]
        return {"min": min(v), "median": st.median(v), "max": max(v), "values": v}

    summary = {
        "B": B, "alpha": ALPHA, "tau": TAU, "pass_mark": PASS_MARK,
        "note": "post-hoc; only the split seed varies",
        "wrong_measured": agg(lambda s: sum(r["measured"]["wrong"] for r in s["items"].values())),
        "wrong_truncated": agg(lambda s: sum(r["truncated"]["wrong"] for r in s["items"].values())),
        "n_items_zero_release": agg(
            lambda s: sum(1 for r in s["items"].values() if r["truncated"]["released"] == 0)),
        "n_items_inert": agg(lambda s: sum(1 for r in s["items"].values() if r["inert"])),
        "truncated_worse_on_aggregate": sum(
            1 for s in splits
            if sum(r["truncated"]["wrong"] for r in s["items"].values())
            > sum(r["measured"]["wrong"] for r in s["items"].values())),
        "per_item": {},
    }
    for it in items:
        summary["per_item"][it] = {
            "cal_error": agg(lambda s, it=it: s["items"][it]["cal_error"]),
            "zero_release_splits": sum(
                1 for s in splits if s["items"][it]["truncated"]["released"] == 0),
            "inert_splits": sum(1 for s in splits if s["items"][it]["inert"]),
            "extra_wrong": agg(lambda s, it=it: s["items"][it]["truncated"]["wrong"]
                               - s["items"][it]["measured"]["wrong"]),
        }
    # Persist the two claims the paper prints that this file previously did NOT
    # carry -- a reviewer had to re-run the sweep to check them (final review):
    # every reached threshold is an endpoint, and the below-step release direction.
    endpoints = reached = 0
    direction = {"rise": 0, "tie": 0, "fall": 0}
    for sp in splits:
        limit = 10.0 / sp["n_cal"]
        for r in sp["items"].values():
            if r["inert"]:
                continue
            reached += 1
            t = r["truncated"]["threshold"]
            endpoints += int(t is None or t in (0.0, 1.0) or t >= 0.999999 or t <= 1e-12)
            if r["cal_error"] <= limit:
                m, tr = r["measured"]["released"], r["truncated"]["released"]
                direction["rise" if tr > m else ("tie" if tr == m else "fall")] += 1
    summary["endpoint_thresholds"] = {"reached": reached, "endpoint": endpoints}
    summary["below_step_release_direction"] = direction
    (HERE / "split_stability.json").write_text(json.dumps(summary, indent=1))

    print(f"B={B} splits, alpha={ALPHA}, tau={TAU}\n")
    for k in ("wrong_measured", "wrong_truncated", "n_items_zero_release", "n_items_inert"):
        a = summary[k]
        print(f"  {k:<24} median {a['median']:>6}   range [{a['min']}, {a['max']}]")
    print(f"\n  splits where truncation raises aggregate wrong verdicts: "
          f"{summary['truncated_worse_on_aggregate']}/{B}")
    print(f"\n{'item':<5}{'cal err (med)':>15}{'zero-release':>14}{'inert':>8}{'extra wrong (med)':>19}")
    for it, r in sorted(summary["per_item"].items(), key=lambda kv: kv[1]["cal_error"]["median"]):
        print(f"{it:<5}{r['cal_error']['median']:>15.3f}{r['zero_release_splits']:>10}/{B}"
              f"{r['inert_splits']:>6}/{B}{r['extra_wrong']['median']:>19}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
