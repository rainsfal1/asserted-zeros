"""The step function, per item: what truncation does to an ISSUED certificate.

At an honest budget (clean channel, k=1, Sigma-alpha = alpha_i = 0.10) each item
is its own single-item decision, so "released" is a certificate the system would
actually issue. Where the tau-floor reaches most of an item's calibration sample
the threshold snaps to an endpoint (0 below an error rate of 10/116, 1 above);
where it does not, the gate is bit-identical to the measured one -- four of the
ten items, which is why this is measured per item rather than asserted. (An
earlier docstring said truncation binarizes every score; it does not, since tau
is a probability floor and not a rank cut. Audit wave 6.)

POST-HOC, not pre-registered; the tau-floor is a controlled simulation of the
readout's equation, not a rank-level reconstruction of top-k truncation.
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

_s = ilu.spec_from_file_location("sw", HERE / "sweep_honest_budget.py")
sw = ilu.module_from_spec(_s)
_s.loader.exec_module(sw)

ALPHA = 0.10          # Sigma-alpha = k*alpha = 0.10 at k=1: an honest budget
TAU = 1e-2
PASS_MARK = 1         # max(1, round(6*1/10))


def main() -> int:
    rows, items = sw.load_per_extractor()
    students = sorted({s for (s, _i) in rows})
    _, cal, test = sw.v3.m6.split_students(students, 0)
    ok = [s for s in cal if all((s, i) in rows for i in items)]
    te = [s for s in test if all((s, i) in rows for i in items)]

    def score(cell, tau):
        ps = [sw.truncate(p, tau) for p in cell["p"]]
        return {o: min(1.0 - p[o] for p in ps) for o in (1, 0)}

    out = {"alpha": ALPHA, "tau": TAU, "pass_mark": PASS_MARK,
           "n_cal": len(ok), "n_test": len(te), "items": [],
           "note": "post-hoc controlled simulation; not pre-registered"}
    for it in items:
        rec = {"item": it}
        # error rate the extractors make on this item's calibration rows
        rec["cal_error"] = sum(
            1 for s in ok if score(rows[(s, it)], TAU)[rows[(s, it)]["gold"]] > 0.5
        ) / len(ok)
        for tag, tau in (("measured", 0.0), ("truncated", TAU)):
            thr = calibrate_threshold(
                [score(rows[(s, it)], tau)[rows[(s, it)]["gold"]] for s in ok], ALPHA)
            rel = wrong = 0
            for s in te:
                v = score(rows[(s, it)], tau)
                keep = frozenset(o for o in (1, 0) if v[o] <= thr)
                sets = {it: keep or frozenset({1, 0})}
                c = uscis.certify(sets, pass_mark=PASS_MARK)
                if c.certified:
                    rel += 1
                    wrong += int(c.verdict != uscis.verdict_of(
                        {it: rows[(s, it)]["gold"]}, pass_mark=PASS_MARK))
            rec[tag] = {"threshold": None if thr == float("inf") else thr,
                        "released": rel, "wrong": wrong,
                        "release_rate": rel / len(te)}
        out["items"].append(rec)

    out["items"].sort(key=lambda r: r["cal_error"])
    (HERE / "per_item_honest.json").write_text(json.dumps(out, indent=1))
    print(f"alpha={ALPHA} tau={TAU} n_test={len(te)}\n")
    print(f"{'item':<5}{'cal err':>9}{'  measured rel/wrong':>22}{'  truncated rel/wrong':>23}")
    for r in out["items"]:
        m, t = r["measured"], r["truncated"]
        print(f"{r['item']:<5}{r['cal_error']:>9.3f}"
              f"{m['released']:>14}/{m['wrong']:<7}{t['released']:>14}/{t['wrong']:<7}")
    gained = sum(t["wrong"] for t in (r["truncated"] for r in out["items"]))
    base = sum(m["wrong"] for m in (r["measured"] for r in out["items"]))
    print(f"\nwrong verdicts issued across the ten single-item decisions: "
          f"measured {base}, truncated {gained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
