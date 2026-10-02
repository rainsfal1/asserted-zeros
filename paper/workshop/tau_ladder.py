"""The tau-ladder: what the floor costs at the gate, at the defect's own scale.

Both final reviewers made the same objection: section 5 quantifies the cost at
tau = 1e-2, 2,000x coarser than the writer's actual resolution (5e-6), and the
paper never shows the reader the ladder. This computes section 5's per-item
measurement at tau in {1e-6, 5e-6, 1e-4, 1e-2} -- including 5e-6, the writer's
own constant, which had never been swept.

Same stream, split, alpha, pass mark and machinery as sweep_per_item.py.
Post-hoc, not pre-registered, like the rest of section 5's simulation.
"""

from __future__ import annotations

import importlib.util as ilu
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from verdict_cert import uscis  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402

_s = ilu.spec_from_file_location("sw", HERE / "sweep_honest_budget.py")
sw = ilu.module_from_spec(_s)
_s.loader.exec_module(sw)

ALPHA, PASS_MARK = 0.10, 1
TAUS = (1e-6, 5e-6, 1e-4, 1e-2)


def main() -> int:
    rows, items = sw.load_per_extractor()
    students = sorted({s for (s, _i) in rows})
    _, cal, test = sw.v3.m6.split_students(students, 0)
    ok = [s for s in cal if all((s, i) in rows for i in items)]
    te = [s for s in test if all((s, i) in rows for i in items)]

    def score(cell, tau):
        ps = [sw.truncate(p, tau) for p in cell["p"]]
        return {o: min(1.0 - p[o] for p in ps) for o in (1, 0)}

    out = {"alpha": ALPHA, "pass_mark": PASS_MARK, "n_cal": len(ok),
           "n_test": len(te), "taus": {}, "measured": {}}

    def gate(tau):
        per, zero_share = {}, 0
        for it in items:
            cal_scores = [score(rows[(s, it)], tau)[rows[(s, it)]["gold"]] for s in ok]
            zero_share += sum(1 for x in cal_scores if x == 0.0)
            thr = calibrate_threshold(cal_scores, ALPHA)
            rel = wrong = 0
            for s in te:
                v = score(rows[(s, it)], tau)
                keep = frozenset(o for o in (1, 0) if v[o] <= thr)
                c = uscis.certify({it: keep or frozenset({1, 0})}, pass_mark=PASS_MARK)
                if c.certified:
                    rel += 1
                    wrong += int(c.verdict != uscis.verdict_of(
                        {it: rows[(s, it)]["gold"]}, pass_mark=PASS_MARK))
            per[it] = {"threshold": None if thr == float("inf") else thr,
                       "released": rel, "wrong": wrong}
        return per, zero_share / (len(ok) * len(items))

    base, _ = gate(0.0)
    out["measured"] = {"per_item": base,
                       "wrong_total": sum(r["wrong"] for r in base.values())}
    print(f"{'tau':>8} {'cal zero-share':>15} {'items changed':>14} "
          f"{'zero-release':>13} {'wrong total':>12}")
    print(f"{'0 (meas.)':>8} {'0.0%':>15} {'-':>14} {'0':>13} "
          f"{out['measured']['wrong_total']:>12}")
    for tau in TAUS:
        per, zs = gate(tau)
        changed = [it for it in per
                   if (per[it]["threshold"], per[it]["released"])
                   != (base[it]["threshold"], base[it]["released"])]
        zero_rel = [it for it in per if per[it]["released"] == 0]
        wrong_total = sum(r["wrong"] for r in per.values())
        out["taus"][f"{tau:.0e}"] = {
            "per_item": per, "cal_zero_share": zs,
            "items_changed": changed, "zero_release_items": zero_rel,
            "wrong_total": wrong_total}
        print(f"{tau:>8.0e} {zs:>14.1%} {len(changed):>14} "
              f"{len(zero_rel):>13} {wrong_total:>12}")

    (HERE / "tau_ladder.json").write_text(json.dumps(out, indent=1))
    print("\nwrote tau_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
