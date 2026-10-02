"""V7: does §5's per-item structure survive a fresh server instance? (prereg amendment 2)

C65 established that a re-launched server does not reproduce its own cache
exactly (154/200 bit-identical, worst |Δ| = 1.27e-2, stable within an instance,
different across them). §5's per-item thresholds sit at exactly that order
(e.g. item 13 at 1.4e-2), so a fresh instance could genuinely move the
structure. V7 registered the test: re-run §5's per-item measurement on freshly
extracted scores; falsified if the qualitative structure changes (which items go
inert, which collapse to zero release).

This scores it as registered — a full fresh extraction of the binary corpus
into ``results/v2_instance2`` (frozen ``v2_pool`` caches untouched) — not by a
perturbation substitute.

Faithfulness guard: before scoring anything, the SAME loop runs against the
frozen caches and must reproduce ``paper/workshop/per_item_honest.json``
bit-for-bit. A reimplementation that cannot reproduce the committed artifact is
not allowed to judge the fresh one.
"""

from __future__ import annotations

import importlib.util as ilu
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKSHOP = REPO / "paper" / "workshop"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from verdict_cert import uscis  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402

_s = ilu.spec_from_file_location("sw", WORKSHOP / "sweep_honest_budget.py")
sw = ilu.module_from_spec(_s)
_s.loader.exec_module(sw)

ALPHA, TAU, PASS_MARK = 0.10, 1e-2, 1   # sweep_per_item.py's pins, verbatim


def per_item(pool_dir: Path) -> dict:
    """sweep_per_item.py's measurement, parameterized by cache dir."""
    sw.POOL = pool_dir                     # load_per_extractor reads module global
    rows, items = sw.load_per_extractor()
    students = sorted({s for (s, _i) in rows})
    _, cal, test = sw.v3.m6.split_students(students, 0)
    ok = [s for s in cal if all((s, i) in rows for i in items)]
    te = [s for s in test if all((s, i) in rows for i in items)]

    def score(cell, tau):
        ps = [sw.truncate(p, tau) for p in cell["p"]]
        return {o: min(1.0 - p[o] for p in ps) for o in (1, 0)}

    out = {"n_cal": len(ok), "n_test": len(te), "items": {}}
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
        rec["inert"] = (rec["measured"]["threshold"] == rec["truncated"]["threshold"]
                        and rec["measured"]["released"] == rec["truncated"]["released"])
        rec["zero_release"] = rec["truncated"]["released"] == 0
        out["items"][it] = rec
    return out


def structure(d: dict) -> dict:
    return {"inert": sorted(it for it, r in d["items"].items() if r["inert"]),
            "zero_release": sorted(it for it, r in d["items"].items() if r["zero_release"]),
            "wrong_measured": sum(r["measured"]["wrong"] for r in d["items"].values()),
            "wrong_truncated": sum(r["truncated"]["wrong"] for r in d["items"].values())}


def main() -> int:
    committed_json = json.loads((WORKSHOP / "per_item_honest.json").read_text())

    # --- faithfulness guard: reproduce the committed artifact first ---
    base = per_item(REPO / "results" / "v2_pool")
    mism = []
    for r in committed_json["items"]:
        it = r["item"]
        b = base["items"][it]
        for tag in ("measured", "truncated"):
            for f in ("threshold", "released", "wrong"):
                if r[tag][f] != b[tag][f]:
                    mism.append((it, tag, f, r[tag][f], b[tag][f]))
    if mism:
        print("FAITHFULNESS GUARD FAILED — this loop does not reproduce "
              "per_item_honest.json; refusing to score V7:")
        for m in mism[:8]:
            print("  ", m)
        return 1
    print(f"faithfulness guard: reproduces per_item_honest.json exactly "
          f"({len(committed_json['items'])} items x 2 gates x 3 fields)")

    # --- the fresh instance ---
    fresh = per_item(REPO / "results" / "v2_instance2")
    sb, sf = structure(base), structure(fresh)
    print(f"\n{'':<12}{'committed instance':>24}{'fresh instance':>22}")
    print(f"{'inert':<12}{','.join(sb['inert']) or '-':>24}{','.join(sf['inert']) or '-':>22}")
    print(f"{'zero-rel':<12}{','.join(sb['zero_release']) or '-':>24}{','.join(sf['zero_release']) or '-':>22}")
    print(f"{'wrong m/t':<12}{str(sb['wrong_measured'])+'/'+str(sb['wrong_truncated']):>24}"
          f"{str(sf['wrong_measured'])+'/'+str(sf['wrong_truncated']):>22}")

    print(f"\n{'item':<5}{'cal_err b->f':>16}{'m_thr b':>12}{'m_thr f':>12}"
          f"{'rel b->f (m)':>14}{'rel b->f (t)':>14}")
    for it in sorted(base["items"], key=lambda i: base["items"][i]["cal_error"]):
        b, f = base["items"][it], fresh["items"][it]
        print(f"{it:<5}{b['cal_error']:.3f}->{f['cal_error']:.3f}   "
              f"{b['measured']['threshold']:>11.2e} {f['measured']['threshold']:>11.2e}"
              f"{str(b['measured']['released'])+'->'+str(f['measured']['released']):>14}"
              f"{str(b['truncated']['released'])+'->'+str(f['truncated']['released']):>14}")

    held = (sb["inert"] == sf["inert"] and sb["zero_release"] == sf["zero_release"])
    inert_note = ("unchanged" if sb["inert"] == sf["inert"]
                  else f"changed {sb['inert']} -> {sf['inert']}")
    zr_note = ("unchanged" if sb["zero_release"] == sf["zero_release"]
               else f"changed {sb['zero_release']} -> {sf['zero_release']}")
    print(f"\nV7 {'HELD' if held else 'FALSIFIED'}: inert set {inert_note}, "
          f"zero-release set {zr_note}")

    out = REPO / "results" / "readout_validation" / "v7_instance.json"
    out.write_text(json.dumps({
        "base": base, "fresh": fresh,
        "structure_base": sb, "structure_fresh": sf,
        "V7_held": held}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
