"""V3 — the certification frontier: where does the gate begin to release?

Design authority: ``docs/v3-frontier-preregistration.md`` (frozen before this file
existed). Pure offline arithmetic over the committed v2 pool and its two §7.2
extraction caches — no model calls, no re-rendering, nothing fitted.

Two axes: conjunction size ``k`` and extraction error ``ε`` (via channel strata).
The registered question is not "does our system work" but "under what measurable
operating conditions can an uncertainty-based decision gate certify usefully?" —
so an empty region is a result, not a failure.

    python experiments/v3_frontier.py --out results/v3_frontier
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verdict_cert import uscis  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402

_spec = _ilu.spec_from_file_location("m6_frozen", REPO / "experiments" / "m6_powergrading.py")
m6 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(m6)

POOL = REPO / "results" / "v2_pool"
TAGS = ("judge", "evaluator")
OPT_MAP = {"a": 1, "b": 0}                      # extraction option -> USCIS domain
STRATA = ("tel", "snr0", "snr_m5", "snr_m10", "mixed")     # §3
K_VALUES = (1, 2, 3, 5, 10)                                # §2
N_SUBSETS = 5
ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)   # §4
CONF_CUTS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999)
BASE_PASS_MARK, BASE_K = 6, 10
DRAW_LEVELS = ("tel", "snr0", "snr_m5", "snr_m10")


def _fnv(s: str) -> int:
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def hash_level(student: str, item: str) -> str:
    """v2's per-(student, item) level assignment — the `mixed` stratum."""
    return DRAW_LEVELS[_fnv(f"{student}:{item}") % len(DRAW_LEVELS)]


def load() -> tuple[dict, list[str]]:
    """(student, item, level) -> {"gold", "score", "p"}; plus the item order."""
    recs = [json.loads(x) for x in (POOL / "records.jsonl").open() if x.strip()]
    ex = {}
    for tag in TAGS:
        ex[tag] = {json.loads(x)["id"]: json.loads(x)
                   for x in (POOL / f"extract-{tag}.jsonl").open() if x.strip()}
    table: dict = {}
    for r in recs:
        dists = [{OPT_MAP[o]: p for o, p in ex[t][r["id"]]["dist"].items()} for t in TAGS]
        p_min = {o: min(d[o] for d in dists) for o in (1, 0)}     # combine_min on p
        table[(r["student"], r["item"], r["level"])] = {
            "gold": r["gold"],
            "score": {o: 1.0 - p_min[o] for o in (1, 0)},          # nonconformity
            "p_point": max(p_min, key=p_min.get),                  # point estimate
            "conf": max(p_min.values()),
        }
    items = sorted({r["item"] for r in recs}, key=lambda i: (_fnv(i), i))
    return table, items


def subsets_for(k: int, items: list[str]) -> list[tuple[str, ...]]:
    """§2's rotation — deterministic, seed-free, five per k (one when k = 10)."""
    if k == len(items):
        return [tuple(items)]
    out = []
    for j in range(N_SUBSETS):
        start = (j * k) % len(items)
        out.append(tuple(items[(start + i) % len(items)] for i in range(k)))
    return out


def pass_mark_for(k: int) -> int:
    return max(1, round(BASE_PASS_MARK * k / BASE_K))


def rows_for(table, students, items, stratum):
    """One row per (student, item) in the stratum, plus the students who have a
    COMPLETE item set.

    Two responses in the pool are unspeakable and were excluded at render time
    (v2 prereg §2/§7.4), so their students cannot form a session. They are
    dropped here and counted — never silently, and never by killing the whole
    stratum, which is what returning None on the first gap did.
    """
    out, complete = {}, []
    for s in students:
        got = {}
        for it in items:
            lv = hash_level(s, it) if stratum == "mixed" else stratum
            r = table.get((s, it, lv))
            if r is None:
                break
            got[(s, it)] = r
        else:
            out.update(got)
            complete.append(s)
    return out, complete


def epsilon(rows, items) -> dict:
    """§3: per item, the share of CAL rows whose true option is not the argmax."""
    eps = {}
    for it in items:
        xs = [r for (s, i), r in rows.items() if i == it]
        eps[it] = sum(1 for r in xs if r["p_point"] != r["gold"]) / len(xs)
    return eps


def thresholds(rows, items, alpha) -> dict:
    return {it: calibrate_threshold(
        [r["score"][r["gold"]] for (s, i), r in rows.items() if i == it], alpha)
        for it in items}


def gate(rows, students, items, thr, pm) -> dict:
    """The conformal gate: sets by threshold, release iff |V(S)| = 1."""
    rel = wrong = 0
    setsize = 0
    for s in students:
        sets, gold = {}, {}
        for it in items:
            r = rows[(s, it)]
            keep = frozenset(o for o in (1, 0) if r["score"][o] <= thr[it])
            sets[it] = keep or frozenset({1, 0})           # empty -> FULL
            gold[it] = r["gold"]
            setsize += len(sets[it])
        cert = uscis.certify(sets, pass_mark=pm)
        truth = uscis.verdict_of(gold, pass_mark=pm)
        if cert.certified:
            rel += 1
            wrong += int(cert.verdict != truth)
    n = len(students)
    return {"n": n, "released": rel, "wrong": wrong,
            "release_rate": rel / n, "wrong_release_rate": wrong / n,
            "err_among_released": (wrong / rel) if rel else None,
            "mean_set_size": setsize / (n * len(items))}


def baseline_confidence(rows, students, items, cut, pm) -> dict:
    """§4 baseline 2: release iff every item's point estimate clears `cut`."""
    rel = wrong = 0
    for s in students:
        pts = {it: rows[(s, it)]["p_point"] for it in items}
        gold = {it: rows[(s, it)]["gold"] for it in items}
        if min(rows[(s, it)]["conf"] for it in items) >= cut:
            rel += 1
            wrong += int(uscis.verdict_of(pts, pass_mark=pm)
                         != uscis.verdict_of(gold, pass_mark=pm))
    n = len(students)
    return {"cut": cut, "n": n, "released": rel, "wrong": wrong,
            "release_rate": rel / n, "wrong_release_rate": wrong / n}


def baseline_always(rows, students, items, pm) -> dict:
    """§4 baseline 1: answer everything from the point estimate."""
    wrong = sum(
        int(uscis.verdict_of({it: rows[(s, it)]["p_point"] for it in items}, pass_mark=pm)
            != uscis.verdict_of({it: rows[(s, it)]["gold"] for it in items}, pass_mark=pm))
        for s in students)
    n = len(students)
    return {"n": n, "released": n, "wrong": wrong,
            "release_rate": 1.0, "wrong_release_rate": wrong / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v3_frontier")
    args = ap.parse_args()
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    table, items = load()
    students = sorted({s for (s, _i, _l) in table})
    _, cal, test = m6.split_students(students, 0)
    print(f"split: cal={len(cal)} test={len(test)}; items={items}")

    res: dict = {"strata": {}, "meta": {
        "alphas": ALPHAS, "k_values": K_VALUES, "n_subsets": N_SUBSETS,
        "item_order": items, "conf_cuts": CONF_CUTS}}
    for stratum in STRATA:
        cal_rows_all, cal_ok = rows_for(table, cal, items, stratum)
        test_rows_all, test_ok = rows_for(table, test, items, stratum)
        eps = epsilon(cal_rows_all, items)
        entry = {"epsilon_by_item": eps,
                 "epsilon_mean": sum(eps.values()) / len(eps),
                 "n_cal": len(cal_ok), "n_test": len(test_ok),
                 "n_cal_dropped": len(cal) - len(cal_ok),
                 "n_test_dropped": len(test) - len(test_ok), "cells": []}
        for k in K_VALUES:
            pm = pass_mark_for(k)
            for j, subset in enumerate(subsets_for(k, items)):
                base = baseline_always(test_rows_all, test_ok, subset, pm)
                conf = [baseline_confidence(test_rows_all, test_ok, subset, c, pm)
                        for c in CONF_CUTS]
                for a in ALPHAS:
                    thr = thresholds(cal_rows_all, subset, a)
                    g = gate(test_rows_all, test_ok, subset, thr, pm)
                    finite = [t for t in thr.values() if not math.isinf(t)]
                    entry["cells"].append({
                        "k": k, "subset": j, "items": list(subset), "pass_mark": pm,
                        "alpha_i": a, "sigma_alpha": round(k * a, 4),
                        "guarantee": (k * a) <= 1.0,
                        "eps_mean_subset": sum(eps[i] for i in subset) / k,
                        "thr_min": min(finite) if finite else None,
                        "thr_max": max(finite) if finite else None,
                        "n_infinite_thr": sum(1 for t in thr.values() if math.isinf(t)),
                        **g,
                        "baseline_always": base,
                        "baseline_confidence": conf,
                    })
        res["strata"][stratum] = entry
        print(f"  {stratum:<8} ε̄={entry['epsilon_mean']:.4f} "
              f"cells={len(entry['cells'])} n_test={entry['n_test']}")
    (outdir / "frontier.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {outdir/'frontier.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
