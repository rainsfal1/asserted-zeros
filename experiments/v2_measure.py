"""v2 measurements — loop arms, both defenses, attack reproductions, P1–P6.

Design authority: ``docs/v2-pool-preregistration.md`` §§2–3 as amended by §6 and §7
(frozen; the LATEST amendment governs).
Everything here is offline math over the rendered pool + the two per-option extraction
caches; no GPU, no network. Phases:

  sessions      test-split sessions (split seed 0 primary; §6.6) — one per test student,
                first elicitations at the hash-assigned per-(student, item) level.
  calibrate     per-item calibration streams from the cal split (one row per
                (student, item) at the hash level — §6.6), floors, and D2's
                Learn-then-Test threshold selection (§6.2).
  arms          engine runs: policies × rules (plain / D1-spend / D2-LTT / legacy
                builder = attack 2) + the `none` baseline; 5 session seeds.
  predictions   P1–P6 scored exactly as amended (§6.1 + §7.1 operating points).
  report        REPORT.md — scores §-amended clauses, states so in the header.

Calibration streams are PER-ITEM (117 gold scores each, threshold per item at
αᵢ = Σα/10) — the structure M6 used and the one §6.1's floor arithmetic assumes.

ATTACK-2 ARM: imports the quarantined legacy builder. This is the ONE disclosed
exception to "v2 never imports the quarantine" (prereg §2 + §6.6; ratchet grown in
this same diff, as the ratchet requires).
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
sys.path.insert(0, str(REPO / "experiments"))

from _legacy import build_set as legacy_build_set  # noqa: E402  (attack-2 arm — the disclosed exception, prereg §6.6)
from verdict_cert import engine, uscis  # noqa: E402
from verdict_cert.calibration import alpha_for_elicitations  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402

_spec = _ilu.spec_from_file_location("m6_frozen", REPO / "experiments" / "m6_powergrading.py")
m6 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(m6)

OUT_DEFAULT = "results/v2_pool"
TAGS = ("judge", "evaluator")
SIGMA_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)   # M6's grid, prereg §2
LTT_GRID = tuple(round(t, 4) for t in                      # the committed 14-point grid
                 (0.0, 1e-05, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95))
POLICIES = ("none", "repeat", "rephrase", "ladder")
SESSION_SEEDS = (0, 1, 2, 3, 4)
BUDGET = 3
K_ITEMS = 10
LTT_ALPHA = 0.05
#: Every scored clause's operating points, DECLARED AS DATA so a test can compare
#: them against the frozen prereg (§6.1 as re-pointed by §7.1). Audit wave 5 found
#: P6 still scored at the superseded 0.05 — a literal buried in the scorer, which
#: nothing could cross-check. Change these only with a dated amendment.
OPERATING_POINTS: dict[str, tuple[float, ...]] = {
    "P1": (0.10, 0.15),          # §6.1
    "P2": (0.20, 0.30, 0.50),    # §6.1
    "P4": (0.10, 0.20),          # §6.1
    "P5": (0.20, 0.30),          # §6.1
    "P6": (0.20,),               # §7.1 — re-pointed from the degenerate 0.05
}
DRAW_LEVELS = ("tel", "snr0", "snr_m5", "snr_m10")
OPT_MAP = {"a": 1, "b": 0}   # extraction options -> USCIS domain


def _fnv(s: str) -> int:
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def load_pool(outdir: Path) -> list[dict]:
    recs = [json.loads(x) for x in (outdir / "records.jsonl").open() if x.strip()]
    for tag in TAGS:
        p = outdir / f"extract-{tag}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p} — run `v2_pool.py extract --tag {tag}` first")
        ex = {json.loads(x)["id"]: json.loads(x) for x in p.open()}
        for r in recs:
            row = ex.get(r["id"])
            if row is None:
                raise SystemExit(f"record {r['id']} missing from extract-{tag}")
            r.setdefault("scores", {})[tag] = {
                OPT_MAP[o]: 1.0 - p_ for o, p_ in row["dist"].items()}
    for r in recs:
        r["score_min"] = {o: min(r["scores"][t][o] for t in TAGS) for o in (1, 0)}
    return recs


def splits(recs: list[dict], seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    """Split seed 0 is primary for every scored clause (§6.6); other seeds serve
    P4's pre-committed 20-seed threshold-reproducibility measurement only."""
    # No pre-sort: split_students re-sorts by hash_str(f"{seed}:{s}") internally, so
    # sorting here by hash_str(s) is a no-op that implies a seeding relationship
    # which does not exist (audit wave 5). Plain sorted() for determinism only.
    return m6.split_students(sorted({r["student"] for r in recs}), seed)


def hash_level(student: str, item: str) -> str:
    return DRAW_LEVELS[_fnv(f"{student}:{item}") % len(DRAW_LEVELS)]


def first_elicitations(recs: list[dict], students: list[str]) -> dict[tuple, dict]:
    """(student, item) -> the record at the hash-assigned level (§6.6)."""
    wanted = {(s, it): hash_level(s, it)
              for s in students for it in sorted({r["item"] for r in recs})}
    out = {}
    for r in recs:
        key = (r["student"], r["item"])
        if wanted.get(key) == r["level"]:
            out[key] = r
    return out


def make_builder(thresholds: dict[str, float], *, spend: bool, sigma: float,
                 legacy: bool = False):
    """build(record, k): per-item conformal set from min-combined scores.

    Spending re-derives the per-item threshold at αᵢ/2ᵏ from the stored per-item
    calibration scores (closure over CAL_SCORES set by phase wiring)."""
    def build(rec, k):
        it = rec["item"]
        if spend:
            a_k = alpha_for_elicitations(sigma / K_ITEMS, k)
            thr = calibrate_threshold(CAL_SCORES[it], a_k)
        else:
            thr = thresholds[it]
        sc = rec["score_min"]
        if legacy:
            return frozenset(legacy_build_set(sc, thr))
        return uscis.sets_from_dists({it: sc}, thr)[it]
    return build


CAL_SCORES: dict[str, list[float]] = {}   # item -> gold nonconformity, cal split


def phase_sessions(outdir: Path) -> None:
    recs = load_pool(outdir)
    _, _, test = splits(recs)
    fe = first_elicitations(recs, test)
    sessions = []
    for s in test:
        items = {}
        for it in sorted({r["item"] for r in recs}):
            r = fe.get((s, it))
            if r is None:
                items = None
                break
            items[it] = {"gold": r["gold"], "wording": r["wording"],
                         "record_id": r["id"], "draw": r["draw"],
                         "score_min": r["score_min"],   # the PINNED record's scores —
                         # priming must use these, not a last-write-wins lookup
                         # (audit wave 4, A1: 75% of slots primed off-level before)
                         "set": None}          # set filled per-arm at run time
        if items is None:
            continue                            # student missing a rendered item
        golds = {it: d["gold"] for it, d in items.items()}
        sessions.append({"student": s, "items": items,
                         "true_verdict": uscis.verdict_of(golds)})
    (outdir / "sessions.json").write_text(json.dumps(
        {"split_seed": 0, "n_test": len(sessions), "sessions": sessions}, indent=1))
    print(f"sessions: {len(sessions)} test sessions")


def phase_calibrate(outdir: Path) -> None:
    recs = load_pool(outdir)
    _, cal, _ = splits(recs)
    fe = first_elicitations(recs, cal)
    per_item: dict[str, list[float]] = {}
    for (s, it), r in sorted(fe.items()):
        per_item.setdefault(it, []).append(r["score_min"][r["gold"]])
    n = {it: len(v) for it, v in per_item.items()}
    floors = {"plain": round(sum(1.0 / (n[it] + 1) for it in n), 6),
              "spend_k1": round(sum(2.0 / (n[it] + 1) for it in n), 6),
              "n_per_item": n}
    thresholds = {}
    for sigma in SIGMA_GRID:
        a_i = sigma / K_ITEMS
        thresholds[str(sigma)] = {
            "plain": {it: calibrate_threshold(per_item[it], a_i) for it in per_item},
            "spend_k1": {it: calibrate_threshold(
                per_item[it], alpha_for_elicitations(a_i, 1)) for it in per_item},
        }
    zero_share = (sum(1 for v in per_item.values() for x in v if x == 0.0)
                  / sum(len(v) for v in per_item.values()))

    # P4's "reported regardless": threshold reproducibility across 20
    # student-disjoint split seeds — the training-conditional dispersion that
    # was v1's standing negative, on a real dial (wave-5 F9: never implemented).
    repro: dict = {}
    for sigma in (0.10, 0.20):
        a_i = sigma / K_ITEMS
        by_item: dict[str, list[float]] = {}
        for sd in range(20):
            _, cal_s, _ = splits(recs, sd)
            fe_s = first_elicitations(recs, cal_s)
            streams: dict[str, list[float]] = {}
            for (st, it), r in fe_s.items():
                streams.setdefault(it, []).append(r["score_min"][r["gold"]])
            for it, xs in streams.items():
                t = calibrate_threshold(xs, a_i)
                by_item.setdefault(it, []).append(None if math.isinf(t) else t)
        repro[str(sigma)] = {
            it: {"n_seeds": len(v),
                 "n_infinite": sum(1 for x in v if x is None),
                 "min": min([x for x in v if x is not None], default=None),
                 "max": max([x for x in v if x is not None], default=None),
                 "seed0": v[0]}
            for it, v in by_item.items()}

    # The score histogram and the off-option residual (§4, P4's line).
    all_scores = [x for v in per_item.values() for x in v]
    edges = (0.0, 1e-9, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001)
    hist = {f"[{edges[i]},{edges[i + 1]})": sum(1 for x in all_scores
                                                if edges[i] <= x < edges[i + 1])
            for i in range(len(edges) - 1)}
    mass = []
    for tag in TAGS:
        ep = outdir / f"extract-{tag}.jsonl"
        if ep.exists():
            mass += [json.loads(x).get("mass_unnormalized") for x in ep.open()
                     if x.strip()]
    mass = sorted(m for m in mass if m is not None)
    mass_q = ({f"p{q}": round(mass[min(int(q / 100 * len(mass)), len(mass) - 1)], 4)
               for q in (5, 25, 50, 75, 95)} if mass else None)

    out = {"floors": floors, "zero_score_share": round(zero_share, 4),
           "threshold_reproducibility_20_splits": repro,
           "score_histogram": hist, "mass_unnormalized_quantiles": mass_q,
           "thresholds": {k: {r: {it: (None if math.isinf(t) else t)
                                  for it, t in d.items()}
                              for r, d in v.items()}
                          for k, v in thresholds.items()},
           "cal_scores": per_item}
    (outdir / "calibration.json").write_text(json.dumps(out, indent=1))
    print(f"calibrate: floors {floors['plain']} / {floors['spend_k1']}, "
          f"zero-share {zero_share:.3f}, 20-split reproducibility measured")


def _run_arm(sessions, recs, students, builder, policy, seed):
    return engine.run_sessions(
        sessions, [r for r in recs if r["student"] in students], builder,
        lambda st: uscis.certify(st), lambda st: uscis.reask_ranking(st),
        allowed_students=frozenset(students), seed=seed, policy=policy,
        budget=BUDGET)


def _prime_sessions(sessions, builder):
    """Fill each session's first-elicitation sets with the arm's builder (k=1),
    from the PINNED record's own scores (carried in the session row)."""
    out = []
    for s in sessions:
        s2 = {**s, "items": {}}
        for it, d in s["items"].items():
            # JSON round-trip turns the int option keys into strings — coerce
            # back at the read boundary, once (audit wave 4, A1 follow-on)
            sc = {int(k): v for k, v in d["score_min"].items()}
            rec = {"item": it, "score_min": sc}
            s2["items"][it] = {**d, "score_min": sc, "set": sorted(builder(rec, 1))}
        out.append(s2)
    return out


def _ltt_select(sessions, recs, cal_students, sigma, policy) -> dict:
    """§6.2 as amended by §8.2 — Learn-then-Test, per policy, utility-selected.

    Two corrections wave 5 forced, both pre-data:

    **The protocol must match (§2's "identical re-ask procedure").** The
    threshold is selected under the SAME policy it is later evaluated under;
    selecting under `rephrase` and applying to `repeat`/`ladder`/`none` would
    attach LTT's guarantee to a loop it never ran on.

    **Selection is by utility among the passers (§8.2), not "largest t".**
    Release is UNIMODAL in t, not monotone (measured): at the bottom of the grid
    no option clears the threshold, the empty set becomes FULL by the §2 rule and
    nothing certifies; at the top every option clears, the sets are FULL again and
    nothing certifies. So the largest candidate passes vacuously — zero release,
    hence zero risk, hence a tiny p-value — and D2 would be a no-op by
    construction. LTT licenses any choice among the candidates that clear the
    corrected test (Bonferroni controls the whole family), so we take the
    most-releasing passer (ties to the smaller t) and report the full pass set.
    """
    passers = []
    for t in LTT_GRID:
        builder = (lambda thr: (lambda rec, k: uscis.sets_from_dists(
            {rec["item"]: rec["score_min"]}, thr)[rec["item"]]))(t)
        primed = _prime_sessions(sessions, builder)
        rows = _run_arm(primed, recs, cal_students, builder, policy, seed=0)
        n = len(rows)
        wrong = sum(r["released_wrong"] for r in rows)
        rel = sum(r["certified"] for r in rows)
        rate = wrong / n if n else 1.0
        # Hoeffding upper p-value for H0: true rate > sigma
        p = math.exp(-2 * n * max(0.0, sigma - rate) ** 2) if rate <= sigma else 1.0
        if p <= LTT_ALPHA / len(LTT_GRID):
            passers.append({"t": t, "p": p, "n_cal": n,
                            "cal_release": rel / n if n else 0.0,
                            "cal_wrong_rate": rate})
    if not passers:
        return {"t": None, "passers": [], "n_cal": len(sessions),
                "why": "no candidate clears Bonferroni — see §8.1's feasibility floor"}
    best = _pick_passer(passers)
    return {"t": best["t"], "selected_cal_release": best["cal_release"],
            "selected_p": best["p"], "passers": passers, "n_cal": best["n_cal"]}


def _pick_passer(passers: list[dict]) -> dict | None:
    """§8.2's selection rule: the MOST-RELEASING passer, ties to the smaller t.

    Every candidate here already cleared the same Bonferroni-corrected test, so
    choosing among them by utility is what LTT licenses — and "largest t" lands
    on the top of the grid, where every set is FULL, nothing certifies, and the
    candidate passes vacuously by releasing nothing at all."""
    if not passers:
        return None
    return max(passers, key=lambda c: (c["cal_release"], -c["t"]))


def _wording_supply(recs, students) -> dict:
    """Distinct wordings available per (item, gold) inside one split — the pool
    arithmetic P6's exhaustion clause is scored against (§6.5)."""
    sup: dict = {}
    for r in recs:
        if r["student"] in students:
            sup.setdefault(f"{r['item']}|{r['gold']}", set()).add(r["wording"])
    return {k: len(v) for k, v in sup.items()}


def _exhaustion_audit(rows, supply) -> dict:
    """§6.5: exhaustion is per-ITEM — "the scoring function counts per-item
    attempts, not the global budget". Every `pool_exhausted` stop is checked
    against the exhausted item's own supply, so a spurious exhaustion (the loop
    quitting while wordings remained) is a violation, not a pass."""
    fired = violations = 0
    by_item: dict = {}
    for r in rows:
        if r["stop"] != "pool_exhausted":
            continue
        fired += 1
        it = r.get("stop_item")
        by_item[str(it)] = by_item.get(str(it), 0) + 1
        gold = r["items"].get(it, {}).get("gold")
        avail = max(supply.get(f"{it}|{gold}", 0) - 1, 0)  # minus the elicited wording
        if r.get("attempts", {}).get(it, 0) < avail:
            violations += 1
    return {"fired": fired, "violations": violations, "by_item": by_item}


def _confound_audit(rows, orig_of, rec_by_id) -> dict:
    """§6.5's pre-committed breakdowns: level transitions (first elicitation ->
    drawn record) and the same-voice subset, so the repeat/rephrase contrast's
    two confounds are measured rather than assumed away."""
    trans: dict = {}
    same = tot = 0
    for r in rows:
        for rd in r["rounds"]:
            src = rec_by_id.get(orig_of.get((r["student"], rd["item"])))
            dst = rec_by_id.get(rd["record_id"])
            if not src or not dst:
                continue
            key = f"{src.get('level')}->{dst.get('level')}"
            trans[key] = trans.get(key, 0) + 1
            tot += 1
            same += int(src.get("voice_code") == dst.get("voice_code"))
    return {"level_transitions": trans, "same_voice": same, "n_rounds": tot}


def _cell(primed, recs, test, builder, policy, supply, rec_by_id) -> list:
    """One arm cell: five session seeds, with the §6.5 audits attached."""
    orig_of = {(s["student"], it): d.get("record_id")
               for s in primed for it, d in s["items"].items()}
    per_seed = []
    for seed in SESSION_SEEDS:
        rows = _run_arm(primed, recs, test, builder, policy, seed)
        n = len(rows)
        rel = sum(r["certified"] for r in rows)
        wrong = sum(r["released_wrong"] for r in rows)
        per_seed.append({
            "n": n, "released": rel, "wrong": wrong,
            "release_rate": rel / n if n else None,
            "wrong_among_released": wrong / rel if rel else None,
            "wrong_per_1000": 1000 * wrong / n if n else None,
            "reasks": sum(r["n_reasks"] for r in rows),
            "stops": {k: sum(1 for r in rows if r["stop"] == k)
                      for k in ("resolved", "budget_exhausted", "pool_exhausted",
                                "stopping_rule", "no_loop")},
            "exhaust": _exhaustion_audit(rows, supply),
            "confound": _confound_audit(rows, orig_of, rec_by_id),
        })
    return per_seed


def phase_arms(outdir: Path) -> None:
    recs = load_pool(outdir)
    _, cal, test = splits(recs)
    sessions = json.loads((outdir / "sessions.json").read_text())["sessions"]
    calib = json.loads((outdir / "calibration.json").read_text())
    CAL_SCORES.clear()
    CAL_SCORES.update(calib["cal_scores"])

    # D2 selection on the CAL split (cal sessions built the same way)
    fe_cal = first_elicitations(recs, cal)
    cal_sessions = []
    for s in cal:
        items = {}
        ok = True
        for it in sorted({r["item"] for r in recs}):
            r = fe_cal.get((s, it))
            if r is None:
                ok = False
                break
            items[it] = {"gold": r["gold"], "wording": r["wording"],
                         "record_id": r["id"], "draw": r["draw"],
                         "score_min": r["score_min"], "set": None}
        if ok:
            golds = {it: d["gold"] for it, d in items.items()}
            cal_sessions.append({"student": s, "items": items,
                                 "true_verdict": uscis.verdict_of(golds)})

    supply = _wording_supply(recs, test)
    rec_by_id = {r["id"]: r for r in recs}
    res: dict = {"grid": SIGMA_GRID, "arms": {},
                 "wording_supply_test": supply}
    d2_selected: dict = {}
    for sigma in SIGMA_GRID:
        thr_plain = {it: (float("inf") if v is None else v)
                     for it, v in calib["thresholds"][str(sigma)]["plain"].items()}
        # D2 is selected PER POLICY — the calibration protocol must be the test
        # protocol (§2's "identical re-ask procedure"; wave-5 F5)
        d2_sel = {pol: _ltt_select(cal_sessions, recs, cal, sigma, pol)
                  for pol in POLICIES}
        d2_selected[str(sigma)] = d2_sel
        rules = {
            "plain_nospend": make_builder(thr_plain, spend=False, sigma=sigma),
            "d1_spend": make_builder(thr_plain, spend=True, sigma=sigma),
            "legacy_nospend": make_builder(thr_plain, spend=False, sigma=sigma,
                                           legacy=True),
        }
        cell: dict = {}
        for rname, builder in rules.items():
            primed = _prime_sessions(sessions, builder)
            for policy in POLICIES:
                cell[f"{rname}|{policy}"] = _cell(primed, recs, test, builder,
                                                  policy, supply, rec_by_id)
        for policy in POLICIES:
            t = d2_sel[policy]["t"]
            if t is None:
                continue                      # disclosed, never silently skipped
            builder = (lambda tt: (lambda rec, k: uscis.sets_from_dists(
                {rec["item"]: rec["score_min"]}, tt)[rec["item"]]))(t)
            primed = _prime_sessions(sessions, builder)
            cell[f"d2_ltt|{policy}"] = _cell(primed, recs, test, builder,
                                             policy, supply, rec_by_id)
        res["arms"][str(sigma)] = cell
    res["d2_selected_thresholds"] = d2_selected
    (outdir / "arms.json").write_text(json.dumps(res, indent=1))
    print("arms: done; D2 t by σ (rephrase):",
          {k: v["rephrase"]["t"] for k, v in d2_selected.items()})


def _pooled(per_seed, num, den):
    n = sum(s[den] for s in per_seed)
    return (sum(s[num] for s in per_seed) / n) if n else None


def _matched_release(arms, rule_d1, rule_d2, policy, band=0.05) -> dict:
    """P5's matched-release comparison — the conjunct that was never scored.

    Each σ cell pools to one (realized release, wrong-per-1000) point; rules are
    compared inside release bands [target, target+band] via the lower risk
    envelope each attains in the band. §6.1's pin: **every target D1 cannot
    attain is reported ABSENT, never a tie** — an unattainable target must not
    be able to prop the prediction up.
    """
    pts: dict = {}
    for rule in (rule_d1, rule_d2):
        acc = []
        for sig in SIGMA_GRID:
            per_seed = arms[str(sig)].get(f"{rule}|{policy}")
            if not per_seed:
                continue
            n = sum(x["n"] for x in per_seed)
            if not n:
                continue
            acc.append({"sigma": sig,
                        "release": sum(x["released"] for x in per_seed) / n,
                        "wrong_per_1000": 1000 * sum(x["wrong"] for x in per_seed) / n})
        pts[rule] = acc
    out: dict = {}
    for i in range(20):
        target = round(0.05 * i, 2)
        sel = {r: [q for q in pts[r] if target <= q["release"] <= target + band]
               for r in pts}
        if not sel[rule_d1] or not sel[rule_d2]:
            out[f"{target}"] = {"status": "absent",
                                "d1_points": len(sel[rule_d1]),
                                "d2_points": len(sel[rule_d2])}
            continue
        d1 = min(q["wrong_per_1000"] for q in sel[rule_d1])
        d2 = min(q["wrong_per_1000"] for q in sel[rule_d2])
        out[f"{target}"] = {"status": "attained", "d1_wrong_per_1000": d1,
                            "d2_wrong_per_1000": d2, "d2_no_higher": d2 <= d1}
    return out


def phase_predictions(outdir: Path) -> None:
    arms = json.loads((outdir / "arms.json").read_text())["arms"]
    calib = json.loads((outdir / "calibration.json").read_text())
    P: dict = {}

    def rate(sig, arm, num="wrong", den="released"):
        return _pooled(arms[str(sig)][arm], num, den)

    def per_seed_rates(sig, arm, num="wrong", den="released"):
        return [(s[num] / s[den] if s[den] else None) for s in arms[str(sig)][arm]]

    # P1 (§6.1): naive loop vs none at Σα ∈ {0.10, 0.15}, ≥4/5 seeds each
    p1 = {}
    for sig in OPERATING_POINTS["P1"]:
        naive = per_seed_rates(sig, "plain_nospend|rephrase")
        none_ = per_seed_rates(sig, "plain_nospend|none")
        wins = sum(1 for a, b in zip(naive, none_)
                   if a is not None and b is not None and a > b)
        p1[str(sig)] = {"seed_wins": wins, "naive": naive, "none": none_}
    P["P1"] = {**p1, "held": all(v["seed_wins"] >= 4 for v in p1.values())}

    # P2 (§6.1): spend strictly fewer wrong UNDER THE LOOPED POLICIES at
    # Σα ∈ {0.20, 0.30, 0.50} pooled — and never more at ANY swept Σα where the
    # unspent arm releases (both conjuncts registered; wave-5 F4 restored them)
    looped = ("repeat", "rephrase", "ladder")
    p2 = {}
    strict_ok = True
    for sig in OPERATING_POINTS["P2"]:
        cell = {}
        for pol in looped:
            w_sp = sum(s["wrong"] for s in arms[str(sig)][f"d1_spend|{pol}"])
            w_no = sum(s["wrong"] for s in arms[str(sig)][f"plain_nospend|{pol}"])
            cell[pol] = {"spend": w_sp, "nospend": w_no}
            strict_ok &= w_sp < w_no
        p2[str(sig)] = cell
    never_more_ok = True
    p2_sweep = {}
    for sig in SIGMA_GRID:
        for pol in looped:
            rel_no = sum(s["released"] for s in arms[str(sig)][f"plain_nospend|{pol}"])
            if rel_no == 0:
                continue  # registered qualifier: only where the unspent arm releases
            w_sp = sum(s["wrong"] for s in arms[str(sig)][f"d1_spend|{pol}"])
            w_no = sum(s["wrong"] for s in arms[str(sig)][f"plain_nospend|{pol}"])
            p2_sweep[f"{sig}|{pol}"] = {"spend": w_sp, "nospend": w_no}
            never_more_ok &= w_sp <= w_no
    P["P2"] = {**p2, "never_more_sweep": p2_sweep,
               "held": strict_ok and never_more_ok}

    # P3: legacy no 0%-risk point in any seed; conformal ≥1 in every seed
    p3 = {"legacy_feasible_seeds": 0, "conformal_feasible_seeds": 0}
    n_seeds = len(SESSION_SEEDS)
    for seed_i in range(n_seeds):
        def feasible(rule):
            return any(
                arms[str(sig)][f"{rule}|none"][seed_i]["released"]
                and arms[str(sig)][f"{rule}|none"][seed_i]["wrong"] == 0
                for sig in SIGMA_GRID)
        p3["legacy_feasible_seeds"] += feasible("legacy_nospend")
        p3["conformal_feasible_seeds"] += feasible("plain_nospend")
    P["P3"] = {**p3, "held": (p3["legacy_feasible_seeds"] == 0
                              and p3["conformal_feasible_seeds"] == n_seeds)}

    # P4 (§6.1): zero-spike < 0.20 and thresholds strictly positive at {0.10, 0.20}
    zs = calib["zero_score_share"]
    thr_pos = all(
        (calib["thresholds"][str(sig)]["plain"][it] or 0) > 0
        for sig in OPERATING_POINTS["P4"] for it in calib["thresholds"][str(sig)]["plain"])
    P["P4"] = {"zero_score_share": zs, "thresholds_positive": thr_pos,
               "held": zs < 0.20 and thr_pos}

    # P5 (§6.1): the registered CONJUNCTION, both halves (wave-5 F3 restored the
    # matched-release half, which was never scored):
    #   (a) at matched realized release, D2's wrong-per-1000 no higher than D1's
    #       at every ATTAINABLE target — unattainable targets absent, never ties;
    #   (b) D2 releases strictly more at equal Σα ∈ {0.20, 0.30} in ≥ 4 of 5 seeds.
    matched = _matched_release(arms, "d1_spend", "d2_ltt", "rephrase")
    p5 = {}
    for sig in OPERATING_POINTS["P5"]:
        if "d2_ltt|rephrase" not in arms[str(sig)]:
            p5[str(sig)] = {"d2": None}   # D2 infeasible at this σ — §8.1, disclosed
            continue
        d2r = [s["release_rate"] for s in arms[str(sig)]["d2_ltt|rephrase"]]
        d1r = [s["release_rate"] for s in arms[str(sig)]["d1_spend|rephrase"]]
        wins = sum(1 for a, b in zip(d2r, d1r) if a is not None and b is not None and a > b)
        p5[str(sig)] = {"seed_wins": wins,
                        "d2_risk": rate(sig, "d2_ltt|rephrase"),
                        "d1_risk": rate(sig, "d1_spend|rephrase")}
    attained = [v for v in matched.values() if v["status"] == "attained"]
    matched_ok = bool(attained) and all(v["d2_no_higher"] for v in attained)
    # a missing D2 cell cannot satisfy the release-advantage half: -1, not 0
    wins_ok = all(v.get("seed_wins", -1) >= 4 for v in p5.values())
    P["P5"] = {**p5, "matched_release": matched, "n_targets_attained": len(attained),
               "matched_ok": matched_ok, "wins_ok": wins_ok,
               "held": matched_ok and wins_ok}

    # P6 (§7.1): |repeat − rephrase| beyond noise at Σα=0.20 spend — 0.05 is below
    # D1's own floor and was the §6.1 defect class, caught by wave 5 still in this
    # scorer after §7.1 re-pointed the clause
    (sig,) = OPERATING_POINTS["P6"]
    rep_ = [s["wrong"] for s in arms[str(sig)]["d1_spend|repeat"]]
    reph = [s["wrong"] for s in arms[str(sig)]["d1_spend|rephrase"]]
    diff = abs(sum(rep_) - sum(reph)) / len(SESSION_SEEDS)
    spread = max(max(rep_) - min(rep_), max(reph) - min(reph))
    # Exhaustion, per §6.5: under `rephrase` ONLY, counted per ITEM against that
    # item's own wording supply. The old scorer summed pool_exhausted over every
    # rule AND policy (repeat/ladder/legacy included) and asked only "> 0" —
    # the global-budget reading §6.5 explicitly corrected (wave-5 F8).
    ex = [s["exhaust"] for s in arms[str(sig)]["d1_spend|rephrase"]]
    fired = sum(e["fired"] for e in ex)
    violations = sum(e["violations"] for e in ex)
    by_item: dict = {}
    for e in ex:
        for k, v in e["by_item"].items():
            by_item[k] = by_item.get(k, 0) + v
    P["P6"] = {"mean_abs_diff": diff, "within_arm_spread": spread,
               "exhaustion_rephrase": {"fired": fired, "violations": violations,
                                       "by_item": by_item},
               "confound_breakdown": {   # §6.5's pre-committed disclosure
                   "repeat": arms[str(sig)]["d1_spend|repeat"][0]["confound"],
                   "rephrase": arms[str(sig)]["d1_spend|rephrase"][0]["confound"]},
               "held": diff > spread and fired > 0 and violations == 0}

    (outdir / "predictions.json").write_text(json.dumps(P, indent=1))
    print("predictions:", {k: v.get("held") for k, v in P.items()})


def phase_report(outdir: Path) -> None:
    """REPORT.md — the scored clauses AND §4's "reported regardless" content.

    §4 pre-commits far more than the six verdicts: the full frontier with
    realized release per cell, per-seed dispersion, n beside every pooled rate,
    the no-abstention baselines, D2's selection record, and §6.5's confound
    breakdowns. Wave-5 F9: rendering only the six verdicts would let the
    strongest cell speak for the instrument.
    """
    P = json.loads((outdir / "predictions.json").read_text())
    calib = json.loads((outdir / "calibration.json").read_text())
    arms_all = json.loads((outdir / "arms.json").read_text())
    arms = arms_all["arms"]
    sess = json.loads((outdir / "sessions.json").read_text())["sessions"]

    # No-abstention baselines (§4): what you get WITHOUT the certificate.
    n_sess = len(sess)
    always_pass_wrong = sum(1 for s in sess if s["true_verdict"] != "PASS")
    point_wrong = 0
    for s in sess:
        prof = {}
        for it, d in s["items"].items():
            sc = {int(k): v for k, v in d["score_min"].items()}
            prof[it] = min(sc, key=sc.get)
        point_wrong += int(uscis.verdict_of(prof) != s["true_verdict"])

    L = ["# v2 pool — REPORT",
         "",
         "**Scores the §4/§6/§7/§8-AMENDED predictions** of "
         "`docs/v2-pool-preregistration.md` (§6.1 + §7.1 operating points; "
         "§8.2's D2 selection; split seed 0) — not superseded text.",
         "",
         f"Floors (measured): plain {calib['floors']['plain']}, "
         f"spend-k1 {calib['floors']['spend_k1']} — §6.1's disclosure, confirmed.",
         f"Calibration zero-score share: {calib['zero_score_share']}"
         f" (v1 was ~0.90 — F-11's fix, measured).",
         "",
         "## Baselines without the certificate (§4)",
         "",
         f"- Test sessions: **n = {n_sess}** (every pooled rate below has this n "
         f"per seed, {n_sess * len(SESSION_SEEDS)} session-runs pooled over "
         f"{len(SESSION_SEEDS)} seeds).",
         f"- Always answer (point estimate, no abstention): "
         f"**{point_wrong}/{n_sess} wrong** "
         f"({1000 * point_wrong / n_sess:.1f} per 1000).",
         f"- Always release PASS: **{always_pass_wrong}/{n_sess} wrong** "
         f"({1000 * always_pass_wrong / n_sess:.1f} per 1000).",
         ""]
    for name in ("P1", "P2", "P3", "P4", "P5", "P6"):
        L.append(f"## {name}: {'HELD' if P[name]['held'] else 'FALSIFIED'}")
        if name == "P3":
            L.append("*Scored on the `|none` arms (builder-only contrast, the clean "
                     "analogue of B1's A/B). Policy `none` consumes no rng, so the five "
                     "session seeds are byte-identical there and the clause's "
                     "\"in any/every seed\" is a single trial, not five.*")
            L.append("")
        L.append("```json")
        L.append(json.dumps({k: v for k, v in P[name].items() if k != 'held'},
                            indent=1)[:2400])
        L.append("```")
        L.append("")

    # P4's pre-committed reproducibility and distributions (§4).
    repro = calib.get("threshold_reproducibility_20_splits", {})
    if repro:
        L += ["## Threshold reproducibility, 20 student-disjoint splits (§4, P4)", "",
              "The training-conditional dispersion v1 could only note as a standing "
              "negative, measured here on a real dial. `no finite t` counts splits "
              "where the item's calibration stream cannot reach the level at all.", "",
              "| Σα | item | seed-0 t | min | max | splits with no finite t |",
              "|---|---|---|---|---|---|"]
        for sig, byitem in repro.items():
            for it, v in sorted(byitem.items()):
                L.append(f"| {sig} | {it} | {v['seed0']} | {v['min']} | {v['max']} "
                         f"| {v['n_infinite']}/{v['n_seeds']} |")
        L.append("")
    if calib.get("score_histogram"):
        L += ["## Score histogram and the off-option residual (§4)", "", "```json",
              json.dumps({"score_histogram": calib["score_histogram"],
                          "mass_unnormalized_quantiles":
                              calib.get("mass_unnormalized_quantiles")}, indent=1),
              "```", ""]

    # The full frontier — every cell, with dispersion and its own denominator.
    L += ["## Frontier — every arm, every Σα (§4: reported regardless)", "",
          "Release and wrong-per-1000 are pooled over seeds; the seed range shows "
          "dispersion. `wrong/1000` counts wrong releases per 1000 SESSIONS "
          "(P(release ∧ wrong)), not per release.", "",
          "| Σα | arm | n (pooled) | release | seed range | wrong/1000 | "
          "wrong among released | re-asks |", "|---|---|---|---|---|---|---|---|"]
    for sig in SIGMA_GRID:
        for key in sorted(arms[str(sig)]):
            ps = arms[str(sig)][key]
            n = sum(x["n"] for x in ps)
            if not n:
                continue
            rel = sum(x["released"] for x in ps)
            wrong = sum(x["wrong"] for x in ps)
            rr = [x["release_rate"] for x in ps if x["release_rate"] is not None]
            span = f"{min(rr):.3f}–{max(rr):.3f}" if rr else "—"
            war = f"{wrong / rel:.4f}" if rel else "—"
            L.append(f"| {sig} | `{key}` | {n} | {rel / n:.3f} | {span} | "
                     f"{1000 * wrong / n:.1f} | {war} | {sum(x['reasks'] for x in ps)} |")
    L.append("")

    # D2's selection record — what LTT could and could not do, per policy.
    L += ["## D2 (Learn-then-Test) selection, per policy (§6.2 + §8.2)", "",
          "`t` is the most-releasing passer (§8.2); `passers` is how many of the "
          "14 candidates cleared Bonferroni at all. `t = None` means D2 is "
          "infeasible at that Σα — §8.1's feasibility floor, not a bug.", "",
          "| Σα | policy | t | passers | cal release at t | n_cal |",
          "|---|---|---|---|---|---|"]
    for sig, by_pol in arms_all.get("d2_selected_thresholds", {}).items():
        for pol, selr in by_pol.items():
            cr = selr.get("selected_cal_release")
            L.append(f"| {sig} | {pol} | {selr.get('t')} | "
                     f"{len(selr.get('passers', []))} | "
                     f"{'—' if cr is None else f'{cr:.3f}'} | "
                     f"{selr.get('n_cal')} |")
    L.append("")

    # §6.5's pre-committed confound breakdowns and the exhaustion audit.
    L += ["## §6.5 confounds and exhaustion, measured", "",
          "`repeat` moves the channel LEVEL; `rephrase` moves the VOICE with "
          "probability 7/8. Both are measured here rather than assumed away.", "",
          "| Σα | arm | exhausted | spurious | same-voice re-asks | level transitions |",
          "|---|---|---|---|---|---|"]
    for sig in SIGMA_GRID:
        for key in sorted(arms[str(sig)]):
            if not key.startswith("d1_spend|") or key.endswith("|none"):
                continue
            ps = arms[str(sig)][key]
            fired = sum(x["exhaust"]["fired"] for x in ps)
            viol = sum(x["exhaust"]["violations"] for x in ps)
            same = sum(x["confound"]["same_voice"] for x in ps)
            tot = sum(x["confound"]["n_rounds"] for x in ps)
            trans: dict = {}
            for x in ps:
                for k, v in x["confound"]["level_transitions"].items():
                    trans[k] = trans.get(k, 0) + v
            top = ", ".join(f"{k}×{v}" for k, v in
                            sorted(trans.items(), key=lambda kv: -kv[1])[:3]) or "—"
            L.append(f"| {sig} | `{key}` | {fired} | {viol} | {same}/{tot} | {top} |")
    L.append("")
    L += ["## Wording supply per (item, gold) in the test split (§6.5 arithmetic)",
          "", "```json",
          json.dumps(arms_all.get("wording_supply_test", {}), indent=1)[:1500],
          "```", ""]
    (outdir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {outdir/'REPORT.md'}")


PHASES = {"sessions": phase_sessions, "calibrate": phase_calibrate,
          "arms": phase_arms, "predictions": phase_predictions,
          "report": phase_report}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=("all", *PHASES))
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    outdir = REPO / args.out
    for name, fn in PHASES.items():
        if args.phase in ("all", name):
            print(f"== {name}")
            fn(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
