"""The per-option probe — bounding v1's truncation artifact on its own pools.

Design authority: ``docs/per-option-probe-preregistration.md`` (frozen). 200 records per
pool (s3, m3), stratified by the support size of the committed truncated distribution
(100 support-1 / 50 support-2 / 50 support-3, FNV-1a order of record id, seed-free),
re-extracted under the SAME v1 prompt — the scoring transport is the only changed
variable. **Transport note (v2 prereg §7.2/§8.5):** ``canonical_per_option``'s forced-pass
transport is inert on this box's vLLM and now RAISES rather than fabricating; the probe
runs on the measured single-call transport §7.2 pins, once implemented. Nothing here
silently scores an unverified pass.

One pin the prereg left open, decided here and disclosed in the REPORT: the stratum is
defined by the **judge's** committed distribution (one 200-record sample per pool, both
extractors run on it; Q1–Q3 are scored per extractor against that extractor's own
committed dist, with per-extractor stratum counts reported).

  python experiments/per_option_probe.py extract --endpoint http://localhost:8001/v1 --tag judge
  python experiments/per_option_probe.py extract --endpoint http://localhost:8001/v1 --tag evaluator
  python experiments/per_option_probe.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verdict_cert.calibration import combine_min  # noqa: E402
from verdict_cert.certificate import calibrate_threshold  # noqa: E402
from verdict_cert.items import load_items  # noqa: E402

OUT = REPO / "results" / "per_option_probe"
POOLS = ("s3", "m3")
TAGS = ("judge", "evaluator")
STRATA = {1: 100, 2: 50, 3: 50}


def _fnv(s: str) -> int:
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _committed(pool: str, tag: str) -> dict[str, dict]:
    return {json.loads(x)["id"]: json.loads(x)
            for x in (REPO / "results" / pool / f"extract-{tag}.jsonl").open()}


def _transcripts(pool: str) -> dict[str, dict]:
    return {json.loads(x)["id"]: json.loads(x)
            for x in (REPO / "results" / pool / "transcripts.jsonl").open()}


def sample(pool: str) -> list[str]:
    """The pre-registered stratified sample, deterministic and seed-free."""
    judge = _committed(pool, "judge")
    by_support: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for rid, row in judge.items():
        sup = sum(1 for v in row["dist"].values() if v > 0.0)
        by_support[sup].append(rid)
    picked: list[str] = []
    for sup, want in STRATA.items():
        pool_ids = sorted(by_support[sup], key=lambda r: (_fnv(r), r))
        picked.extend(pool_ids[:want])
    return picked


def cmd_extract(args) -> int:
    from verdict_cert.extraction import Extractor

    items = {it.id: it for it in load_items()}
    OUT.mkdir(parents=True, exist_ok=True)
    from verdict_cert.extraction import PerOptionScoringError
    ex = Extractor(args.endpoint)
    for pool in POOLS:
        tx = _transcripts(pool)
        ids = sample(pool)
        out_path = OUT / f"{pool}-peropt-{args.tag}.jsonl"
        done = set()
        if out_path.exists():
            done = {json.loads(x)["id"] for x in out_path.open() if x.strip()}
        todo = [i for i in ids if i not in done]
        print(f"{pool}/{args.tag}: model={ex.model}, {len(todo)} of {len(ids)} to go")
        with out_path.open("a") as sink:
            for n, rid in enumerate(todo, 1):
                rec = tx[rid]
                try:
                    po = ex.canonical_measured(items[rec["item"]], rec["transcript"],
                                               top_k=args.top_k)
                except PerOptionScoringError as e:
                    # An option with no variant even in the top-100 is exactly the
                    # truncation this probe measures. Recording it as UNSCOREABLE
                    # keeps the guard's meaning (never a silent zero) without
                    # letting one record halt the run; the report counts them.
                    sink.write(json.dumps({"id": rid, "unscoreable": True,
                                           "why": str(e)[:200]}) + "\n")
                    continue
                sink.write(json.dumps({
                    "id": rid,
                    # full precision — see v2_pool.cmd_extract (audit wave 5)
                    "dist": dict(po.dist),
                    "raw_logprob": dict(po.raw_logprob),
                    "mass_unnormalized": po.mass_unnormalized,
                    "choice": po.choice,
                }) + "\n")
                if n % 50 == 0:
                    sink.flush()
                    print(f"  {n}/{len(todo)}")
    print("probe extract done")
    return 0


def _tv(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _verdict(ok: bool | None) -> str:
    return "UNSCOREABLE (no rows)" if ok is None else ("HELD" if ok else "FALSIFIED")


def cmd_report(_args) -> int:
    res: dict = {"strata": STRATA, "stratum_source": "judge committed dist"}
    lines = ["# Per-option probe — REPORT", "",
             "Scores the pre-registered Q1–Q4 (`docs/per-option-probe-preregistration.md`).",
             "Stratum defined by the judge's committed distribution (harness pin, §doc).",
             "",
             "Each question carries its registered verdict below — the prereg commits to "
             "reporting outcomes \"right or wrong\", which a table of measurements without "
             "verdicts does not do (audit wave 5).", ""]
    for pool in POOLS:
        tx = _transcripts(pool)
        ids = sample(pool)
        sup_of = {}
        judge_committed = _committed(pool, "judge")
        for rid in ids:
            sup_of[rid] = sum(1 for v in judge_committed[rid]["dist"].values() if v > 0.0)
        pool_res: dict = {}
        for tag in TAGS:
            path = OUT / f"{pool}-peropt-{tag}.jsonl"
            if not path.exists():
                lines.append(f"## {pool}/{tag}: NOT EXTRACTED YET")
                continue
            allrows = [json.loads(x) for x in path.open() if x.strip()]
            unscoreable = [r for r in allrows if r.get("unscoreable")]
            probe = {r["id"]: r for r in allrows if not r.get("unscoreable")}
            committed = _committed(pool, tag)
            rows = []
            for rid in ids:
                if rid not in probe:
                    continue
                gold = tx[rid]["gold"]
                c_dist, p_dist = committed[rid]["dist"], probe[rid]["dist"]
                c_sup = sum(1 for v in c_dist.values() if v > 0.0)
                rows.append({
                    "id": rid, "stratum": sup_of[rid], "own_support": c_sup,
                    "tv": _tv(c_dist, p_dist),
                    "gold_score_committed": 1.0 - c_dist.get(gold, 0.0),
                    "gold_score_peropt": 1.0 - p_dist.get(gold, 0.0),
                    "argmax_flips": max(c_dist, key=c_dist.get) != probe[rid]["choice"],
                    "mass_unnormalized": probe[rid]["mass_unnormalized"],
                    "surviving_is_gold": c_sup == 1 and c_dist.get(gold, 0.0) == 1.0,
                })
            n = len(rows)
            s1 = [r for r in rows if r["own_support"] == 1]
            s1_gold = [r for r in s1 if r["surviving_is_gold"]]
            q1_hits = sum(1 for r in s1_gold if r["gold_score_peropt"] > 0.0)
            def med(xs):
                """Lower median on even n — stated because Q2 compares two medians
                and an upper/lower convention must not be left implicit."""
                return sorted(xs)[(len(xs) - 1) // 2] if xs else float("nan")
            tv_s1 = med([r["tv"] for r in rows if r["own_support"] == 1])
            tv_s3 = med([r["tv"] for r in rows if r["own_support"] == 3])
            flips = sum(r["argmax_flips"] for r in rows)
            low_mass = sum(1 for r in rows if r["mass_unnormalized"] < 0.5)
            pool_res[tag] = {
                "n": n, "n_unscoreable": len(unscoreable), "n_support1_own": len(s1), "n_s1_gold": len(s1_gold),
                "q1_gold_leaves_zero": {"num": q1_hits, "den": len(s1_gold),
                                        "rate": q1_hits / len(s1_gold) if s1_gold else None},
                "q2_median_tv_s1": tv_s1, "q2_median_tv_s3": tv_s3,
                "q3_argmax_flip": {"num": flips, "den": n, "rate": flips / n if n else None},
                "mass_below_half": {"num": low_mass, "den": n},
                "rows": rows,
            }
            # The registered falsifiers, scored (wave-5 F10):
            #   Q1 > 0 in >= 95% of support-1-gold rows
            #   Q2 median TV(support-1) >= 0.05 AND > median TV(support-3)
            #   Q3 argmax flips on < 10% of rows
            q1_ok = (q1_hits / len(s1_gold) >= 0.95) if s1_gold else None
            q2_ok = ((tv_s1 >= 0.05 and tv_s1 > tv_s3)
                     if not (tv_s1 != tv_s1 or tv_s3 != tv_s3) else None)
            q3_ok = (flips / n < 0.10) if n else None
            pool_res[tag]["verdicts"] = {"Q1": q1_ok, "Q2": q2_ok, "Q3": q3_ok}
            lines += [f"## {pool}/{tag} (n={n})", "",
                      f"- **Q1 — {_verdict(q1_ok)}**: gold score leaves exactly-zero on "
                      f"{q1_hits}/{len(s1_gold)} support-1-gold rows "
                      f"(registered: > 0 in ≥ 95%)",
                      f"- **Q2 — {_verdict(q2_ok)}**: median TV support-1 {tv_s1:.4f} vs "
                      f"support-3 {tv_s3:.4f} (registered: ≥ 0.05 and support-1 > support-3)",
                      f"- **Q3 — {_verdict(q3_ok)}**: argmax flips {flips}/{n} "
                      f"(registered: < 10%)",
                      f"- mass_unnormalized < 0.5: {low_mass}/{n}",
                      f"- UNSCOREABLE (an option absent even at top-100, so "
                      f"refused rather than zeroed): {len(unscoreable)}", ""]
        # Q4: min-combined gold scores over probe rows only, threshold at alpha=0.1
        if all((OUT / f"{pool}-peropt-{t}.jsonl").exists() for t in TAGS):
            probes = {t: {json.loads(x)["id"]: json.loads(x)
                          for x in (OUT / f"{pool}-peropt-{t}.jsonl").open()} for t in TAGS}
            both = [rid for rid in ids if all(rid in probes[t] for t in TAGS)]
            scores = [combine_min({t: probes[t][rid]["dist"] for t in TAGS}, TAGS)[tx[rid]["gold"]]
                      for rid in both]
            thr = calibrate_threshold(scores, 0.1) if scores else None
            zero_share = sum(1 for s in scores if s == 0.0) / len(scores) if scores else None
            q4_ok = (thr is not None and thr > 0.0)
            pool_res["q4"] = {"n": len(both), "threshold_alpha_0.1": thr,
                              "zero_score_share": zero_share, "verdict": q4_ok}
            lines += [f"**Q4 — {_verdict(q4_ok)}** ({pool}, illustration at "
                      f"n={len(both)}): α=0.1 threshold = {thr} (registered: strictly "
                      f"positive), exactly-zero share = {zero_share:.3f}", ""]
        res[pool] = pool_res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(res, indent=2))
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'probe.json'} and {OUT/'REPORT.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--endpoint", required=True)
    e.add_argument("--tag", required=True, choices=TAGS)
    e.add_argument("--top-k", type=int, default=100, dest="top_k")
    sub.add_parser("report")
    args = ap.parse_args()
    return {"extract": cmd_extract, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
