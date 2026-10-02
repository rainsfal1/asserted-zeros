"""
M6 — the certificate on a SECOND, EXTERNAL decision function.

Everything through M3 runs on our own 11-item scorer: we wrote the items, the
point map and the bands. A reviewer can reasonably ask whether the aggregation
rule was shaped to flatter the method. M6 answers that by running the identical
machinery on a decision function we did not write and cannot tune:

    the USCIS naturalization civics test — pass iff >= 6 of 10 items correct.

The items are literal USCIS questions, the acceptable-answer key is USCIS's own,
and the pass mark is federal policy (8 CFR 312.1 / the 2008 civics test as
administered). The responses are 698 real crowdworker answers per item, graded
by three independent human graders (Basu, Jacobs & Vanderwende, TACL 2013).

Why this benchmark and not ASAP-SAS: ASAP would have required *inventing* an
aggregation rule over ten unrelated prompts never administered together, which
is precisely the criticism this experiment exists to defuse.

WHAT IS AND IS NOT VALIDATED HERE
---------------------------------
The certificate's calibration source is the *extractor re-run* reading: the
captured response is fixed, and the uncertainty being calibrated is the grader's
over how to read it. That is what Powergrading supports — it has exactly one
response per (student, question), so a *human* re-ask cannot be validated on it
and is not claimed. The recovery loop is therefore not exercised here; M6 tests
the gate.

THE GUARANTEE IS NON-VACUOUS HERE, WHICH IS THE POINT
-----------------------------------------------------
Split conformal on n calibration units per item cannot promise miscoverage below
1/(n+1); with k items the verdict guarantee Sigma_i alpha_i inherits a floor of
Sigma_i 1/(n_i+1). On our own S3 pool (n=7, k=11) that floor is 1.375 — a 0%
bound, i.e. no guarantee at all. Here it is ~0.04-0.07 depending on whether you
count calibration rows or *distinct* responses, so a 90% verdict guarantee is
attainable rather than arithmetically impossible. Both readings are reported.

Run:
    python experiments/m6_powergrading.py --out results/m6
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from _legacy import build_set  # noqa: E402  (quarantined legacy builder, F-3; name kept so metric_suite_run's m6.build_set resolves)
from verdict_cert.certificate import calibrate_threshold  # noqa: E402

DATA = Path.home() / "vc-lab" / "data" / "powergrading"
ANSWERS_TSV = DATA / "studentanswers_grades_698.tsv"
KEY_TSV = DATA / "questions_answer_key.tsv"

# The ten items carrying three-grader gold. The other ten questions in the corpus
# hold the -1 sentinel (ungraded), so k=10 is a property of the data, not a choice.
GOLD_ITEMS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "13", "20")
PASS_MARK = 6  # USCIS: 6 of 10 correct.

# Items whose three graders agree only moderately (Fleiss' kappa < 0.60), computed
# in research/venues-and-benchmarks.md 2.2. Dropped only in the sensitivity arm --
# the primary arm keeps all ten so the decision function stays the real one.
LOW_KAPPA: frozenset[str] = frozenset({"3", "20"})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_rows() -> list[dict]:
    """(student, item, answer, gold) for the ten graded items."""
    if not ANSWERS_TSV.exists():
        raise SystemExit(
            f"missing {ANSWERS_TSV}\n"
            "Fetch Powergrading-1.0-Corpus.zip from the Microsoft Download Center "
            "(id=52397) and unpack it there; sha256 "
            "c1e0e3912be8b25357b5b99fe26263ed1c4f233f2a355949a359ca5058e15293"
        )
    rows: list[dict] = []
    with ANSWERS_TSV.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            item = f[idx["Q#"]].strip()
            if item not in GOLD_ITEMS:
                continue
            grades = [int(f[idx[g]]) for g in ("G1", "G2", "G3")]
            if any(g < 0 for g in grades):  # ungraded sentinel
                continue
            rows.append({
                "student": f[idx["student"]].strip(),
                "item": item,
                "answer": f[idx["answer"]],
                # gold = majority of three independent graders
                "gold": 1 if sum(grades) >= 2 else 0,
                "grades": grades,
            })
    return rows


def load_key() -> dict[str, list[str]]:
    """USCIS acceptable answers per item, from the corpus's own key."""
    key: dict[str, list[str]] = {}
    with KEY_TSV.open(encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            item = f[0].strip()
            if item in GOLD_ITEMS:
                key[item] = [a for a in (x.strip() for x in f[2:]) if a]
    return key


# ---------------------------------------------------------------------------
# The extractor: a grader that scores {correct, incorrect} for a captured answer
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")
_STOP = frozenset("the a an of to and or is are was were in on for i we you it that this".split())


def normalize(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()


def tokens(s: str) -> set[str]:
    return {t for t in normalize(s).split() if t and t not in _STOP}


def token_f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    return 2 * p * r / (p + r)


def features(answer: str, acceptable: Sequence[str]) -> list[float]:
    """Cheap, fully deterministic lexical features against the USCIS key."""
    ta = tokens(answer)
    best_f1 = 0.0
    covered = 0.0
    contains = 0.0
    na = normalize(answer)
    for acc in acceptable:
        tb = tokens(acc)
        best_f1 = max(best_f1, token_f1(ta, tb))
        if tb:
            covered = max(covered, len(ta & tb) / len(tb))  # recall of the key answer
        nacc = normalize(acc)
        if nacc and nacc in na:
            contains = 1.0
    return [best_f1, covered, contains, 1.0 if ta else 0.0]


def fit_logistic(X: list[list[float]], y: list[int], *, steps: int = 4000, lr: float = 0.5) -> list[float]:
    """Tiny batch-GD logistic regression (no sklearn dependency, fully reproducible)."""
    d = len(X[0]) + 1
    w = [0.0] * d
    n = len(X)
    rows = [[1.0] + x for x in X]
    for _ in range(steps):
        grad = [0.0] * d
        for xi, yi in zip(rows, y):
            z = sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + pow(2.718281828459045, -max(-30.0, min(30.0, z))))
            e = p - yi
            for j in range(d):
                grad[j] += e * xi[j]
        for j in range(d):
            w[j] -= lr * grad[j] / n
    return w


def predict_p(w: Sequence[float], x: Sequence[float]) -> float:
    z = w[0] + sum(wj * xj for wj, xj in zip(w[1:], x))
    return 1.0 / (1.0 + pow(2.718281828459045, -max(-30.0, min(30.0, z))))


# ---------------------------------------------------------------------------
# Reachability over the EXTERNAL decision function
# ---------------------------------------------------------------------------


def reachable_verdicts_uscis(sets: Mapping[str, frozenset[int]], *, pass_mark: int = PASS_MARK) -> set[str]:
    """
    EXACT set of verdicts reachable over the product of the per-item sets.

    Each S_i subseteq {0,1}. The rule counts correct answers and thresholds, so the
    reachable count spans [sum min S_i, sum max S_i] and every intermediate value is
    attainable (items are independent). Hence the O(k) form below -- which
    `_brute_force_reachable` re-derives by enumerating the whole product set.
    """
    lo = sum(min(s) for s in sets.values())
    hi = sum(max(s) for s in sets.values())
    out: set[str] = set()
    if hi >= pass_mark:
        out.add("pass")
    if lo < pass_mark:
        out.add("fail")
    return out


def _brute_force_reachable(sets: Mapping[str, frozenset[int]], *, pass_mark: int = PASS_MARK) -> set[str]:
    keys = sorted(sets)
    return {
        "pass" if sum(combo) >= pass_mark else "fail"
        for combo in itertools.product(*(sorted(sets[k]) for k in keys))
    }


def verdict_of(correct: Mapping[str, int], *, pass_mark: int = PASS_MARK) -> str:
    return "pass" if sum(correct.values()) >= pass_mark else "fail"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def split_students(students: Sequence[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    """
    Deterministic 3-way split: train (fit the grader), cal (conformal), test.

    The grader must NOT be fitted on the conformal calibration split -- reusing it
    breaks the exchangeability that the coverage guarantee rests on.
    """
    order = sorted(students, key=lambda s: (hash_str(f"{seed}:{s}"), s))
    n = len(order)
    a, b = n // 3, 2 * n // 3
    return order[:a], order[a:b], order[b:]


def hash_str(s: str) -> int:
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_arm(rows: list[dict], items: Sequence[str], key: Mapping[str, list[str]],
            alphas: Sequence[float], seed: int, pass_mark: int) -> list[dict]:
    """One risk-coverage sweep: for each total Sigma alpha, gate and measure."""
    by_student: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["item"] in items:
            by_student[r["student"]][r["item"]] = r
    # keep only students answering every item in this arm
    students = [s for s, d in by_student.items() if len(d) == len(items)]
    train, cal, test = split_students(students, seed)

    # --- fit the grader per item on the TRAIN split only
    weights: dict[str, list[float]] = {}
    for it in items:
        X = [features(by_student[s][it]["answer"], key.get(it, [])) for s in train]
        y = [by_student[s][it]["gold"] for s in train]
        weights[it] = fit_logistic(X, y)

    def score(student: str, it: str) -> dict[int, float]:
        p = predict_p(weights[it], features(by_student[student][it]["answer"], key.get(it, [])))
        return {1: 1.0 - p, 0: p}  # nonconformity: lower = more plausible

    # --- conformal calibration per item on the CAL split
    noncon: dict[str, list[float]] = {}
    for it in items:
        noncon[it] = [score(s, it)[by_student[s][it]["gold"]] for s in cal]

    out: list[dict] = []
    for total_alpha in alphas:
        a_i = total_alpha / len(items)
        thr = {it: calibrate_threshold(noncon[it], a_i) for it in items}
        released = certified_wrong = withheld = 0
        cov_hits = cov_n = 0
        setsize = 0
        for s in test:
            sets = {it: build_set(score(s, it), thr[it]) for it in items}
            truth = {it: by_student[s][it]["gold"] for it in items}
            for it in items:
                cov_n += 1
                cov_hits += int(truth[it] in sets[it])
                setsize += len(sets[it])
            reach = reachable_verdicts_uscis(sets, pass_mark=pass_mark)
            if len(reach) == 1:
                released += 1
                if next(iter(reach)) != verdict_of(truth, pass_mark=pass_mark):
                    certified_wrong += 1
            else:
                withheld += 1
        n_test = len(test)
        out.append({
            "sigma_alpha": round(total_alpha, 6),
            "alpha_i": round(a_i, 6),
            "k": len(items),
            "n_test": n_test,
            "released": released,
            "withheld": withheld,
            "release_rate": released / n_test if n_test else 0.0,
            "certified_wrong": certified_wrong,
            "verdict_error_released": certified_wrong / released if released else 0.0,
            "item_coverage": cov_hits / cov_n if cov_n else 0.0,
            "mean_set_size": setsize / cov_n if cov_n else 0.0,
            "n_train": len(train), "n_cal": len(cal),
        })
    return out


def floor_report(rows: list[dict], items: Sequence[str], n_cal: int, seed: int = 0) -> dict:
    """The Sigma alpha floor. `floor_by_cal_rows` is the headline and the correct one.

    CORRECTED 2026-08-15 (audit wave 2). The previous headline, `floor_by_distinct`,
    counted distinct responses over the WHOLE corpus (698 students) while the conformal
    quantile is computed on the CAL split alone (233). That mixed populations and was not
    even conservative in one direction: for three items the corpus-wide distinct count
    EXCEEDS n_cal, making those per-item terms smaller than the true floor. It is replaced
    by `floor_by_distinct_cal`, computed on the same split the quantile uses.

    Why `floor_by_cal_rows` is the right headline here, and the distinct reading is not:
    the exchangeability unit in M6 is the STUDENT. Each student answers each item exactly
    once, so two students who happen to write the same words are still two independent
    draws -- duplicates are ties in the score, not broken exchangeability. The distinct
    -wordings discipline that governs S3/M3 exists there because ONE utterance is
    re-rendered through 10 channel conditions, which is repeated measurement of a single
    unit. That does not apply to a one-response-per-student corpus.

    Empirically confirmed: the release cliff lands exactly on k/(n_cal+1) -- 0 of 233
    released at Sigma alpha = 0.0427, 55 of 233 at 0.0428.
    """
    by_student: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["item"] in items:
            by_student[r["student"]][r["item"]] = r
    students = [st for st, d in by_student.items() if len(d) == len(items)]
    _, cal, _ = split_students(students, seed)
    cal_set = set(cal)

    distinct_cal: dict[str, int] = {}
    for it in items:
        distinct_cal[it] = len({normalize(by_student[st][it]["answer"]) for st in cal_set})
    return {
        "k": len(items),
        "floor_by_cal_rows": round(len(items) / (n_cal + 1), 6),
        "floor_by_distinct_cal": round(sum(1.0 / (distinct_cal[it] + 1) for it in items), 6),
        "distinct_per_item_cal": distinct_cal,
        "n_cal": n_cal,
        "floor_seed": seed,
    }


def reask_degeneracy(items: Sequence[str], pass_mark: int) -> dict:
    """
    Does re-ask ranking buy anything under a plain counting rule?

    Our own scorer weights items unequally, so `reask_ranking` has real signal. A
    >=6-of-10 count is symmetric in the items, so resolving ANY unresolved item moves
    the reachable span by exactly one. We check that rather than assume it.
    """
    import random

    rng = random.Random(6)
    profiles = 0
    varied = 0            # profiles where two unresolved items differ in gain
    any_positive = 0      # profiles where re-asking one item can resolve the verdict
    for _ in range(2000):
        sets = {it: frozenset(rng.choice([{0}, {1}, {0, 1}])) for it in items}
        if len(reachable_verdicts_uscis(sets, pass_mark=pass_mark)) == 1:
            continue  # already certifiable; nothing to re-ask
        unresolved = [it for it in items if len(sets[it]) > 1]
        if len(unresolved) < 2:
            continue
        base = len(reachable_verdicts_uscis(sets, pass_mark=pass_mark))
        gains = [
            base - min(
                len(reachable_verdicts_uscis({**sets, it: frozenset({v})}, pass_mark=pass_mark))
                for v in (0, 1)
            )
            for it in unresolved
        ]
        profiles += 1
        varied += int(len(set(gains)) > 1)
        any_positive += int(max(gains) > 0)
    return {
        "profiles_tested": profiles,
        "profiles_with_differing_gains": varied,
        "profiles_where_one_reask_can_resolve": any_positive,
        "degenerate": profiles > 0 and varied == 0,
    }


def self_check(items: Sequence[str], pass_mark: int) -> None:
    """The O(k) reachability must equal brute-force enumeration of the product set."""
    import random

    rng = random.Random(20260809)
    for _ in range(300):
        sets = {it: frozenset(rng.choice([{0}, {1}, {0, 1}])) for it in items}
        fast = reachable_verdicts_uscis(sets, pass_mark=pass_mark)
        slow = _brute_force_reachable(sets, pass_mark=pass_mark)
        assert fast == slow, (sets, fast, slow)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m6")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pass-mark", type=int, default=PASS_MARK)
    args = ap.parse_args()

    items = list(GOLD_ITEMS)
    self_check(items, args.pass_mark)

    rows = load_rows()
    key = load_key()
    alphas = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    arms: dict[str, list[str]] = {
        "all gold items (k=10, real USCIS rule)": items,
        "kappa >= 0.60 sensitivity (k=8)": [it for it in items if it not in LOW_KAPPA],
    }

    res: dict[str, dict] = {}
    for name, arm_items in arms.items():
        # pass mark scales with k for the sensitivity arm, else the rule is not comparable
        pm = args.pass_mark if len(arm_items) == len(items) else round(
            args.pass_mark * len(arm_items) / len(items))
        per_seed = [run_arm(rows, arm_items, key, alphas, s, pm) for s in range(args.seeds)]
        agg = []
        for i, a in enumerate(alphas):
            agg.append({
                "sigma_alpha": a,
                "release_rate": statistics.mean(p[i]["release_rate"] for p in per_seed),
                "verdict_error_released": statistics.mean(p[i]["verdict_error_released"] for p in per_seed),
                "certified_wrong": statistics.mean(p[i]["certified_wrong"] for p in per_seed),
                "item_coverage": statistics.mean(p[i]["item_coverage"] for p in per_seed),
                "mean_set_size": statistics.mean(p[i]["mean_set_size"] for p in per_seed),
                "n_test": per_seed[0][i]["n_test"],
            })
        res[name] = {
            "pass_mark": pm, "k": len(arm_items), "items": arm_items,
            "sweep": agg, "per_seed": per_seed,
            "floor": floor_report(rows, arm_items, per_seed[0][0]["n_cal"], seed=0),
            "reask": reask_degeneracy(arm_items, pm),
        }

    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "m6.json").write_text(json.dumps(res, indent=2))
    (outdir / "REPORT.md").write_text(render(res, args.seeds))
    print(f"wrote {outdir/'m6.json'} and {outdir/'REPORT.md'}")
    for name, r in res.items():
        print(f"\n{name}: pass mark {r['pass_mark']}/{r['k']}, "
              f"floor {r['floor']['floor_by_cal_rows']:.4f} (cal rows) / "
              f"{r['floor']['floor_by_distinct_cal']:.4f} (distinct within cal)")
        print(f"  {'Sigma a':>8} {'release':>8} {'err|rel':>8} {'item cov':>9} {'set sz':>7}")
        for s in r["sweep"]:
            print(f"  {s['sigma_alpha']:>8.2f} {s['release_rate']:>8.3f} "
                  f"{s['verdict_error_released']:>8.4f} {s['item_coverage']:>9.4f} "
                  f"{s['mean_set_size']:>7.3f}")


def render(res: dict, seeds: int) -> str:
    """Every sentence below is derived from the numbers, per the M2 report defect."""
    L: list[str] = []
    L.append("# M6 — the certificate on an external decision function (USCIS civics)\n")
    L.append(f"Powergrading 1.0 (Basu, Jacobs & Vanderwende, TACL 2013). {seeds} seeds, "
             "3-way student split: grader fitted on train, conformal calibrated on cal, "
             "measured on test.\n")
    L.append("The decision function is **not ours**: 10 USCIS civics items, USCIS's own "
             "acceptable-answer key, federal pass mark of 6/10.\n")

    for name, r in res.items():
        f = r["floor"]
        L.append(f"\n## {name}\n")
        L.append(f"Pass mark {r['pass_mark']} of {r['k']}. "
                 f"n_cal = {f['n_cal']} students per item.\n")
        floor = f["floor_by_cal_rows"]
        L.append(f"**Σᵢαᵢ floor:** {floor:.4f} = k/(n_cal+1), the split-conformal floor on the "
                 f"calibration split. ")
        L.append(f"A 90% verdict guarantee needs Σᵢαᵢ ≤ 0.10, so the floor "
                 f"{'clears' if floor <= 0.10 else 'does NOT clear'} it "
                 f"({floor:.4f} vs 0.10) — "
                 f"{'non-vacuous' if floor <= 0.10 else 'VACUOUS'}, against 1.375 on the "
                 "S3 pool.\n")
        L.append(f"\n*The exchangeability unit here is the **student**: each answers each item "
                 f"once, so duplicate wordings are ties in the score, not repeated measurement "
                 f"of one unit. The distinct-response reading that governs S3/M3 therefore does "
                 f"not transfer; computed on this split it would give "
                 f"{f['floor_by_distinct_cal']:.4f}. A prior version of this report quoted a "
                 f"distinct count taken over the whole corpus, which mixed populations — "
                 f"corrected 2026-08-15, audit wave 2.*\n")
        L.append("\n| Σαᵢ | release rate | verdict error \\| released | item coverage | mean set size |")
        L.append("|---|---|---|---|---|")
        for s in r["sweep"]:
            L.append(f"| {s['sigma_alpha']:.2f} | {s['release_rate']:.3f} | "
                     f"{s['verdict_error_released']:.4f} | {s['item_coverage']:.4f} | "
                     f"{s['mean_set_size']:.3f} |")

        # --- data-derived prose. No claim here is written in advance.
        sw = r["sweep"]
        at10 = min(sw, key=lambda s: abs(s["sigma_alpha"] - 0.10))
        # A point that releases NOTHING cannot test the bound: verdict_error_released is
        # DEFINED as 0.0 there, so it would silently count as a point where the bound held.
        live = [s for s in sw if s["release_rate"] > 0.0]
        vacuous = len(sw) - len(live)
        broke = [s for s in live if s["verdict_error_released"] > s["sigma_alpha"]]
        L.append(f"\nAt Σαᵢ = {at10['sigma_alpha']:.2f} the gate releases "
                 f"{at10['release_rate']:.1%} of sessions with a measured verdict error of "
                 f"{at10['verdict_error_released']:.2%} among those released.")
        if broke:
            L.append(f" The guarantee is **violated at {len(broke)} of {len(sw)} operating "
                     f"points** (" + ", ".join(f"Σα={s['sigma_alpha']:.2f}: "
                     f"{s['verdict_error_released']:.2%}" for s in broke) + "), which is a "
                     "finding, not a footnote — see the caveat below.")
        else:
            L.append(f" Measured error stays at or below Σαᵢ at **all {len(live)} operating "
                     f"points that release anything**, i.e. the bound holds everywhere it "
                     f"could be tested.")
            if vacuous:
                L.append(f" {vacuous} of the {len(sw)} swept points release nothing at all and "
                         f"are excluded: error is undefined there, and counting them would "
                         f"inflate the claim.")
        nominal = 1.0 - at10["sigma_alpha"] / r["k"]
        L.append(f" Item-level coverage at that point is {at10['item_coverage']:.2%} against a "
                 f"nominal {nominal:.2%} per item.")
        lo, hi = sw[0], sw[-1]
        L.append(f" Across the sweep, loosening Σαᵢ from {lo['sigma_alpha']:.2f} to "
                 f"{hi['sigma_alpha']:.2f} moves release from {lo['release_rate']:.1%} to "
                 f"{hi['release_rate']:.1%} and set size from {lo['mean_set_size']:.2f} to "
                 f"{hi['mean_set_size']:.2f}.\n")

        # --- the floor, observed rather than asserted
        dead = [s for s in sw if s["release_rate"] == 0.0]
        alive = [s for s in sw if s["release_rate"] > 0.0]
        if dead and alive:
            hi_dead, lo_alive = max(s["sigma_alpha"] for s in dead), min(s["sigma_alpha"] for s in alive)
            fl = f["floor_by_cal_rows"]
            if hi_dead < fl <= lo_alive:
                L.append(f"\n**The floor is not a footnote — it is visible in the sweep.** "
                         f"Release is exactly 0% at Σαᵢ = {hi_dead:.2f} and turns on at "
                         f"{lo_alive:.2f}; the calibration-row floor is k/(n_cal+1) = "
                         f"{fl:.4f}, which lies between them. Below the floor `calibrate_threshold` "
                         f"cannot reach the requested quantile, returns +∞, and every item set "
                         f"becomes {{correct, incorrect}} (mean set size "
                         f"{max(dead, key=lambda s: s['sigma_alpha'])['mean_set_size']:.2f} of 2) — "
                         "so the gate withholds everything. The arithmetic that makes the S3 pool "
                         "vacuous is the same arithmetic, observed here as a cliff.\n")

        rk = r["reask"]
        if rk["degenerate"]:
            L.append(f"\n**Re-ask ranking is degenerate on this decision function.** Across "
                     f"{rk['profiles_tested']} randomly drawn uncertain profiles, "
                     f"**{rk['profiles_with_differing_gains']}** had two unresolved items with "
                     f"different information gain — the ranking is flat every time (a single "
                     f"re-ask can still resolve the verdict, in "
                     f"{rk['profiles_where_one_reask_can_resolve']}/{rk['profiles_tested']} "
                     "profiles; what it cannot do is prefer an item). The reachable set depends "
                     "on the profile only through (min count, max count), so resolving *any* "
                     "unresolved item moves it identically. Our own scorer "
                     "weights items unequally, which is where the ranking earns its keep; on a "
                     "counting rule, choosing *which* item to re-ask cannot beat choosing at "
                     "random. This bounds the generality of the re-ask contribution and should "
                     "be stated in the paper rather than discovered by a reviewer.\n")
        else:
            L.append(f"\n**Re-ask ranking is informative here:** gains {rk['gains']}.\n")

    L.append("\n## Caveats that belong next to these numbers\n")
    L.append("- **Coverage is of the graders, not of truth.** Gold is a majority of three human "
             "graders; on the two items with Fleiss' κ < 0.60 (Q3 = 0.574, Q20 = 0.449) the "
             "graders themselves disagree, so those αᵢ are bounded below by label noise this "
             "model does not represent. The κ ≥ 0.60 arm above is the sensitivity check.\n")
    L.append("- **The recovery loop is not exercised.** Powergrading has exactly one response "
             "per (student, item), so re-asking a *human* cannot be validated on it. M6 tests "
             "the gate; the loop is tested in M1.5/M2 and on the M3 channel.\n")
    L.append("- **The grader is lexical, not an LLM.** It is deterministic and reproducible "
             "with no GPU, which makes M6 a clean test of the *certificate*, not of an "
             "extractor. Swapping in an LLM grader changes the scores fed to `build_set` and "
             "nothing else in this file.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
