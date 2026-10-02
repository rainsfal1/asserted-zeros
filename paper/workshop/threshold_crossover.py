"""Where does the calibrated threshold actually enter the zero pile?

Section 2 said "when ~90% of those scores sit at exactly 0.0, the threshold is
0.0 for every alpha >= 0.1". An external referee recomputed the order statistic
and found that false on the second corpus, and they were right: the threshold is
the ceil((n+1)(1-alpha))-th smallest score, so it lands in the pile only once
the zero count reaches it. At 89.7% zeros that needs alpha > 0.1, not >= 0.1.

This computes the exact crossover on both committed corpora, and the band where
section 5's error indicator ("both judges below 0.5") and the tau = 1e-2 floor
("both judges below 1e-2") can disagree -- the figure axis assumes they coincide.

Reads committed artifacts only.
"""

from __future__ import annotations

import importlib.util as ilu
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

TAGS = ("judge", "evaluator")


def true_option_scores(corpus: str) -> list[float]:
    bank = {r["id"]: r["gold"] for r in
            (json.loads(x) for x in (REPO / f"results/{corpus}/transcripts.jsonl").open() if x.strip())
            if r.get("kind") == "bank"}
    ex = {t: {json.loads(x)["id"]: json.loads(x)["dist"]
              for x in (REPO / f"results/{corpus}/extract-{t}.jsonl").open() if x.strip()}
          for t in TAGS}
    # NOT .get(gold, 0.0): a missing key means the cache is wrong, and defaulting
    # it to zero is the exact equation this paper is about.
    return [min(1.0 - ex[t][rid][g] for t in TAGS) for rid, g in bank.items()]


def crossover(scores: list[float]) -> dict:
    s = sorted(scores)
    n = len(s)
    zeros = sum(1 for x in s if x == 0.0)
    lo = None
    for i in range(1, 1000):                      # 0.001 grid is finer than we quote
        a = i / 1000
        if math.ceil((n + 1) * (1 - a)) <= zeros:
            lo = a
            break
    at_10 = s[math.ceil((n + 1) * 0.9) - 1]
    return {"n": n, "zeros": zeros, "zero_share": zeros / n,
            "alpha_crossover": lo, "threshold_at_alpha_0.10": at_10}


def indicator_band() -> dict:
    """Records where 'both judges below 0.5' and 'below tau=1e-2' disagree."""
    _s = ilu.spec_from_file_location("sw", HERE / "sweep_honest_budget.py")
    sw = ilu.module_from_spec(_s)
    _s.loader.exec_module(sw)
    rows, _items = sw.load_per_extractor()
    band = sum(1 for c in rows.values()
               if 1e-2 <= max(p[c["gold"]] for p in c["p"]) < 0.5)
    return {"records": len(rows), "in_band": band, "share": band / len(rows)}


def main() -> int:
    out = {"corpora": {c: crossover(true_option_scores(c)) for c in ("s3", "m3")},
           "indicator_band_tau_1e-2_vs_0.5": indicator_band()}
    for c, r in out["corpora"].items():
        print(f"{c}: n={r['n']} zeros={r['zeros']} ({r['zero_share']:.4f}) "
              f"-> threshold is 0.0 for alpha >= {r['alpha_crossover']:.3f}; "
              f"at alpha=0.10 threshold = {r['threshold_at_alpha_0.10']:.1e}")
    b = out["indicator_band_tau_1e-2_vs_0.5"]
    print(f"indicator band: {b['in_band']} of {b['records']} records ({b['share']:.4%})")
    (HERE / "threshold_crossover.json").write_text(json.dumps(out, indent=1))
    print("wrote threshold_crossover.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
