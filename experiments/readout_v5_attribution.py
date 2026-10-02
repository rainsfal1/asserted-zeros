"""V5/F2: replay the OLD readout on a persisted window, and factor the routes.

With the raw top-k window persisted (`s3m3_measured.py`), the old readout is no
longer something we reason about — it is something we can **re-run**. Take the
fresh top-100 window, cut it to the top 20 by rank, aggregate with the OLD
normalizer, renormalize, round to five decimals: that is the v1 readout, exactly.

Three factors could each erase an option's probability, and the replay toggles
each independently: the top-20 RANK CUT, the old NORMALIZER (`strip().lower()`,
blind to "(a)" and "a."), and the five-decimal WRITER.

Two questions come apart, and only the second is causal:

    DIRECT          which factor erased a given zero FIRST. Sums to the zero
                    count, but flatters whichever factor is tested earliest.
    COUNTERFACTUAL  whether repairing that factor ALONE would have removed the
                    zero. Does not sum: a zero can be removable by more than one
                    repair, or -- as with the rank cut here -- by none.

The counterfactual column is the finding. Repairing the window removes zero of
7,861, because the writer would have erased those options anyway; the mechanism
originally published is not secondary but entirely preempted.

CAVEAT, stated because it bounds every number here: the window is from a fresh
server instance, and instances differ (C65: 77% bit-identical, worst 1.27e-2).
Reproduction of the committed record is therefore approximate, and the
`reproduces_committed` rate below is the honest measure of how far this replay
can be trusted.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verdict_cert.extraction import _norm_token  # noqa: E402

OPTIONS = ("a", "b", "c")
OUT = REPO / "results" / "readout_validation"


def old_norm(tok: str) -> str:
    return tok.strip().lower()


def replay(window, *, k: int, norm, round5: bool) -> dict[str, float] | None:
    """The v1 readout, parameterized. None if no option survives (v1's 2nd path)."""
    cut = window[:k] if k else window
    mass: dict[str, float] = {}
    for tok, lp in cut:
        key = norm(tok)
        if key in OPTIONS:
            mass[key] = mass.get(key, 0.0) + math.exp(lp)
    z = sum(mass.values())
    if z <= 0:
        return None
    dist = {o: mass.get(o, 0.0) / z for o in OPTIONS}
    return {o: round(v, 5) for o, v in dist.items()} if round5 else dist


def zeros_of(d: dict[str, float]) -> frozenset:
    return frozenset(o for o, v in d.items() if v == 0.0)


def main() -> int:
    src = REPO / "results" / "s3m3_measured"
    rows, agg = [], Counter()
    for pool in ("s3", "m3"):
        for tag in ("judge", "evaluator"):
            f = src / f"{pool}-{tag}.jsonl"
            if not f.exists():
                print(f"  (skip {pool}/{tag}: not extracted yet)")
                continue
            old = {json.loads(x)["id"]: json.loads(x)
                   for x in (REPO / f"results/{pool}/extract-{tag}.jsonl").open() if x.strip()}
            n = repro = 0
            routes = Counter()
            for line in f.open():
                r = json.loads(line)
                w = [(t, lp) for t, lp in r["window"]]
                w.sort(key=lambda p: -p[1])          # rank order, defensively
                committed = old[r["id"]]["dist"]
                if not zeros_of(committed):
                    continue
                n += 1
                v1 = replay(w, k=20, norm=old_norm, round5=True)
                if v1 is not None and zeros_of(v1) == zeros_of(committed):
                    repro += 1
                # --- factor the routes for each committed zero ---
                # BUG FIXED (audit wave 10): the normalizer branch used to test
                # the old normalizer over the TOP-100. v1 only ever saw the
                # top-20, so any option whose old-visible spelling sat at rank
                # 21-100 made that test pass and its zero was misfiled as
                # "writer". That understated the normalizer route by 97 zeros
                # and made it look empty. Every branch now uses v1's own window.
                cut_new = replay(w, k=20, norm=_norm_token, round5=False)
                cut_old = replay(w, k=20, norm=old_norm, round5=False)
                full_old = replay(w, k=None, norm=old_norm, round5=False)
                for o in zeros_of(committed):
                    # DIRECT: what erased this zero first, inside v1's window
                    if cut_new is None or cut_new.get(o, 0.0) == 0.0:
                        routes["direct: window (absent from top-20)"] += 1
                    elif cut_old is None or cut_old.get(o, 0.0) == 0.0:
                        routes["direct: normalizer (spelling unmatched)"] += 1
                    elif round(cut_old[o], 5) == 0.0:
                        routes["direct: writer (rounded at 5dp)"] += 1
                    else:
                        routes["direct: unexplained"] += 1
                    # COUNTERFACTUAL: does repairing ONE factor alone remove it?
                    # This is the causal column: the direct one only says which
                    # factor got there first, which flatters whichever is tested
                    # earliest.
                    if cut_old is not None and 0.0 < cut_old.get(o, 0.0) \
                            and round(cut_old[o], 5) == 0.0:
                        routes["counterfactual: writer alone"] += 1
                    if round(cut_new.get(o, 0.0) if cut_new else 0.0, 5) > 0.0:
                        routes["counterfactual: normalizer alone"] += 1
                    if round(full_old.get(o, 0.0) if full_old else 0.0, 5) > 0.0:
                        routes["counterfactual: window alone"] += 1
            print(f"  {pool}/{tag}: {n} records with a committed zero; "
                  f"v1 replay reproduces the zero-set on {repro} ({repro/n:.1%})" if n else
                  f"  {pool}/{tag}: no records with committed zeros")
            for kk, v in routes.most_common():
                print(f"      {kk:<38} {v}")
            agg.update(routes)
            rows.append({"pool": pool, "tag": tag, "n_records": n,
                         "replay_reproduces": repro, "routes": dict(routes)})

    # The two families count the SAME zeros and must not be pooled: direct sums
    # to the zero count, counterfactual does not (a zero can be removable by
    # more than one repair, or by none).
    direct = {k: v for k, v in agg.items() if k.startswith("direct")}
    cfac = {k: v for k, v in agg.items() if k.startswith("counterfactual")}
    n_zeros = sum(direct.values())
    print(f"\n=== {n_zeros} committed zeros ===")
    print("  DIRECT (which factor erased it first, inside v1's own top-20):")
    for k, v in sorted(direct.items(), key=lambda kv: -kv[1]):
        print(f"    {k.split(': ', 1)[1]:<34} {v:>6}  ({v / n_zeros:.2%})")
    print("  COUNTERFACTUAL (repairing that factor ALONE removes it) -- the causal column:")
    for k, v in sorted(cfac.items(), key=lambda kv: -kv[1]):
        print(f"    {k.split(': ', 1)[1]:<34} {v:>6}  ({v / n_zeros:.2%})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v5_attribution.json").write_text(json.dumps(
        {"per_file": rows, "direct": direct, "counterfactual": cfac,
         "n_zeros": n_zeros}, indent=1))
    print(f"\nwrote {OUT / 'v5_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
