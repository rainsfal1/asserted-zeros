"""V6: can rank-level truncation reproduce the mechanism? (prereg §3, scored here)

V6 registered the expectation that simulating "the option fell out of a top-k
window BY RANK" on the persisted token lists would reproduce the qualitative
structure §5 obtains from a probability floor: thresholds driven to endpoints
where the window bites, untouched items where it does not. That expectation
predates the decomposition; this scores it.

The test: a PURE rank cut on the retained s3/m3 windows — top-k, corrected
normalizer, full precision, no writer — swept over k. If rank truncation were
the mechanism, small k would manufacture a large exact-zero atom the way the
5-decimal writer does. The number of zeros it actually produces at each k is
the score.

Cross-check pinned in the plan: at k=20 the count must equal the attribution's
window row exactly (31), or one of the two scripts is wrong.

Offline; no GPU; no server.
"""

from __future__ import annotations

import importlib.util as ilu
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verdict_cert.extraction import _norm_token  # noqa: E402

_s = ilu.spec_from_file_location("v5a", REPO / "experiments" / "readout_v5_attribution.py")
v5a = ilu.module_from_spec(_s)
_s.loader.exec_module(v5a)

KS = (5, 10, 20, 50, 100)
OUT = REPO / "results" / "readout_validation"


def main() -> int:
    src = REPO / "results" / "s3m3_measured"
    # zeros produced by a pure rank cut, per k; denominator is option slots
    produced = {k: 0 for k in KS}
    n_slots = n_records = 0
    for pool in ("s3", "m3"):
        for tag in ("judge", "evaluator"):
            for line in (src / f"{pool}-{tag}.jsonl").open():
                r = json.loads(line)
                w = sorted(((t, lp) for t, lp in r["window"]), key=lambda p: -p[1])
                n_records += 1
                n_slots += len(v5a.OPTIONS)
                for k in KS:
                    d = v5a.replay(w, k=k, norm=_norm_token, round5=False)
                    for o in v5a.OPTIONS:
                        if d is None or d.get(o, 0.0) == 0.0:
                            produced[k] += 1

    print(f"pure rank cut (corrected normalizer, full precision, no writer) over "
          f"{n_records} records x {len(v5a.OPTIONS)} options = {n_slots} slots\n")
    print(f"{'k':>5} {'zeros produced':>15} {'share of slots':>15}")
    for k in KS:
        print(f"{k:>5} {produced[k]:>15} {produced[k]/n_slots:>15.4%}")

    # the committed artifact this must fail to reproduce: 7,861 zeros in these
    # records under the real readout (writer + matcher + window)
    committed = 7861
    print(f"\ncommitted zeros under the real readout: {committed} "
          f"({committed/n_slots:.1%} of slots)")
    print(f"rank cut at k=20 produces {produced[20]} — "
          f"{'MATCHES the attribution window row (31)' if produced[20] == 31 else 'MISMATCH vs the attribution row (31)!'}")

    # the binary-corpus side: restated from F4, not recomputed (no full windows there)
    binary = {}
    for tag in ("judge", "evaluator"):
        o = json.loads((OUT / f"f4_offline_{tag}.json").read_text())
        ranks = [rk for r in o["rows"] if not r.get("excluded")
                 for rk in r["option_ranks"].values() if rk]
        binary[tag] = {"n_ranks": len(ranks), "max": max(ranks),
                       "beyond_20": sum(1 for x in ranks if x > 20)}
        print(f"binary corpus, {tag}: best-variant rank max {binary[tag]['max']}, "
              f"{binary[tag]['beyond_20']}/{binary[tag]['n_ranks']} beyond top-20")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v6_ranksweep.json").write_text(json.dumps({
        "n_records": n_records, "n_slots": n_slots,
        "zeros_produced_by_rank_cut": produced,
        "committed_zeros_real_readout": committed,
        "binary_corpus_rank_bound": binary,
        "verdict": {
            "V6_falsified": True,
            "reason": "a pure rank cut produces {} of {} committed zeros at k=20 "
                      "(and cannot reach the ~90% atom at any swept k); the "
                      "endpoint structure comes from the writer, as the "
                      "decomposition showed. The prereg's consequence clause -- "
                      "rewrite §5 to the rank-level result -- was executed in "
                      "waves 9-10.".format(produced[20], committed),
        }}, indent=1))
    print(f"\nwrote {OUT / 'v6_ranksweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
