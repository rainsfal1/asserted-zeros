"""Re-extract s3 + m3 under the corrected readout, persisting the raw window.

Design: ``docs/readout-validation-preregistration.md`` §2.2. One GPU pass buys
three answers that the committed caches cannot give:

**F3 — a like-for-like before/after.** The committed s3/m3 extractions used the
OLD readout; Figure 1 currently compares two different corpora with different
option arities. Re-extracting the SAME records with the corrected readout holds
records, options, models and conditions fixed and moves only the readout.

**F2 — rank-level truncation.** §5 simulates the defect with a probability floor
because the raw top-k was never persisted. With ``window`` on every row (added
to ``PerOptionExtraction`` in audit wave 8) a top-20 RANK cut is reconstructible.

**V5 — the normalizer attribution, and the 21 rows.** The old readout normalized
tokens with ``strip().lower()``; the corrected one also strips ``"(). "``. On the
judge those differ by 15 token forms. Re-aggregating one persisted window under
BOTH normalizers separates "the option left the window" from "the old normalizer
could not see the spelling it was under" — a third route that the current
analysis would misclassify as window absence, and one that lands on exactly the
20-of-21 judge rows carrying the paper's least secure number.

Serving is sequential, so run once per extractor:

    python experiments/s3m3_measured.py --endpoint http://localhost:8001/v1 --tag judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verdict_cert.extraction import Extractor  # noqa: E402
from verdict_cert.items import load_items  # noqa: E402

POOLS = ("s3", "m3")
OPTIONS = ("a", "b", "c")
TOP_K = 100
OUT = REPO / "results" / "s3m3_measured"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--tag", required=True, choices=("judge", "evaluator"))
    ap.add_argument("--top-k", type=int, default=TOP_K, dest="top_k")
    args = ap.parse_args()

    items = {it.id: it for it in load_items()}
    ex = Extractor(args.endpoint)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"extract[{args.tag}]: model={ex.model}")

    for pool in POOLS:
        src = REPO / "results" / pool / "transcripts.jsonl"
        dest = OUT / f"{pool}-{args.tag}.jsonl"
        recs = [json.loads(x) for x in src.open() if x.strip()]
        done = set()
        if dest.exists():
            done = {json.loads(x)["id"] for x in dest.open() if x.strip()}
        todo = [r for r in recs if r["id"] not in done]
        print(f"  {pool}: {len(todo)} of {len(recs)} to go")
        with dest.open("a") as sink:
            for i, r in enumerate(todo, 1):
                po = ex.canonical_measured(items[r["item"]], r["transcript"],
                                           options=OPTIONS, top_k=args.top_k)
                sink.write(json.dumps({
                    "id": r["id"], "item": r["item"], "gold": r["gold"],
                    "kind": r.get("kind"), "choice": po.choice,
                    # FULL precision on every field. Rounding here is the defect
                    # this corpus exists to measure; five decimals would silently
                    # re-manufacture it at write time (wave 5, and §3's own
                    # six-decimal incident).
                    "dist": dict(po.dist),
                    "raw_logprob": dict(po.raw_logprob),
                    "mass_unnormalized": po.mass_unnormalized,
                    # the whole point of this pass: rank becomes recoverable
                    "window": [list(t) for t in po.window],
                }, ensure_ascii=False) + "\n")
                if i % 100 == 0:
                    sink.flush()
                    print(f"    {i}/{len(todo)}")
        print(f"  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
