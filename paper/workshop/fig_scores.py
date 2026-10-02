"""Figure 1 — the artifact and the measurement, on the SAME records.

Left: the committed v1 readout on the s3 bank records (the 90.6% exactly-zero
spike, E-116). Right: the corrected readout on the IDENTICAL 660 records
(results/s3m3_measured, re-extracted with the same models and prompt — only the
readout changed). Like-for-like by construction, and enforced below: the two
panels must be built from exactly the same record ids or the figure refuses.

Same records, same n, same units => shared y-axis. The earlier cross-corpus
version needed per-panel scaling and a caption apology; both die here.

Reads only committed artifacts. Vector PDF, greyscale-safe, ~8pt at final size.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42})


def bank_records() -> dict[str, str]:
    """s3 bank record id -> gold option letter."""
    return {r["id"]: r["gold"]
            for r in (json.loads(x) for x in (REPO / "results/s3/transcripts.jsonl").open()
                      if x.strip())
            if r.get("kind") == "bank"}


def v1_scores(bank: dict[str, str]) -> dict[str, float]:
    ex = {}
    for tag in ("judge", "evaluator"):
        ex[tag] = {json.loads(x)["id"]: json.loads(x)["dist"]
                   for x in (REPO / f"results/s3/extract-{tag}.jsonl").open() if x.strip()}
    # NOT .get(gold, 0.0): defaulting a missing option to probability zero is
    # precisely the equation this paper is about. If the key is absent the cache
    # is wrong and the figure must not be drawn.
    return {rid: min(1.0 - ex[t][rid][gold] for t in ("judge", "evaluator"))
            for rid, gold in bank.items()}


def corrected_scores(bank: dict[str, str]) -> dict[str, float]:
    ex = {}
    for tag in ("judge", "evaluator"):
        ex[tag] = {json.loads(x)["id"]: json.loads(x)["dist"]
                   for x in (REPO / f"results/s3m3_measured/s3-{tag}.jsonl").open()
                   if x.strip()}
    return {rid: min(1.0 - ex[t][rid][gold] for t in ("judge", "evaluator"))
            for rid, gold in bank.items()}


def main() -> int:
    bank = bank_records()
    v1, cor = v1_scores(bank), corrected_scores(bank)
    # the like-for-like guarantee, enforced rather than asserted in a caption
    assert set(v1) == set(cor) == set(bank), "panels are not the same records"

    xs1, xs2 = list(v1.values()), list(cor.values())
    z1 = sum(1 for x in xs1 if x == 0.0) / len(xs1)
    z2 = sum(1 for x in xs2 if x == 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), sharey=True)
    floor = 1e-10
    bins = [b for b in (floor * 10 ** (i / 3) for i in range(0, 32)) if b <= 1.0]
    bins.append(1.0)
    for ax, xs, name, ztxt in (
            (axes[0], xs1, "original readout", f"exactly 0.0: {z1:.1%}"),
            (axes[1], xs2, "corrected readout (this work)",
             f"exactly 0.0: {z2} of {len(xs2)}")):
        w = [1.0 / len(xs)] * len(xs)
        ax.hist([max(x, floor) for x in xs], bins=bins, weights=w,
                color="0.35", edgecolor="white", linewidth=.3)
        ax.set_xscale("log")
        ax.set_title(f"{name}\n{ztxt}  (n={len(xs)}, same records)", fontsize=8)
        ax.set_xlabel("true-option nonconformity $1-p(\\mathrm{true\\ option})$")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylim(0, 1.0)          # shared, honest: same records, same units
    axes[0].set_ylabel("share of records")
    axes[0].annotate("asserted,\nnot measured", xy=(floor, z1),
                     xytext=(1e-7, 0.62 * z1), fontsize=6.5,
                     arrowprops=dict(arrowstyle="->", lw=.6, color="0.2"))
    fig.tight_layout()
    fig.savefig(OUT / "fig_scores.pdf")
    print(f"fig1: v1 zero-share {z1:.4f} (C3) | corrected zeros {z2}/{len(xs2)} (C77)")
    assert abs(z1 - 0.9061) < 5e-4, "C3 violated -- do not ship this figure"
    assert z2 == 0, "C77 violated -- do not ship this figure"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
