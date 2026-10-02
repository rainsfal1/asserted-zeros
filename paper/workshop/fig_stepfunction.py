"""Fig 3 (new) -- the step function: what truncation does to issued certificates.

Reads per_item_honest.json. Ten single-item decisions at an honest budget
(Sigma-alpha = alpha_i = 0.10, clean channel), ordered by the item's calibration
error rate. Greyscale-safe: marker and line style carry the series, not colour.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "per_item_honest.json").read_text())
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42})


def main() -> int:
    e = [r["cal_error"] for r in D["items"]]
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.1))
    for k, (key, lab, mk, ls) in enumerate((("measured", "measured readout", "o", "-"),
                                            ("truncated", "with floor ($\\tau=10^{-2}$)", "s", "--"))):
        ax[0].plot(e, [r[key]["release_rate"] for r in D["items"]], ls=ls, marker=mk,
                   ms=4, lw=1, color="0.15", label=lab)
        ax[1].plot(e, [r[key]["wrong"] for r in D["items"]], ls=ls, marker=mk,
                   ms=4, lw=1, color="0.15")
    # The step is where the order statistic allows at most 10 wrong of n_cal --
    # NOT at alpha_i. An earlier version drew alpha_i here, which put the marker
    # in the wrong place and invited the reader to attribute the step to the
    # budget rather than to the quantile index (audit wave 7).
    step = 10 / D["n_cal"]
    for a in ax:
        a.axvline(step, color="0.55", lw=.8)
        a.set_xlabel("item's calibration error rate")
        a.grid(alpha=.25)
    # rotated along the line: the horizontal version collided with the legend
    ax[0].text(step * 1.03, .28, r"$10/n_{\rm cal}$", fontsize=6.5, color="0.35",
               rotation=90, va="bottom")
    ax[0].set_ylabel("release rate")
    ax[1].set_ylabel("wrong verdicts issued")
    # a count axis takes integer ticks: 12.5 wrong verdicts is not a thing,
    # and fractional ticks also trip the claims gate as untraceable numbers
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    # The caption names "the two above the step" and "the four below it" but the
    # figure carried no item numbers, so neither claim could be located in it.
    # Ties share an x (items 1,2 at 0.0), so label grouped by position -- an
    # annotation per marker would overlap: items 1 and 2 sit 0.025 apart in y.
    groups: dict[float, list[str]] = {}
    for r in D["items"]:
        groups.setdefault(round(r["cal_error"], 6), []).append(str(r["item"]))
    # Groups 1,2 and 7,8 sit 0.009 apart on an axis spanning 0.17, so a single
    # row of labels overprints them (as does 13 against 4,5,6). Stagger into two
    # rows: adjacent groups never share a row, so nothing can collide.
    # Two rows keep the labels from overprinting, but staggering alone loses the
    # association: an external reader twice read "4,5,6" (error 0.069) as sitting
    # at item 20's 0.155. Draw a leader from each label down to its own x.
    for a in ax:
        tr = a.get_xaxis_transform()
        for j, (x, ids) in enumerate(groups.items()):
            top = 1.03 + 0.10 * (j % 2)
            a.plot([x, x], [1.0, top - 0.01], transform=tr, lw=.5,
                   color="0.55", clip_on=False, zorder=1)
            a.text(x, top, ",".join(ids), transform=tr,
                   ha="center", va="bottom", fontsize=6, color="0.25")
        a.tick_params(top=False)
    ax[0].set_title("item", fontsize=7, pad=19)
    ax[1].set_title("item", fontsize=7, pad=19)
    # Readers could not locate "the two above the step release nothing" in the
    # plot -- the caption was carrying it alone. Mark it on the mark itself.
    # The panel is crowded: every interior placement collided with the legend,
    # the step label or the descending dashed line. Put it in the margin
    # directly beneath the zero-release markers, where nothing else can go.
    zr = [r["cal_error"] for r in D["items"] if r["truncated"]["release_rate"] == 0]
    if zr:
        ax[0].set_ylim(-0.16, 1.05)
        ax[0].plot(zr, [0] * len(zr), marker="s", ms=7, mfc="none",
                   mec="0.25", lw=0, zorder=5)
        ax[0].text(max(zr) + 0.007, -0.145, "floor releases nothing",
                   fontsize=6.5, ha="right", va="bottom", color="0.2")
    ax[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "figs" / "fig_stepfunction.pdf", bbox_inches="tight")
    print("wrote fig_stepfunction.pdf")
    # The figure may not disagree with the artifact it is drawn from. fig_scores.py
    # has guarded its two headline numbers since it was written; this one did not,
    # and a cold reviewer caught the supplement's README claiming that both did.
    n = len(D["items"])
    wm = sum(r["measured"]["wrong"] for r in D["items"])
    wt = sum(r["truncated"]["wrong"] for r in D["items"])
    zr = sum(1 for r in D["items"] if r["truncated"]["released"] == 0)
    print(f"fig2: {n} items | wrong measured {wm} -> truncated {wt} | zero-release {zr}/{n}")
    assert n == 10, f"figure 2 is a ten-item panel, got {n}"
    assert (wm, wt) == (57, 44), f"C54 violated: wrong {wm}->{wt}, expected 57->44"
    assert zr == 2, f"C54 violated: zero-release {zr}/{n}, expected 2"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
