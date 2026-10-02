"""V3 figures 1-3 (prereg §6) and the two re-analyses §9 records.

Q4 and Q5 were FALSIFIED as registered, and in both cases the registered test was
the wrong instrument: Q4's slack counted test-set noise but not calibration-draw
variability, and Q5's ±20% RELATIVE band is meaningless when both rates sit at or
below the 1/116 resolution limit. The registered verdicts stand; these re-analyses
are reported beside them, never instead of them.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import beta as Beta  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "v3_frontier"
STRATA = ("tel", "snr0", "snr_m5", "snr_m10", "mixed")
RES = json.loads((OUT / "frontier.json").read_text())
K = RES["meta"]["k_values"]
ALPHAS = RES["meta"]["alphas"]
N = 116
RESOLUTION = 1.0 / N          # one session — the finest difference measurable


def cells(s, k=None, a=None):
    out = RES["strata"][s]["cells"]
    if k is not None:
        out = [c for c in out if c["k"] == k]
    if a is not None:
        out = [c for c in out if abs(c["alpha_i"] - a) < 1e-12]
    return out


def mean_release(s, k, a):
    return statistics.mean(c["release_rate"] for c in cells(s, k, a))


def required_sigma(s, k, target=0.5):
    for a in sorted(ALPHAS):
        if mean_release(s, k, a) >= target:
            return k * a
    return None


def tc_upper(alpha, n=N, q=0.95):
    """Vovk (2012) training-conditional miscoverage: Beta(n+1-l, l), l = ceil((n+1)(1-α))."""
    lo = min(max(math.ceil((n + 1) * (1 - alpha)), 1), n)
    return float(Beta.ppf(q, n + 1 - lo, lo))


# ---------------------------------------------------------------- re-analyses
def q4_reanalysis():
    marginal, conditional, surviving = 0, 0, []
    for s in STRATA:
        for c in cells(s):
            sig, n = c["sigma_alpha"], c["n"]
            if sig > 1.0:
                continue
            if c["wrong_release_rate"] > sig + 1.96 * math.sqrt(sig * (1 - sig) / n):
                marginal += 1
                bound = min(1.0, c["k"] * tc_upper(c["alpha_i"]))
                if c["wrong_release_rate"] > bound + 1.96 * math.sqrt(
                        max(bound * (1 - bound), 0) / n):
                    conditional += 1
                    surviving.append({"stratum": s, "k": c["k"],
                                      "alpha_i": c["alpha_i"],
                                      "realized": c["wrong_release_rate"],
                                      "tc_bound": round(bound, 4)})
    return {"breaches_marginal": marginal, "breaches_training_conditional": conditional,
            "n_cells_checked": sum(1 for s in STRATA for c in cells(s)
                                   if c["sigma_alpha"] <= 1.0),
            "surviving": surviving,
            "inflation": {str(a): round(tc_upper(a) / a, 2) for a in (0.01, 0.02, 0.05, 0.10)}}


def q5_reanalysis():
    """Absolute-difference reading, with the 1/116 resolution limit stated."""
    diffs, beyond = [], 0
    for s in STRATA:
        for k in K:
            cs = cells(s, k)
            conf = sorted({(b["release_rate"], b["wrong_release_rate"])
                           for c in cs for b in c["baseline_confidence"]})
            for i in range(1, 10):
                t = 0.1 * i
                cf = [c["wrong_release_rate"] for c in cs if t <= c["release_rate"] <= t + 0.05]
                cb = [w for r, w in conf if t <= r <= t + 0.05]
                if not cf or not cb:
                    continue
                d = min(cf) - min(cb)
                diffs.append(d)
                beyond += abs(d) > RESOLUTION
    return {"n_matched": len(diffs), "resolution": round(RESOLUTION, 4),
            "n_beyond_one_session": beyond,
            "median_abs_diff": round(statistics.median(abs(d) for d in diffs), 4),
            "conformal_lower": sum(1 for d in diffs if d < -1e-12),
            "confidence_lower": sum(1 for d in diffs if d > 1e-12),
            "tied": sum(1 for d in diffs if abs(d) <= 1e-12)}


# ------------------------------------------------------------------- figures
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, a in zip(axes, (0.05, 0.10)):
        for k in K:
            xs = [RES["strata"][s]["epsilon_mean"] for s in STRATA]
            ys = [mean_release(s, k, a) for s in STRATA]
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order],
                    marker="o", label=f"k={k}")
        ax.set_xlabel("extraction error rate  ε̄")
        ax.set_title(f"per-item α = {a}   (Σα = {a}·k)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("certified release rate")
    axes[0].legend(title="conjunction size", fontsize=8)
    fig.suptitle("Figure 1 — release collapses with extraction error, and earlier for larger k")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_release_vs_epsilon.png", dpi=160)


def fig2():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for s in STRATA:
        ys = [required_sigma(s, k) for k in K]
        xs = [k for k, y in zip(K, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, marker="o",
                label=f"{s} (ε̄={RES['strata'][s]['epsilon_mean']:.2f})")
    ax.axhline(1.0, ls="--", c="k", lw=1)
    ax.text(1.05, 1.06, "Σα > 1 is not a guarantee", fontsize=8)
    ax.axhline(0.10, ls=":", c="gray", lw=1)
    ax.text(1.05, 0.075, "Σα = 0.10", fontsize=8, color="gray")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("conjunction size k")
    ax.set_ylabel("required Σα for release ≥ 50%")
    ax.set_title("Figure 2 — the price of conjunction: required Σα vs k")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_sigma_vs_k.png", dpi=160)


def fig3():
    show_k = (1, 3, 10)
    show_s = ("tel", "mixed", "snr_m10")
    fig, axes = plt.subplots(len(show_k), len(show_s), figsize=(12, 9),
                             sharex=True, sharey=True)
    for r, k in enumerate(show_k):
        for c, s in enumerate(show_s):
            ax = axes[r][c]
            cs = cells(s, k)
            pts = sorted({(round(x["release_rate"], 4), x["wrong_release_rate"])
                          for x in cs})
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    label="conformal gate", ms=4)
            conf = sorted({(b["release_rate"], b["wrong_release_rate"])
                           for x in cs for b in x["baseline_confidence"]})
            ax.plot([p[0] for p in conf], [p[1] for p in conf], "s--",
                    label="confidence threshold", ms=4, alpha=.8)
            al = cs[0]["baseline_always"]
            ax.plot([al["release_rate"]], [al["wrong_release_rate"]], "r*",
                    ms=13, label="always answer")
            ax.set_title(f"k={k}, {s} (ε̄={RES['strata'][s]['epsilon_mean']:.2f})",
                         fontsize=9)
            ax.grid(alpha=.3)
            if r == len(show_k) - 1:
                ax.set_xlabel("release rate (coverage)")
            if c == 0:
                ax.set_ylabel("P(release ∧ wrong)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Figure 3 — risk–coverage: the gate buys a guarantee, not a better curve")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_risk_coverage.png", dpi=160)


def main() -> int:
    re4, re5 = q4_reanalysis(), q5_reanalysis()
    (OUT / "reanalysis.json").write_text(json.dumps({"Q4": re4, "Q5": re5}, indent=1))
    print("Q4 re-analysis:", json.dumps(re4, indent=1)[:400])
    print("Q5 re-analysis:", json.dumps(re5, indent=1))
    fig1()
    fig2()
    fig3()
    print("wrote fig1/fig2/fig3 +", OUT / "reanalysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
