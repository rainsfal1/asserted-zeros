"""
Calibrating the uncertainty sets — how extractor distributions become S_i  (M2).

``certificate.py`` owns the *decision*: given the sets S_i, which verdicts are
reachable and may one be released. This module owns how those sets are *built*,
which is the object the §2.3 guarantee actually attaches to. M1 built them with a
provisional rule — the legacy builder per family at a hardcoded p ≥ 0.2, then union —
and both M1's report and the build guide say M2 must *replace* that with a single
calibrated object rather than inherit it. That is what a :class:`SetRule` is.

Three things here are not in the textbook split-conformal recipe, and each exists
because the data or an experiment forced it:

**1. Grouping.** The S3 pool's 770 records are 77 unique utterances — 66 bank
(two wordings for each of 11 items × 3 options) plus 11 probes — re-drawn
across 5 channel levels × 2 seeds. Ten rows sharing an utterance are not ten
independent draws, so a record-level calibration/evaluation split leaks the same
utterance into both halves and reports a coverage that will not hold on a new
speaker. :func:`grouped_split` splits on the utterance, and the harness reports
the naive-vs-grouped gap rather than hiding it.

**2. Multiplicity over re-asks.** M1.5 found that re-asking the *same* wording and
stopping at the first singleton released wrong verdicts in 10.9% of session-runs
where M1 released none: repeated elicitation of one item is a selection procedure
over the extractors' noise, and §2.3's union bound Σ_i α_i covers one elicitation
per item, not K of them. A loop that may
re-elicit therefore opts into **geometric α-spending** (``spend=True``): the k-th
elicitation of an item runs at α/2^k, so the total spent over ANY number of
re-asks and ANY stopping rule is Σ_k α/2^k < α, and **re-asking widens the set
instead of narrowing it by luck**. Two deliberate consequences: harmonic spending
α/k is NOT used — it looks like Bonferroni but sums to α·H_K ≈ 1.83α at K=3, so
it merely shrinks the exploit rather than bounding it; and under spending even
the first elicitation runs at α/2, one notch wider than plain split conformal,
because the budget must be reserved up front for re-asks that may never happen.
``spend=False`` is the plain object, valid only when an item is elicited exactly
once — and since F-4's fix it must be passed EXPLICITLY; there is no default.

**3. Empty means ignorant, not confident.** The legacy builder (quarantined in
``experiments/_legacy.py`` since F-3's fix) guarantees a non-empty set by falling
back to the argmin option. At M1's threshold of 0.8 that fallback can never fire
(three options summing to 1 always leave one above 0.2), but a *calibrated*
threshold is far smaller, and there the fallback would convert "no option is
plausible" into a confident singleton — the exact shape that certifies a wrong
verdict. :func:`conformal_set` maps the empty set to the FULL set {a,b,c}:
maximum ignorance, forcing a re-ask; the quarantine keeps M1/M1.5 reproducible.

No external dependencies; like ``certificate.py`` this file is meant to be read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from .certificate import calibrate_threshold
from .scorer import OPTIONS

#: Renormalized mass over the options in the decision token's top-k, NOT the
#: model's distribution over {a,b,c} — see `extraction.recover_mass` and F-11.
Dist = Mapping[str, float]
#: All seated extractors' distributions for a single elicitation, keyed by tag.
Dists = Mapping[str, Dist]
#: A nonconformity score per option (lower = more plausible).
Scores = dict[str, float]

POOLED = "pooled"
MIN = "min"
SPLIT = "split"


# ---------------------------------------------------------------------------
# Nonconformity and the combination rules
# ---------------------------------------------------------------------------


def nonconformity(dist: Dist, option: str) -> float:
    """The conformal score for one option: probability mass NOT on it."""
    return 1.0 - dist.get(option, 0.0)


def combine_pooled(dists: Dists, tags: Sequence[str]) -> Scores:
    """Average the extractors' probabilities, then score once.

    A single fused opinion. Note this is a fixed arithmetic rule inside the
    validator over two *already-committed, blind* extractions — it is not
    reconciliation and does not tune either extractor toward the other (T2).
    """
    n = len(tags)
    return {o: 1.0 - sum(dists[t].get(o, 0.0) for t in tags) / n for o in OPTIONS}


def combine_min(dists: Dists, tags: Sequence[str]) -> Scores:
    """Keep an option if ANY extractor finds it plausible: s(o) = min_t s_t(o).

    Thresholding this at t is exactly the union of the per-extractor sets at t —
    i.e. M1's provisional union rule, except that the single threshold is now
    calibrated rather than hardcoded. It is the conservative reading of
    disagreement: one extractor vouching is enough to keep an option in play.
    """
    return {o: min(nonconformity(dists[t], o) for t in tags) for o in OPTIONS}


COMBINERS: dict[str, Callable[[Dists, Sequence[str]], Scores]] = {
    POOLED: combine_pooled,
    MIN: combine_min,
}


def conformal_set(scores: Scores, threshold: float) -> frozenset[str]:
    """Options whose score clears the threshold; EMPTY becomes the FULL set.

    See the module docstring: an empty conformal set is a statement of ignorance
    ("no option is plausible"), and the safe encoding of ignorance in this system
    is {a,b,c}, which straddles every boundary and forces recovery. Collapsing it
    to the single best-scoring option instead would manufacture confidence
    exactly where the extractors had none.
    """
    keep = frozenset(o for o, s in scores.items() if s <= threshold)
    return keep if keep else frozenset(OPTIONS)


# ---------------------------------------------------------------------------
# Grouped splitting — exchangeability is per utterance, not per record
# ---------------------------------------------------------------------------


def grouped_split(
    records: Sequence[Mapping],
    *,
    frac: float = 0.5,
    seed: int = 0,
    group_key: str = "group",
) -> tuple[list[Mapping], list[Mapping]]:
    """Split into (calibration, evaluation) so no group spans both halves.

    ``frac`` is the share of GROUPS assigned to calibration. Deterministic given
    ``seed``. The whole point is that the evaluation half contains utterances the
    threshold has never seen.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    groups = sorted({r[group_key] for r in records})
    rng = _Rng(seed)
    rng.shuffle(groups)
    n_cal = max(1, min(len(groups) - 1, round(len(groups) * frac)))
    cal_groups = set(groups[:n_cal])
    cal = [r for r in records if r[group_key] in cal_groups]
    ev = [r for r in records if r[group_key] not in cal_groups]
    return cal, ev


class _Rng:
    """Tiny deterministic shuffler — avoids depending on ``random``'s global state
    and keeps splits stable across Python versions."""

    def __init__(self, seed: int) -> None:
        self.state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def _next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.state >> 33

    def shuffle(self, seq: list) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self._next() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]


# ---------------------------------------------------------------------------
# The calibrated set rule
# ---------------------------------------------------------------------------


def alpha_for_elicitations(alpha: float, k: int) -> float:
    """Geometric α-spending over repeated elicitations of one item (M1.5).

    The k-th elicitation runs at α/2^k. An item elicited any number of times,
    under any stopping rule, has total miscoverage bounded by Σ_k α/2^k < α —
    the union bound over which elicitation ends up being the final one. This is
    what makes re-asking safe: each fresh elicitation REPLACES the set, so every
    one of them is a fresh chance for the final set to miss, and the levels must
    sum to the budget.

    Deliberately NOT α/k: that harmonic schedule sums to α·H_K (≈1.83α at K=3),
    so it shrinks the M1.5 exploit without actually bounding it. The cost of the
    valid schedule is paid up front — k=1 runs at α/2, one notch wider than
    plain split conformal, because budget is reserved for re-asks that may never
    happen.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    return alpha / (2.0 ** k)


@dataclass(frozen=True)
class SetRule:
    """A rule fitted on calibration data that maps extractor distributions to S_i.

    Holds the calibration *scores* rather than a single fitted number so that the
    threshold can be re-derived at any elicitation count k without refitting.
    """

    mode: str
    tags: tuple[str, ...]
    alpha: float
    #: true-option nonconformity scores per threshold stream. Single-threshold
    #: modes use the key ``"*"``; ``split`` keeps one stream per extractor.
    cal_scores: dict[str, list[float]] = field(default_factory=dict)

    # -- thresholds ---------------------------------------------------------

    def thresholds(self, *, k: int = 1, spend: bool) -> dict[str, float]:
        """Threshold(s) for the k-th elicitation. Split mode divides α across tags.

        ``spend`` selects the multiplicity regime, and it belongs to the RE-ASK
        POLICY, not the rule — it is REQUIRED (F-4): ``spend=False`` runs every
        elicitation at the plain split-conformal level α, valid if and only if
        the item is elicited once. A loop that may re-elicit must pass
        ``spend=True`` from the FIRST elicitation, pricing elicitation k at
        α/2^k so the total stays under α however the loop stops. spend=False
        plus re-asking is exactly the M1.5 exploit; B1 priced it at 20×.
        """
        base = self.alpha / len(self.tags) if self.mode == SPLIT else self.alpha
        a = alpha_for_elicitations(base, k) if spend else base
        if self.mode == SPLIT:
            return {t: calibrate_threshold(self.cal_scores[t], a) for t in self.tags}
        return {"*": calibrate_threshold(self.cal_scores["*"], a)}

    # -- application --------------------------------------------------------

    def scores(self, dists: Dists) -> Scores:
        """The combined per-option nonconformity (single-threshold modes only)."""
        if self.mode == SPLIT:
            raise ValueError("split mode has one score stream per extractor; use build()")
        return COMBINERS[self.mode](dists, self.tags)

    def build(self, dists: Dists, *, k: int = 1, spend: bool) -> frozenset[str]:
        """The calibrated uncertainty set for one item's k-th elicitation."""
        thr = self.thresholds(k=k, spend=spend)
        if self.mode == SPLIT:
            keep: set[str] = set()
            for t in self.tags:
                keep |= {o for o in OPTIONS if nonconformity(dists[t], o) <= thr[t]}
            return frozenset(keep) if keep else frozenset(OPTIONS)
        return conformal_set(self.scores(dists), thr["*"])

    def covers(self, dists: Dists, gold: str, *, k: int = 1, spend: bool) -> bool:
        return gold in self.build(dists, k=k, spend=spend)


def fit(
    records: Iterable[Mapping],
    *,
    mode: str,
    tags: Sequence[str],
    alpha: float,
) -> SetRule:
    """Fit a :class:`SetRule` on labelled calibration records.

    Each record needs ``gold`` (the true option) and ``dists`` (tag -> distribution).
    Fitting only collects the true-option scores; the quantile is taken later, so
    one fitted rule serves every elicitation count.
    """
    if mode not in (POOLED, MIN, SPLIT):
        raise ValueError(f"unknown mode {mode!r}")
    tags = tuple(tags)
    streams: dict[str, list[float]] = {t: [] for t in tags} if mode == SPLIT else {"*": []}
    for r in records:
        gold, dists = r["gold"], r["dists"]
        if mode == SPLIT:
            for t in tags:
                streams[t].append(nonconformity(dists[t], gold))
        else:
            streams["*"].append(COMBINERS[mode](dists, tags)[gold])
    if not any(streams.values()):
        raise ValueError("empty calibration set")
    return SetRule(mode=mode, tags=tags, alpha=alpha, cal_scores=streams)


# ---------------------------------------------------------------------------
# Loop 2 — the certificate learns from resolved outcomes
# ---------------------------------------------------------------------------


def decaying_step(step0: float, t: int) -> float:
    """Step size for online conformal at time t (1-indexed): step0 / sqrt(t).

    Decaying rather than constant (Angelopoulos, Barber & Bates 2024): a constant
    step chases noise forever and leaves the released/withheld boundary jittering
    between reruns, while a decaying step converges to the population quantile in
    the stationary regime and still adapts to shift early on.
    """
    if t < 1:
        raise ValueError("t must be >= 1")
    return step0 / math.sqrt(t)


def online_update(threshold: float, *, missed: bool, alpha: float, step: float) -> float:
    """One online-conformal update from a single RESOLVED outcome.

    ``missed`` is the ground-truth signal a resolved session yields: did the true
    answer fall outside the uncertainty set? Miss → widen; hit → tighten slightly.
    The stationary point is the (1-α) quantile of the score distribution, so the
    threshold tracks drift without anyone retraining a model.

    This is the whole of the feedback loop at the certificate layer, and it is
    deliberately a fixed, auditable formula over outcomes — the validator stays
    deterministic, and the signal is ground truth, never judge/evaluator
    agreement.
    """
    thr = threshold + step * ((1.0 if missed else 0.0) - alpha)
    return min(1.0, max(0.0, thr))


# ---------------------------------------------------------------------------
# Selection-corrected calibration
# ---------------------------------------------------------------------------


def weighted_quantile(scores: Sequence[float], weights: Sequence[float], level: float) -> float:
    """Weighted empirical quantile with the split-conformal (+1) correction.

    Needed because the sessions that get RESOLVED are chosen by our own withhold
    rule, so the resolved pool is not exchangeable with deployment (feedback
    covariate shift). Re-weighting each calibration point by the inverse of its
    probability of having been resolved restores the target distribution.
    """
    if len(scores) != len(weights):
        raise ValueError("scores and weights differ in length")
    if not scores:
        raise ValueError("empty calibration set")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    total = sum(weights) + max(weights)  # the (n+1)-style correction, weighted
    acc = 0.0
    for i in order:
        acc += weights[i]
        if acc / total >= level:
            return scores[i]
    return math.inf  # not enough weight to guarantee the level — include everything
