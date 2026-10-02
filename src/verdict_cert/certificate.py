"""
The verdict-referenced fidelity certificate  (proposal §2 — the M0 artifact).

Given a per-item *uncertainty set* S_i ⊆ {a, b, c} for each of the 11 items —
the answers the respondent plausibly gave, as judged from the judge/evaluator
extractions — decide whether the placement verdict is *invariant* across every
plausible profile. If it is, the verdict is licensed: issue the certificate. If
not, the verdict is not yet earned and the pivotal items must be re-asked.

Because the scorer (``scorer.py``) is deterministic, additive, and tiny, this is
EXACT, not a bound. We compute the full set of reachable verdicts

    V(S) = { placement_decision(x) : x ∈ S_1 × … × S_11 }

via a subset-sum dynamic program over per-item point contributions, with the p10
integrity gate handled explicitly, and issue iff |V(S)| == 1. The DP is checked
against brute-force enumeration of the whole product set in the test suite.

Guarantee (proposal §2.3). If split-conformal calibration makes each set cover
the true answer with probability ≥ 1 − α_i, then a released verdict equals the
true-answer verdict with probability ≥ 1 − Σ_i α_i (union bound over the items).

No external dependencies — this file is meant to be read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .scorer import ITEMS, OPTIONS, SCORE_MAP, overall_band

# An uncertainty profile: every item -> a non-empty subset of {a, b, c}.
# A fully unextracted item is the FULL set {a, b, c} (never a default answer),
# which is why the "unextracted defaults to the max-scoring option" bug cannot
# recur under the certificate: the full set straddles the boundaries and forces
# a re-ask instead of silently scoring.
UncertaintySets = Mapping[str, frozenset[str]]


def full_sets() -> dict[str, frozenset[str]]:
    """Maximum ignorance: every item unresolved."""
    return {it: frozenset(OPTIONS) for it in ITEMS}


def singleton_sets(answers: Mapping[str, str]) -> dict[str, frozenset[str]]:
    """A fully resolved profile as singleton sets (always certifiable)."""
    return {it: frozenset({answers[it]}) for it in ITEMS}


def _validate(sets: UncertaintySets) -> None:
    for it in ITEMS:
        s = sets.get(it)
        if not s:
            raise ValueError(f"item {it}: uncertainty set is empty; use {{a,b,c}} for unknown")
        if not set(s) <= set(OPTIONS):
            raise ValueError(f"item {it}: {set(s)} not a subset of {OPTIONS}")


PAYMENT_RISKS = ("low", "high")
#: Maximum ignorance about the exogenous payment-risk input. This is the
#: DEFAULT since F-6's fix (2026-08-16): an unknown payment risk straddles
#: overseas_ready / needs_support on high-band profiles and forces a withhold —
#: the same ignorance semantics as the empty→FULL set rule, applied to the
#: exogenous layer. The deployed "missing → favorable branch" shape (the exact
#: upstream bug the certificate exists to kill) is no longer expressible by
#: forgetting an argument; v1 harnesses pass payment_risk="low" explicitly.
PAYMENT_RISK_UNKNOWN: frozenset[str] = frozenset(PAYMENT_RISKS)


def _risk_values(payment_risk: str | frozenset[str]) -> tuple[str, ...]:
    vals = (payment_risk,) if isinstance(payment_risk, str) else tuple(sorted(payment_risk))
    if not vals or any(v not in PAYMENT_RISKS for v in vals):
        raise ValueError(f"payment_risk {payment_risk!r} not within {PAYMENT_RISKS}")
    return vals


def reachable_verdicts(
    sets: UncertaintySets,
    *,
    payment_risk: str | frozenset[str] = PAYMENT_RISK_UNKNOWN,
    integrity_flag: bool = False,
) -> set[str]:
    """
    EXACT set of verdicts reachable over the product S_1 × … × S_11.

    p10 is special: it both contributes to the total AND gates the integrity
    block (option 'c' -> 'low' -> blocked). We split on it, then subset-sum the
    other ten items. ``payment_risk`` may be a SET of plausible values; the
    reachable set unions over them (F-6 fix — unknown risk must widen, never
    silently pick the favorable branch).
    """
    _validate(sets)
    risks = _risk_values(payment_risk)

    # A raised voice-agent integrity flag blocks regardless of the answers.
    if integrity_flag:
        return {"blocked"}

    verdicts: set[str] = set()

    s10 = sets["p10"]
    if "c" in s10:  # some plausible profile picks p10='c' -> integrity 'low' -> blocked
        verdicts.add("blocked")

    non_c_p10 = [o for o in s10 if o != "c"]
    if non_c_p10:
        # Reachable partial sums from the other ten items (subset-sum over point sets).
        sums: set[int] = {0}
        for it in ITEMS:
            if it == "p10":
                continue
            pts = {SCORE_MAP[it][o] for o in sets[it]}
            sums = {s + p for s in sums for p in pts}

        for o in non_c_p10:
            p10_pts = SCORE_MAP["p10"][o]  # a->2, b->1
            for s in sums:
                overall = overall_band(s + p10_pts)
                if overall == "high":
                    for r in risks:
                        verdicts.add("needs_support" if r == "high" else "overseas_ready")
                elif overall == "medium":
                    verdicts.add("needs_support")
                else:
                    verdicts.add("high_risk")

    return verdicts


@dataclass(frozen=True)
class Certificate:
    """Result of a certification attempt for one session."""

    certified: bool
    verdict: str | None          # the licensed verdict, iff certified
    reachable: frozenset[str]    # the full reachable set (|.|==1 iff certified)
    pivotal_items: tuple[str, ...]  # items still carrying verdict-relevant uncertainty

    def __bool__(self) -> bool:  # `if certificate(...):`
        return self.certified


def certify(
    sets: UncertaintySets,
    *,
    payment_risk: str | frozenset[str] = PAYMENT_RISK_UNKNOWN,
    integrity_flag: bool = False,
) -> Certificate:
    """Issue the certificate iff the verdict is invariant over all plausible answers."""
    reachable = reachable_verdicts(sets, payment_risk=payment_risk, integrity_flag=integrity_flag)
    certified = len(reachable) == 1
    pivotal = () if certified else tuple(
        it for it, _gain in reask_ranking(
            sets, payment_risk=payment_risk, integrity_flag=integrity_flag
        )
    )
    return Certificate(
        certified=certified,
        verdict=next(iter(reachable)) if certified else None,
        reachable=frozenset(reachable),
        pivotal_items=pivotal,
    )


def reask_ranking(
    sets: UncertaintySets,
    *,
    payment_risk: str | frozenset[str] = PAYMENT_RISK_UNKNOWN,
    integrity_flag: bool = False,
) -> list[tuple[str, int]]:
    """
    Rank unresolved items by verdict information gain: for each item with |S_i|>1,
    the best-case reduction in |V(S)| if that item were resolved to a singleton.

    Best-case (the most favorable option) is the optimistic estimate — the item
    that *can* collapse the reachable set the most is the most worth re-asking.
    Returned high-gain first; ties broken by item order.
    """
    base = len(reachable_verdicts(sets, payment_risk=payment_risk, integrity_flag=integrity_flag))
    ranked: list[tuple[str, int]] = []
    for it in ITEMS:
        if len(sets[it]) <= 1:
            continue
        best_after = min(
            len(reachable_verdicts(
                {**sets, it: frozenset({o})},
                payment_risk=payment_risk,
                integrity_flag=integrity_flag,
            ))
            for o in sets[it]
        )
        ranked.append((it, base - best_after))
    ranked.sort(key=lambda kv: (-kv[1], ITEMS.index(kv[0])))
    return ranked


# ---------------------------------------------------------------------------
# Split-conformal calibration of the per-item uncertainty sets
# ---------------------------------------------------------------------------


def calibrate_threshold(nonconformity: list[float], alpha: float) -> float:
    """
    Split-conformal threshold at miscoverage α.

    ``nonconformity`` is the list of true-label nonconformity scores on a held-out
    calibration set (e.g. 1 − p_true, the extractor's probability mass NOT on the
    answer the respondent actually gave). Returns q̂ such that including every
    option with score ≤ q̂ yields marginal coverage ≥ 1 − α, distribution-free,
    with the standard finite-sample (n+1) correction.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(nonconformity)
    if n == 0:
        raise ValueError("empty calibration set")
    # rank of the (1-alpha) quantile with finite-sample correction; clamp to n
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf  # not enough calibration data to guarantee 1-alpha -> include all
    return sorted(nonconformity)[k - 1]


# ---------------------------------------------------------------------------
# The probability-weighted counterpart — for the ACTING baselines only
# ---------------------------------------------------------------------------


def verdict_distribution(
    item_dists: Mapping[str, Mapping[str, float]],
    *,
    payment_risk: str,
    integrity_flag: bool = False,
) -> dict[str, float]:
    """
    EXACT distribution over verdicts induced by per-item option probabilities.

    The weighted twin of :func:`reachable_verdicts`: same p10 split, same
    subset-sum over the other ten items, carrying probability mass instead of
    mere reachability. Appended at the end of the module deliberately — the
    line numbers above are cited by ``docs/architecture.md``.

    **This exists only to give the "act anyway" baselines something to be
    optimal with respect to** (``experiments/decision_ablations.py``). The
    release rule never consults it, and it is not part of the certified path.

    ⚠ **It assumes the items are conditionally independent given the
    extraction** — which the certificate's union bound deliberately does NOT.
    That assumption is the price of having a distribution at all; it belongs to
    the acting baseline, never to the gate. A reviewer comparing the two must
    see that the baseline is the one buying an assumption, not us.

    Invariant, checked in the tests and re-checked at runtime by the harness:
    the support of this distribution equals ``reachable_verdicts(sets)`` where
    ``sets[i] = {o : item_dists[i][o] > 0}``. Options with zero mass are dropped
    rather than carried, which is what makes the identity exact.
    """
    if not isinstance(payment_risk, str):
        raise ValueError(
            "verdict_distribution needs a CONCRETE payment_risk (or marginalize "
            "under your own prior) — a set has no probability without one. The "
            "acting baselines always pass a concrete value.")
    if payment_risk not in PAYMENT_RISKS:
        raise ValueError(f"payment_risk {payment_risk!r} not within {PAYMENT_RISKS}")
    for it in ITEMS:
        d = item_dists.get(it)
        if not d:
            raise ValueError(f"item {it}: no distribution; verdict_distribution needs all {len(ITEMS)}")
        if not set(d) <= set(OPTIONS):
            raise ValueError(f"item {it}: {set(d)} not a subset of {OPTIONS}")
        if any(p < 0.0 for p in d.values()):
            raise ValueError(f"item {it}: negative probability")
        total = sum(d.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"item {it}: probabilities sum to {total!r}, not 1")

    if integrity_flag:
        return {"blocked": 1.0}

    out: dict[str, float] = {}

    s10 = item_dists["p10"]
    p_blocked = s10.get("c", 0.0)
    if p_blocked > 0.0:  # p10='c' -> integrity 'low' -> blocked, whatever else was said
        out["blocked"] = p_blocked

    non_c_p10 = [(o, p) for o, p in s10.items() if o != "c" and p > 0.0]
    if non_c_p10:
        # Weighted reachable partial sums from the other ten items.
        sums: dict[int, float] = {0: 1.0}
        for it in ITEMS:
            if it == "p10":
                continue
            nxt: dict[int, float] = {}
            for s, w in sums.items():
                for o, p in item_dists[it].items():
                    if p <= 0.0:
                        continue
                    key = s + SCORE_MAP[it][o]
                    nxt[key] = nxt.get(key, 0.0) + w * p
            sums = nxt

        for o, p10_p in non_c_p10:
            p10_pts = SCORE_MAP["p10"][o]  # a->2, b->1
            for s, w in sums.items():
                overall = overall_band(s + p10_pts)
                if overall == "high":
                    v = "needs_support" if payment_risk == "high" else "overseas_ready"
                elif overall == "medium":
                    v = "needs_support"
                else:
                    v = "high_risk"
                out[v] = out.get(v, 0.0) + p10_p * w

    return out
