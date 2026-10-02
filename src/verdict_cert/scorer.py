"""
The deployed scorer — transcribed verbatim and made self-contained.

This is the *known, deterministic decision function* the certificate reasons
against. It is copied, not imported, so this repository stands alone and the
scorer the certificate protects is pinned to an auditable snapshot.

The source of truth is a deployed screening service, whose scoring map and
placement decision are transcribed here from its webhook handler. The service
is not identified: what matters for this paper is that the decision function is
fixed, deterministic and not written by us.

Kept as pure, dependency-free functions so the certificate can enumerate and
reason about the decision boundaries exactly.
"""

from __future__ import annotations

from typing import Mapping

ITEMS: tuple[str, ...] = tuple(f"p{i:02d}" for i in range(1, 12))
OPTIONS: tuple[str, ...] = ("a", "b", "c")

# a/b/c -> points, verbatim. Note the two non-standard items:
#   p09 scores b=2 (a and b are both "correct"); p11 is reverse-keyed (c is best).
SCORE_MAP: dict[str, dict[str, int]] = {
    "p01": {"a": 2, "b": 1, "c": 0},
    "p02": {"a": 2, "b": 1, "c": 0},
    "p03": {"a": 2, "b": 1, "c": 0},
    "p04": {"a": 2, "b": 1, "c": 0},
    "p05": {"a": 2, "b": 1, "c": 0},
    "p06": {"a": 2, "b": 1, "c": 0},
    "p07": {"a": 2, "b": 1, "c": 0},
    "p08": {"a": 2, "b": 1, "c": 0},
    "p09": {"a": 2, "b": 2, "c": 0},
    "p10": {"a": 2, "b": 1, "c": 0},  # integrity item
    "p11": {"a": 1, "b": 0, "c": 2},  # reverse-keyed
}

INTEGRITY_ITEM = "p10"
PSYCHOMETRIC_MAX = 22
VERDICTS: tuple[str, ...] = ("blocked", "high_risk", "needs_support", "overseas_ready")


def total_score(answers: Mapping[str, str]) -> int:
    """Sum of item points for a complete 11-item answer profile."""
    return sum(SCORE_MAP[it][answers[it]] for it in ITEMS)


def overall_band(total: int) -> str:
    """total -> {high, medium, low}. Thresholds verbatim: >=18 high, <=11 low."""
    return "high" if total >= 18 else "low" if total <= 11 else "medium"


def integrity_band(p10_option: str) -> str:
    """p10 option -> {high, medium, low}; 'low' (option c, 0 pts) is disqualifying."""
    s = SCORE_MAP["p10"][p10_option]
    return "high" if s == 2 else "low" if s == 0 else "medium"


def placement_decision(
    answers: Mapping[str, str],
    *,
    payment_risk: str = "low",
    integrity_flag: bool = False,
) -> str:
    """
    The full deployed verdict function over a complete answer profile.

    Mirrors the deployed placement decision:
      - a failed integrity signal (voice-agent flag OR p10 == 'c') is a hard block;
      - otherwise the overall band routes the verdict, with payment_risk == 'high'
        demoting an otherwise-ready candidate to needs_support.

    ``answers`` maps each of p01..p11 to one of 'a'/'b'/'c'.
    """
    if integrity_flag or integrity_band(answers["p10"]) == "low":
        return "blocked"
    overall = overall_band(total_score(answers))
    if overall == "high":
        return "needs_support" if payment_risk == "high" else "overseas_ready"
    if overall == "medium":
        return "needs_support"
    return "high_risk"
