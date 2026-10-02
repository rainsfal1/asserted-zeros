"""The USCIS decision function for the v2 pool — reachability, certification, ranking.

Ported from the frozen ``experiments/m6_powergrading.py`` (whose copy is byte-untouched:
M6 is a scored v1 record) per ``docs/v2-pool-preregistration.md`` §6.3. The rule: **pass
iff ≥ ``pass_mark`` of ``k`` items are answered correctly** — a counting rule, so exact
reachability needs only the min/max achievable correct-count (the O(k) "span shortcut"
audit wave 2 verified exhaustively on the m6 copy over 774,198 cases; this port carries
its own brute-force test).

Per-item uncertainty sets are subsets of {1, 0} (correct / incorrect); {1, 0} is maximum
ignorance and, exactly as in the v1 certificate, an EMPTY set is a caller error — the safe
encoding of "no evidence" is the full set, never a default answer.

No external dependencies; like ``certificate.py`` this file is meant to be read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

USCIS_OPTIONS: tuple[int, ...] = (1, 0)
PASS_MARK = 6

#: An uncertainty profile: item id -> non-empty subset of {1, 0}.
UscisSets = Mapping[str, frozenset]


def _validate(sets: UscisSets) -> None:
    if not sets:
        raise ValueError("no items")
    for it, s in sets.items():
        if not s:
            raise ValueError(f"item {it}: empty set; use {{1, 0}} for unknown")
        if not set(s) <= set(USCIS_OPTIONS):
            raise ValueError(f"item {it}: {set(s)} not a subset of {USCIS_OPTIONS}")


def reachable_verdicts(sets: UscisSets, *, pass_mark: int = PASS_MARK) -> set[str]:
    """EXACT reachable set over the product — via the correct-count span.

    The verdict is monotone in the correct-count, and every count between the
    min (items forced correct) and max (items allowed correct) is achievable by
    toggling free items one at a time — so the reachable verdict set is exactly
    determined by whether the span crosses ``pass_mark``.
    """
    _validate(sets)
    n_min = sum(1 for s in sets.values() if set(s) == {1})
    n_max = sum(1 for s in sets.values() if 1 in s)
    out: set[str] = set()
    if n_max >= pass_mark:
        out.add("pass")
    if n_min < pass_mark:
        out.add("fail")
    return out


@dataclass(frozen=True)
class UscisCertificate:
    certified: bool
    verdict: str | None
    reachable: frozenset
    pivotal_items: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.certified


def certify(sets: UscisSets, *, pass_mark: int = PASS_MARK) -> UscisCertificate:
    """Release iff exactly one verdict is reachable — the same gate as v1."""
    reachable = reachable_verdicts(sets, pass_mark=pass_mark)
    certified = len(reachable) == 1
    pivotal = () if certified else tuple(
        it for it, _g in reask_ranking(sets, pass_mark=pass_mark))
    return UscisCertificate(
        certified=certified,
        verdict=next(iter(reachable)) if certified else None,
        reachable=frozenset(reachable),
        pivotal_items=pivotal,
    )


def reask_ranking(sets: UscisSets, *, pass_mark: int = PASS_MARK) -> list[tuple[str, int]]:
    """Best-case |V| reduction per unresolved item — v1's formula, this rule."""
    base = len(reachable_verdicts(sets, pass_mark=pass_mark))
    ranked: list[tuple[str, int]] = []
    for it in sets:
        if len(sets[it]) <= 1:
            continue
        best_after = min(
            len(reachable_verdicts({**sets, it: frozenset({o})}, pass_mark=pass_mark))
            for o in sets[it])
        ranked.append((it, base - best_after))
    ranked.sort(key=lambda kv: (-kv[1], str(kv[0])))
    return ranked


def verdict_of(correct: Mapping[str, int], *, pass_mark: int = PASS_MARK) -> str:
    """The concrete rule on a resolved profile."""
    return "pass" if sum(correct.values()) >= pass_mark else "fail"


def uscis_margin(n_correct: int, *, pass_mark: int = PASS_MARK, k: int = 10) -> int:
    """Distance-to-cut for the decision-ablations B-ii arm on the v2 pool
    (prereg §6.6 — `metrics.verdict_margin` hardcodes the v1 cuts and does not
    port). Minimal number of item flips that changes the verdict."""
    if not 0 <= n_correct <= k:
        raise ValueError(f"n_correct {n_correct} outside 0..{k}")
    if n_correct >= pass_mark:
        return n_correct - pass_mark + 1
    return pass_mark - n_correct


def sets_from_dists(dists: Mapping[str, Mapping[int, float]],
                    threshold: float) -> dict[str, frozenset]:
    """Per-item conformal sets with the normative empty→FULL semantics."""
    out: dict[str, frozenset] = {}
    for it, sc in dists.items():
        keep = frozenset(o for o, s in sc.items() if s <= threshold)
        out[it] = keep if keep else frozenset(USCIS_OPTIONS)
    return out


def sequence(items: Sequence[str]) -> tuple[str, ...]:
    """Stable item order for deterministic iteration."""
    return tuple(sorted(items))
