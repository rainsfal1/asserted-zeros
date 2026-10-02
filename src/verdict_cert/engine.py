"""The v2 session engine — decision-function-parameterized, split-safe, honest labels.

Built per ``docs/v2-pool-preregistration.md`` §6.3, which is where its design decisions
are pre-registered; audit wave 3 established that the v1 loop could not serve (its
``certify`` is hardwired to the 11-item scorer, and its ``_variant_index`` machinery
assumes the bank pools' uniform multiplicity, which the Powergrading wording skew
rightly violates).

The three decisions §6.3 pinned, implemented here exactly:

- **Keying**: rephrase-exclusion keys on the **normalized wording** — "already elicited"
  means the words, not the record. ``repeat`` is *another channel draw of the same
  (student, item) response*, hash-shuffled per §7.3 (ascending draw order would bias
  repeat toward the clean channel), NOT via ``_variant_index``.
- **The decision function is a parameter** (``certify_fn`` / ``ranking_fn``) — the engine
  never imports a scorer, so the same loop serves USCIS now and any §7-portable rule later.
- **The cross-split rule is enforced, not caller discipline**: the engine takes
  ``allowed_students`` explicitly and raises on any drawn record outside it.

Record schema (one row per rendered channel draw):
``{"student", "item", "gold", "wording", "draw", "id", ...}`` — ``wording`` is the
normalized response text, ``gold`` ∈ {1, 0}, ``draw`` the channel-draw index.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping, Sequence


def _fnv(s: str) -> int:
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

REASK_POLICIES: tuple[str, ...] = ("repeat", "rephrase", "ladder")


def _index_pool(records: Sequence[Mapping], allowed_students: frozenset):
    """(student, item) -> draws in draw order; (item, gold) -> wording -> records."""
    by_response: dict[tuple, list[Mapping]] = {}
    by_gold: dict[tuple, dict[str, list[Mapping]]] = {}
    for r in records:
        if r["student"] not in allowed_students:
            raise ValueError(
                f"record {r.get('id', '?')!r}: student {r['student']!r} is outside the "
                "allowed split — the pool handed to the engine must already be "
                "split-filtered, and this raises so a leak cannot be silent "
                "(prereg §6.3, wave-3 finding 16)")
        by_response.setdefault((r["student"], r["item"]), []).append(r)
        by_gold.setdefault((r["item"], r["gold"]), {}).setdefault(
            r["wording"], []).append(r)
    for draws in by_response.values():
        draws.sort(key=lambda r: r["draw"])  # stable base order; repeat shuffles per §7.3
    return by_response, by_gold


def _draw(policy: str, attempt: int, item: str, gold, student: str,
          elicited_wordings: set[str], used_ids: set,
          by_response: Mapping, by_gold: Mapping, rng: random.Random):
    """One re-elicitation. Returns (record, mode_effective) or (None, None)."""
    mode = policy
    if policy == "ladder":
        mode = "repeat" if attempt == 0 else "rephrase"
    if mode == "repeat":
        cands = [r for r in by_response.get((student, item), ())
                 if r["id"] not in used_ids]
        if not cands:
            return None, None
        # Hash-shuffled, not ascending: draw order d1..d4 maps to clean->noisy
        # levels, so ascending would bias repeat toward the CLEAN channel —
        # a directional confound inside the policy contrast (prereg §7.3).
        cands.sort(key=lambda r: (_fnv(f"{r['id']}:repeat"), r["id"]))
        return cands[0], "repeat"
    wordings = by_gold.get((item, gold), {})
    fresh = sorted(w for w in wordings if w not in elicited_wordings)
    while fresh:
        w = fresh.pop(rng.randrange(len(fresh)))
        cands = [r for r in wordings[w] if r["id"] not in used_ids]
        if cands:
            return rng.choice(cands), "rephrase"
    return None, None


def run_sessions(
    sessions: Sequence[Mapping],
    records: Sequence[Mapping],
    build: Callable[[Mapping, int], frozenset],
    certify_fn: Callable[[Mapping], object],
    ranking_fn: Callable[[Mapping], list],
    *,
    allowed_students: frozenset,
    seed: int,
    policy: str,
    budget: int = 3,
) -> list[dict]:
    """The withhold → re-ask → re-certify → escalate loop, v2.

    ``sessions``: rows with ``student``, ``true_verdict`` and ``items`` mapping
    item -> {"gold", "wording", "record_id", "set", "draw"} for the first
    elicitations. ``certify_fn(sets)`` must return an object with
    ``certified``/``verdict``/``reachable``/``pivotal_items``; ``ranking_fn(sets)``
    a high-gain-first [(item, gain)] list. The v1 rules inherited unchanged:
    REPLACE never intersect; stop when no re-ask can shrink (exact gain 0);
    α-spending belongs to the caller via ``build(record, k)``.
    """
    if policy not in (*REASK_POLICIES, "none"):
        raise ValueError(f"unknown policy {policy!r}")
    if budget < 0:
        raise ValueError("budget must be >= 0")
    by_response, by_gold = _index_pool(records, allowed_students)

    rng = random.Random(seed)
    out: list[dict] = []
    for sess in sessions:
        student = sess["student"]
        if student not in allowed_students:
            raise ValueError(f"session student {student!r} outside the allowed split")
        sets = {it: frozenset(d["set"]) for it, d in sess["items"].items()}
        per_item = {it: dict(d) for it, d in sess["items"].items()}
        elicited = {it: {d["wording"]} for it, d in sess["items"].items()}
        used_ids = {d["record_id"] for d in sess["items"].values()}
        attempts: dict[str, int] = {}
        rounds: list[dict] = []
        stop = None
        stop_item = None      # which item's pool ran out — the §6.5 audit needs it
        for _ in range(budget if policy != "none" else 0):
            if certify_fn(sets).certified:
                stop = "resolved"
                break
            ranking = ranking_fn(sets)
            if not ranking or ranking[0][1] == 0:
                stop = "stopping_rule"
                break
            target = ranking[0][0]
            gold = per_item[target]["gold"]
            rec, mode_eff = _draw(policy, attempts.get(target, 0), target, gold,
                                  student, elicited[target], used_ids,
                                  by_response, by_gold, rng)
            if rec is None:
                stop = "pool_exhausted"
                stop_item = target   # exhaustion is per-ITEM, never the global budget
                break
            attempts[target] = attempts.get(target, 0) + 1
            used_ids.add(rec["id"])
            elicited[target].add(rec["wording"])
            new_set = build(rec, attempts[target] + 1)
            sets[target] = new_set  # REPLACE — same rule, same reason as v1
            per_item[target] = {"gold": gold, "wording": rec["wording"],
                                "record_id": rec["id"], "set": sorted(new_set),
                                "draw": rec["draw"]}
            rounds.append({
                "item": target, "mode": policy, "mode_effective": mode_eff,
                "wording": rec["wording"], "record_id": rec["id"],
                "student_drawn": rec["student"], "level": rec.get("level"),
                "draw": rec["draw"], "k": attempts[target] + 1,
                "new_set": sorted(new_set), "covers_gold": gold in new_set,
            })
        cert = certify_fn(sets)
        if stop is None:
            stop = "resolved" if cert.certified else (
                "no_loop" if policy == "none" else "budget_exhausted")
        row = dict(sess)
        row.update({
            "certified": cert.certified,
            "released_verdict": cert.verdict,
            "released_correct": cert.certified and cert.verdict == sess["true_verdict"],
            "released_wrong": cert.certified and cert.verdict != sess["true_verdict"],
            "reachable": sorted(cert.reachable),
            "pivotal": list(cert.pivotal_items),
            "items": per_item, "n_reasks": len(rounds), "stop": stop,
            "stop_item": stop_item, "attempts": dict(attempts),
            "escalated": not cert.certified, "policy": policy, "rounds": rounds,
        })
        out.append(row)
    return out
