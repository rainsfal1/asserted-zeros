"""Simulated session generation — the voice-constrained draw  (A2).

Why this exists: every committed session simulation (``m1_text_loop.py``,
``m2_calibration.sessions_for``, ``m2_calibration._simulate``) draws each item's
transcript independently from a variants index keyed on (item, utterance) only.
On the M3 pool each key spans all 8 accented voices, so P(a session is spoken by
one voice) is 8·(1/8)¹¹ ≈ 1.5e-9 — **every simulated session is a chimera of ~8
speakers**, and per-voice session metrics (per-group release rate, accuracy
given release — ledger row E-62) are not well-defined on them. This module adds
the missing draw: ``voice="IN"`` restricts the variants index to one voice, so a
session is elicited end-to-end in a single accent.

Deliberately a NEW generator with its OWN deterministic RNG sequence. The three
existing draws are pinned byte-for-byte by ``tests/test_session_draws_pinned.py``
and must not change; this one is pinned by its own golden test
(``tests/test_sessions.py``) so B1 sweeps cannot silently desync from A2's
committed per-voice numbers. ``DEFAULT_STRATA`` duplicates ``m1_text_loop.STRATA``
(src cannot import experiments); a test asserts tuple equality so drift is
impossible.

Emitted rows are a strict superset of the M1 ``sessions.json`` schema, adding
``voice`` and keeping ``reachable`` / ``forced_*`` / per-item ``transcript_id``
— everything ``verdict_cert.metrics`` needs (set_accuracy, the credal suite, the
binding rejoin) and everything ``m2_calibration._simulate`` drops.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping, Sequence

from .certificate import certify, reask_ranking
from .items import load_items
from .metrics import argmax_option
from .scorer import ITEMS, OPTIONS, placement_decision, total_score

#: The M1.5 re-ask policies (``m1_5_reask_loop.POLICIES``); see
#: :func:`reask_sessions`. A test asserts tuple equality so drift is impossible.
REASK_POLICIES: tuple[str, ...] = ("repeat", "rephrase", "ladder")

#: Mirrors m1_text_loop.STRATA — (name, count, sampler-constraint).
DEFAULT_STRATA: tuple = (
    ("uniform", 30, None),
    ("near_high_18", 40, (16, 19)),
    ("near_low_11", 30, (10, 13)),
    ("blocked_p10c", 20, "p10c"),
)


def sample_profile(rng: random.Random, constraint) -> dict[str, str]:
    """M1's stratified true-profile sampler (same logic, this module's RNG)."""
    while True:
        prof = {it: rng.choice(OPTIONS) for it in ITEMS}
        if constraint == "p10c":
            prof["p10"] = "c"
            return prof
        if constraint is None:
            if prof["p10"] == "c":
                continue
            return prof
        lo, hi = constraint
        if prof["p10"] != "c" and lo <= total_score(prof) <= hi:
            return prof


def _variant_index(records: Sequence[Mapping],
                   voice: str | None) -> dict[tuple[str, str], list[Mapping]]:
    """``(item, utterance) -> the bank records realizing it``, voice-restricted.

    Shared by :func:`draw_sessions` and :func:`reask_sessions` so a re-ask can
    never draw from a wider pool than the initial elicitation did.
    """
    variants: dict[tuple[str, str], list[Mapping]] = {}
    all_keys: set[tuple[str, str]] = set()
    for r in records:
        if r["kind"] != "bank":
            continue
        all_keys.add((r["item"], r["utterance"]))
        if voice is not None and r.get("voice") != voice:
            continue
        variants.setdefault((r["item"], r["utterance"]), []).append(r)
    if not variants:
        raise ValueError(f"no bank records for voice={voice!r}")
    if voice is not None and set(variants) != all_keys:
        # The paired-draw property (same profiles/utterances/level slots across
        # voices at one seed) rests on every voice covering the full key set:
        # rng.choice consumes a VARIABLE number of bits with list length, so a
        # missing cell would silently desynchronize the draw from other voices.
        # Fail loudly instead.
        missing = sorted(all_keys - set(variants))[:5]
        raise ValueError(
            f"voice={voice!r} is missing {len(all_keys) - len(variants)} "
            f"(item, utterance) cells the pool has for other voices — the "
            f"paired-draw property would silently break. First missing: {missing}")
    # Coverage is NOT enough (audit wave 2, finding 17): rng.choice consumes a
    # VARIABLE amount of the stream depending on list length (the rejection
    # loop in _randbelow), so two voices with the same keys but different
    # per-key record counts would desynchronize the paired draw and pass the
    # coverage check above. The property the draw actually needs is uniform
    # multiplicity — every key backed by the same number of records.
    sizes = {len(v) for v in variants.values()}
    if len(sizes) > 1:
        by_size = sorted((len(v), k) for k, v in variants.items())
        raise ValueError(
            f"non-uniform records per (item, utterance) cell for voice={voice!r}: "
            f"counts {sorted(sizes)} — e.g. {by_size[0][1]} has {by_size[0][0]} and "
            f"{by_size[-1][1]} has {by_size[-1][0]}. The paired draw needs uniform "
            "multiplicity; fix the pool, not the guard.")
    return variants


def draw_sessions(
    records: Sequence[Mapping],
    build: Callable[[Mapping], frozenset[str]],
    *,
    seed: int,
    voice: str | None = None,
    baseline_tag: str = "judge",
    strata: Sequence[tuple] = DEFAULT_STRATA,
) -> list[dict]:
    """Draw stratified sessions from a frozen record pool and certify each.

    ``records`` are ``m2_calibration.load_records`` rows (must carry ``voice``
    when ``voice=`` is requested). ``build`` maps one record to its uncertainty
    set — the whole B1 sweep axis (seating, combiner, threshold, spending) lives
    inside this callable. ``voice=None`` is chimera mode (M1-compatible
    semantics, this module's RNG); ``voice="IN"`` restricts every elicitation to
    that voice. ``baseline_tag`` names the extractor whose argmax forms the
    no-certificate forced baseline.
    """
    items = {it.id: it for it in load_items()}
    variants = _variant_index(records, voice)

    rng = random.Random(seed)
    out: list[dict] = []
    for stratum, count, constraint in strata:
        for _ in range(count):
            true = sample_profile(rng, constraint)
            true_verdict = placement_decision(true)
            sets: dict[str, frozenset[str]] = {}
            forced: dict[str, str] = {}
            forced_all: dict[str, dict[str, str]] = {}
            per_item: dict[str, dict] = {}
            for pid, opt in true.items():
                utts = [u for u in items[pid].options[opt].examples
                        if (pid, u) in variants]
                if not utts:
                    raise ValueError(f"no pooled variants for {pid}/{opt} "
                                     f"under voice={voice!r}")
                utt = rng.choice(utts)
                rec = rng.choice(variants[(pid, utt)])
                sets[pid] = build(rec)
                forced[pid] = argmax_option(rec["dists"][baseline_tag])
                # decree C12: the always-answer baseline is per-tag and plural —
                # an arm seating Gemma must not be scored against a Qwen
                # baseline. Every tag present in the pool gets its own profile.
                for t, d in rec["dists"].items():
                    forced_all.setdefault(t, {})[pid] = argmax_option(d)
                per_item[pid] = {"true": opt, "transcript_id": rec["id"],
                                 "set": sorted(sets[pid])}
            cert = certify(sets, payment_risk="low")
            forced_verdict = placement_decision(forced)
            out.append({
                "stratum": stratum,
                "voice": voice,
                "true_total": total_score(true),
                "true_verdict": true_verdict,
                "certified": cert.certified,
                "released_verdict": cert.verdict,
                "released_correct": cert.certified and cert.verdict == true_verdict,
                "released_wrong": cert.certified and cert.verdict != true_verdict,
                "forced_verdict": forced_verdict,
                "forced_wrong": forced_verdict != true_verdict,
                # per-tag always-answer baselines (C12); the scalar pair above
                # stays as the baseline_tag view so existing callers are unchanged
                "forced_wrong_by_tag": {
                    t: placement_decision(p) != true_verdict
                    for t, p in sorted(forced_all.items())},
                "reachable": sorted(cert.reachable),
                "pivotal": list(cert.pivotal_items),
                "items": per_item,
            })
    return out


# ---------------------------------------------------------------------------
# The within-session recovery loop  (B1)
# ---------------------------------------------------------------------------


def _reask_draw(policy: str, attempt: int, item: str, true_opt: str, cur_utt: str,
                used: set[str], utt_index: Mapping, variants: Mapping,
                rng: random.Random):
    """A fresh transcript for one re-ask, or None when the pool is exhausted.

    The respondent is re-answering: under ``repeat`` they say the same thing
    again (a fresh channel draw of the SAME wording); under ``rephrase`` they
    express the same true stance with a DIFFERENT bank wording; ``ladder``
    repeats first and rephrases thereafter (respondent-burden ordering). The
    alternative wording is chosen blind — never by which one the extractors
    would agree on (threat T2).

    Note the pool's hard limit: every (item, option) cell holds exactly two
    wordings, so ``rephrase`` has exactly one alternative and a second rephrase
    of the same item exhausts it.
    """
    mode = policy
    if policy == "ladder":
        mode = "repeat" if attempt == 0 else "rephrase"
    if mode == "repeat":
        cands = [r for r in variants[(item, cur_utt)] if r["id"] not in used]
        return rng.choice(cands) if cands else None
    others = [u for u in utt_index[(item, true_opt)] if u != cur_utt]
    if not others:
        return None
    utt = rng.choice(others)
    cands = [r for r in variants[(item, utt)] if r["id"] not in used]
    return rng.choice(cands) if cands else None


def reask_sessions(
    sessions: Sequence[Mapping],
    records: Sequence[Mapping],
    build: Callable[[Mapping, int], frozenset[str]],
    *,
    seed: int,
    policy: str = "rephrase",
    budget: int = 3,
    voice: str | None = None,
) -> list[dict]:
    """Run M1.5's withhold → re-ask → re-certify → escalate loop over a draw.

    ``sessions`` are :func:`draw_sessions` rows; ``records`` the same pool they
    were drawn from. Certified sessions pass through with ``n_reasks = 0``, so
    the returned list is the whole population and the A2 suite
    (``metrics.summarize_sessions``) applies to it unchanged — plus VM-08
    (``time_to_verdict``) and VM-09 (``burden_exchange``), which need the
    ``n_reasks`` field only this function emits.

    **``build`` takes the elicitation count.** ``build(record, k)`` returns the
    uncertainty set for an item's k-th elicitation, k = 1 for the one
    :func:`draw_sessions` already made. That signature exists because
    α-spending belongs to the CALLER (``calibration.SetRule.thresholds``): a
    loop that may re-elicit must price every elicitation at α/2^k *from the
    first*, so the initial draw must have used ``build(r, 1)`` too. Passing a
    k-independent builder is the M1.5 exploit, and is the deliberate arm of
    B1's re-ask axis rather than a bug — the report labels which arm is which.

    Three rules inherited from ``experiments/m1_5_reask_loop.py``, none of them
    free choices:

    - **REPLACE, never intersect.** The fresh elicitation replaces the item's
      set. Intersecting two 1−α sets can miss the truth at up to 2α — it would
      resolve far more sessions while silently trading the §2.3 guarantee away.
    - **The PSER-style stopping rule.** Escalate as soon as no single re-ask
      *could* shrink the reachable set (best gain in ``reask_ranking`` is 0)
      rather than burning the budget first. Our DP is exact, so this predicted
      gain is exact, not an estimate.
    - **The baseline never re-asks.** ``forced_*`` fields pass through
      untouched: the always-answer comparator answers off the first
      elicitation, which is what makes VM-09's exchange rate meaningful.
    """
    if policy not in REASK_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {REASK_POLICIES}")
    if budget < 0:
        raise ValueError("budget must be >= 0")
    variants = _variant_index(records, voice)
    utt_index: dict[tuple[str, str], list[str]] = {}
    by_id = {}
    for r in records:
        if r["kind"] != "bank":
            continue
        by_id[r["id"]] = r
        if voice is not None and r.get("voice") != voice:
            continue
        seen = utt_index.setdefault((r["item"], r["gold"]), [])
        if r["utterance"] not in seen:
            seen.append(r["utterance"])

    rng = random.Random(seed)
    out: list[dict] = []
    for sess in sessions:
        sets = {pid: frozenset(d["set"]) for pid, d in sess["items"].items()}
        per_item = {pid: dict(d) for pid, d in sess["items"].items()}
        cur_utt = {pid: by_id[d["transcript_id"]]["utterance"]
                   for pid, d in sess["items"].items()}
        used = {d["transcript_id"] for d in sess["items"].values()}
        attempts: dict[str, int] = {}
        rounds: list[dict] = []
        stop = None
        for _ in range(budget):
            if certify(sets, payment_risk="low").certified:
                stop = "resolved"
                break
            ranking = reask_ranking(sets, payment_risk="low")
            if not ranking or ranking[0][1] == 0:
                stop = "stopping_rule"
                break
            target = ranking[0][0]
            true_opt = per_item[target]["true"]
            rec = _reask_draw(policy, attempts.get(target, 0), target, true_opt,
                              cur_utt[target], used, utt_index, variants, rng)
            if rec is None:
                stop = "pool_exhausted"
                break
            attempts[target] = attempts.get(target, 0) + 1
            used.add(rec["id"])
            cur_utt[target] = rec["utterance"]
            # k counts elicitations of THIS item: draw_sessions made the 1st.
            new_set = build(rec, attempts[target] + 1)
            sets[target] = new_set              # REPLACE — see the docstring
            per_item[target] = {"true": true_opt, "transcript_id": rec["id"],
                                "set": sorted(new_set)}
            rounds.append({
                "item": target,
                "mode": ("repeat" if policy == "repeat"
                         else "rephrase" if policy == "rephrase"
                         else "repeat" if attempts[target] == 1 else "rephrase"),
                "transcript_id": rec["id"], "level": rec["level"],
                "wer": rec["wer"], "k": attempts[target] + 1,
                "new_set": sorted(new_set), "covers_true": true_opt in new_set,
            })
        cert = certify(sets, payment_risk="low")
        if stop is None:
            stop = "resolved" if cert.certified else "budget_exhausted"
        row = dict(sess)
        row.update({
            "certified": cert.certified,
            "released_verdict": cert.verdict,
            "released_correct": cert.certified and cert.verdict == sess["true_verdict"],
            "released_wrong": cert.certified and cert.verdict != sess["true_verdict"],
            "reachable": sorted(cert.reachable),
            "pivotal": list(cert.pivotal_items),
            "items": per_item,
            "n_reasks": len(rounds),
            "stop": stop,
            "escalated": not cert.certified,
            "policy": policy,
            "rounds": rounds,
        })
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# v2 re-ask loop — true exhaustion, honest labels (audit wave 2, F-6 exposure)
# ---------------------------------------------------------------------------


def _reask_draw_v2(policy: str, attempt: int, item: str, true_opt: str,
                   cur_utt: str, elicited_utts: set[str], used: set[str],
                   utt_index: Mapping, variants: Mapping,
                   rng: random.Random):
    """The v1 draw had two defects this fixes (research/audit-wave-2.md):

    1. **The toggle.** v1 computed rephrase alternatives against the CURRENT
       wording only, so with two wordings per cell it alternated 1 → 0 → 1,
       relabelling a same-wording redraw as a rephrase from k=3 on. v2 excludes
       every wording ALREADY ELICITED for the item, so `rephrase` genuinely
       exhausts and `pool_exhausted` is a reachable stop.
    2. **The label.** The caller records which wording was actually drawn and
       whether the draw was effectively a repeat or a rephrase — the trace
       carries `wording` and `mode_effective`, never just the policy name.

    Returns (record, mode_effective) or (None, None) on exhaustion.
    """
    mode = policy
    if policy == "ladder":
        mode = "repeat" if attempt == 0 else "rephrase"
    if mode == "repeat":
        cands = [r for r in variants[(item, cur_utt)] if r["id"] not in used]
        return (rng.choice(cands), "repeat") if cands else (None, None)
    fresh = [u for u in utt_index[(item, true_opt)] if u not in elicited_utts]
    if not fresh:
        return None, None
    utt = rng.choice(fresh)
    cands = [r for r in variants[(item, utt)] if r["id"] not in used]
    return (rng.choice(cands), "rephrase") if cands else (None, None)



def reask_sessions_v2(sessions: Sequence[Mapping], records: Sequence[Mapping],
                      build: Callable[[Mapping, int], frozenset[str]],
                      *, seed: int, policy: str = "rephrase", budget: int = 3,
                      voice: str | None = None,
                      payment_risk,
                      integrity_flag: bool = False) -> list[dict]:
    """The v2 withhold → re-ask → re-certify → escalate loop.

    Differences from :func:`reask_sessions` (v1, byte-frozen for B1/M1.5):

    - **True exhaustion**: `rephrase` excludes all wordings already elicited
      for the item; `pool_exhausted` actually fires.
    - **Honest labels**: every round logs the `wording` drawn and
      `mode_effective` alongside the policy.
    - **Exogenous inputs are explicit**: `payment_risk` / `integrity_flag`
      thread through to `certify` (v1 could not pass them at all — audit
      wave 2). The default "low" matches the deployed v1 semantics; pass
      `certificate.PAYMENT_RISK_UNKNOWN` to run the F-6 ignorance regime.
    """
    if policy not in REASK_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {REASK_POLICIES}")
    if budget < 0:
        raise ValueError("budget must be >= 0")
    variants = _variant_index(records, voice)
    utt_index: dict[tuple[str, str], list[str]] = {}
    by_id = {}
    for r in records:
        if r["kind"] != "bank":
            continue
        by_id[r["id"]] = r
        if voice is not None and r.get("voice") != voice:
            continue
        seen = utt_index.setdefault((r["item"], r["gold"]), [])
        if r["utterance"] not in seen:
            seen.append(r["utterance"])

    rng = random.Random(seed)
    out: list[dict] = []
    for sess in sessions:
        sets = {pid: frozenset(d["set"]) for pid, d in sess["items"].items()}
        per_item = {pid: dict(d) for pid, d in sess["items"].items()}
        cur_utt = {pid: by_id[d["transcript_id"]]["utterance"]
                   for pid, d in sess["items"].items()}
        elicited = {pid: {cur_utt[pid]} for pid in cur_utt}
        used = {d["transcript_id"] for d in sess["items"].values()}
        attempts: dict[str, int] = {}
        rounds: list[dict] = []
        stop = None
        for _ in range(budget):
            if certify(sets, payment_risk=payment_risk,
                       integrity_flag=integrity_flag).certified:
                stop = "resolved"
                break
            ranking = reask_ranking(sets, payment_risk=payment_risk,
                                    integrity_flag=integrity_flag)
            if not ranking or ranking[0][1] == 0:
                stop = "stopping_rule"
                break
            target = ranking[0][0]
            true_opt = per_item[target]["true"]
            rec, mode_eff = _reask_draw_v2(
                policy, attempts.get(target, 0), target, true_opt,
                cur_utt[target], elicited[target], used, utt_index, variants, rng)
            if rec is None:
                stop = "pool_exhausted"
                break
            attempts[target] = attempts.get(target, 0) + 1
            used.add(rec["id"])
            cur_utt[target] = rec["utterance"]
            elicited[target].add(rec["utterance"])
            new_set = build(rec, attempts[target] + 1)
            sets[target] = new_set  # REPLACE — same rule as v1, same reason
            per_item[target] = {"true": true_opt, "transcript_id": rec["id"],
                                "set": sorted(new_set)}
            rounds.append({
                "item": target,
                "mode": policy,
                "mode_effective": mode_eff,
                "wording": rec["utterance"],
                "transcript_id": rec["id"], "level": rec["level"],
                "wer": rec["wer"], "k": attempts[target] + 1,
                "new_set": sorted(new_set), "covers_true": true_opt in new_set,
            })
        cert = certify(sets, payment_risk=payment_risk, integrity_flag=integrity_flag)
        if stop is None:
            stop = "resolved" if cert.certified else "budget_exhausted"
        row = dict(sess)
        row.update({
            "certified": cert.certified,
            "released_verdict": cert.verdict,
            "released_correct": cert.certified and cert.verdict == sess["true_verdict"],
            "released_wrong": cert.certified and cert.verdict != sess["true_verdict"],
            "reachable": sorted(cert.reachable),
            "pivotal": list(cert.pivotal_items),
            "items": per_item,
            "n_reasks": len(rounds),
            "stop": stop,
            "escalated": not cert.certified,
            "policy": policy,
            "rounds": rounds,
        })
        out.append(row)
    return out
