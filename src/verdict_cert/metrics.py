"""Canonical evaluation metrics — the one place a rate is defined  (A2).

Before this module, the same quantities were computed independently across the
harnesses, and not always the same way. The inventory that forced each decision
(spec: ``docs/metrics.md``, decrees C1–C10):

* **Release rate** had five implementations; **risk (wrong among released)** had
  five under three names — and three of those silently return ``0.0`` when
  nothing was released (``f9_extractor_ablation.py:64``, ``m2_calibration.py:997``,
  ``m6_powergrading.py:316``), so a zero-release operating point reads as
  *risk-free* and passes F9's frontier feasibility filter. Here an empty
  denominator is NaN (``null`` in JSON), never 0.0 — see :class:`Frac`.
* **Error rates come in two units** and both must always travel together:
  rewrite-spec §1.4 shows E1b's prediction ("a worse judge never produces more
  wrong released verdicts") is true as a *count per population* and false as a
  *rate among released*. :class:`ErrorRates` makes the pair inseparable.
* **"Coverage" has three live semantics** in the committed results: the raw
  conformal event (M2 phases 4–5), coverage through a set builder with
  empty→FULL semantics (``calibration.conformal_set``), and coverage through the
  legacy builder with an empty→argmin fallback (now quarantined outside the
  library in ``experiments/_legacy.py``, used
  by M6). All three stay expressible — :func:`coverage_event` takes a score and
  threshold; :func:`coverage_set` takes the caller's builder, empty-set policy
  included — and every reported number must say which it is.
* **The per-group axis** (FAccT / rewrite-spec §5.1–5.2, ledger row E-62) did
  not exist anywhere: :func:`by_group` wraps any metric here over any key.
* **The credal protocol** is adopted from Corani & Zaffalon (JMLR 2008 §4.2) and
  Zaffalon, Corani & Mauá (IJAR 2012) rather than invented: determinacy IS the
  release rate, single accuracy IS accuracy-given-release, and the discounted
  utility u_w(k) = a/k + b/k² (a = 4w−1, b = 2−4w) prices an indeterminate
  answer of size k, with u65/u80 the two standard operating points.
* **The binding decomposition** (rewrite-spec §5.6) answers the psychometric
  reviewer's question — how much of our withholding would a plain
  distance-from-threshold rule reproduce? — via the margin form over the pinned
  scorer's cut points. δ is a sensitivity band, not a cut-score SE estimate.

Pure stdlib. Not re-exported from ``__init__`` — import as
``from verdict_cert import metrics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Mapping, Sequence

from .scorer import OPTIONS

NAN = float("nan")


# ---------------------------------------------------------------------------
# The two-unit primitive (decrees C1, C2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frac:
    """A ratio that remembers its denominator.

    ``rate`` is NaN when the denominator is empty — never 0.0, which is a claim
    ("no errors") the data does not support. ``jsonable`` renders NaN as null:
    ``json.dumps(float("nan"))`` emits bare ``NaN``, which violates RFC 8259.
    """

    num: float
    den: float

    @property
    def rate(self) -> float:
        return self.num / self.den if self.den else NAN

    def jsonable(self) -> dict:
        r = self.rate
        return {"num": self.num, "den": self.den, "rate": None if r != r else r}


@dataclass(frozen=True)
class ErrorRates:
    """One error count, both units — they can never desynchronize because they
    share the numerator (decree C2 / rewrite-spec §1.4)."""

    per_population: Frac
    among_released: Frac

    def jsonable(self) -> dict:
        return {"per_population": self.per_population.jsonable(),
                "among_released": self.among_released.jsonable()}


def jsonable(obj: object) -> object:
    """Recursively convert Frac/dataclass results and NaN to JSON-safe values."""
    if hasattr(obj, "jsonable"):
        return obj.jsonable()  # type: ignore[union-attr]
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def feasible(rate: float | Frac, target: float) -> bool:
    """Does a risk clear a tolerance? NaN (nothing released) is NOT feasible —
    the f9 ``best_at`` filter treated the old silent 0.0 as risk-free."""
    r = rate.rate if isinstance(rate, Frac) else rate
    return r == r and r <= target


# ---------------------------------------------------------------------------
# Session metrics (rows: M1 sessions.json schema or a superset)
# ---------------------------------------------------------------------------


def _released_wrong(s: Mapping) -> bool:
    if "released_wrong" in s:
        return bool(s["released_wrong"])
    return bool(s["certified"]) and not s["released_correct"]


def release_rate(sessions: Sequence[Mapping]) -> Frac:
    """VM-01 · determinacy: sessions with exactly one reachable verdict, over
    ALL sessions in the arm (decree C4 — distinct from post-loop resolution)."""
    return Frac(sum(bool(s["certified"]) for s in sessions), len(sessions))


def verdict_errors(sessions: Sequence[Mapping]) -> ErrorRates:
    """VM-02 · certified-and-wrong, in both units."""
    wrong = sum(_released_wrong(s) for s in sessions)
    released = sum(bool(s["certified"]) for s in sessions)
    return ErrorRates(Frac(wrong, len(sessions)), Frac(wrong, released))


def risk_among_released(sessions: Sequence[Mapping]) -> Frac:
    """VM-02's rate unit alone — the canonical name for what the harnesses call
    ``risk`` / ``wrong_of_released`` / ``verdict_error_released``."""
    return verdict_errors(sessions).among_released


def accuracy_given_release(sessions: Sequence[Mapping]) -> Frac:
    """VM-03 · P(correct | released) — ledger row E-62's definition (decree C8).
    Per-group, this is the FAccT metric: the group withheld from most gets the
    weakest guarantee on the verdicts it does receive."""
    released = [s for s in sessions if s["certified"]]
    return Frac(sum(not _released_wrong(s) for s in released), len(released))


def escalation_rate(sessions: Sequence[Mapping]) -> Frac:
    """VM-04 · sessions ending with no released verdict. At the gate (no loop)
    this equals the withhold rate; under current code escalation and
    no-verdict-ever are aliases (architecture finding F-7)."""
    return Frac(sum(not s["certified"] for s in sessions), len(sessions))


def set_accuracy(sessions: Sequence[Mapping]) -> Frac:
    """VM-05 · true verdict ∈ V(S) over the WITHHELD sessions — the credal
    protocol's set-accuracy, which is the accuracy of the *indeterminate*
    outputs (Corani & Zaffalon 2008 §4.2) and therefore shares VM-06's
    denominator. (2026-08-09 audit amendment: an earlier draft computed this
    over ALL sessions, which is a different quantity — see VM-22.)"""
    withheld = [s for s in sessions if not s["certified"]]
    return Frac(sum(s["true_verdict"] in s["reachable"] for s in withheld),
                len(withheld))


def verdict_set_coverage(sessions: Sequence[Mapping]) -> Frac:
    """VM-22 · true verdict ∈ V(S) over ALL sessions. NOT the credal protocol's
    set-accuracy — that is VM-05, conditioned on withheld.

    ⚠ This is NOT 1 − VM-02's per-population rate, despite what this docstring
    and docs/metrics.md both claimed until 2026-08-15. The complement of
    {true ∈ V} is {released ∧ wrong} ∪ {withheld ∧ true ∉ V}; the identity holds
    only when VM-05 set-accuracy is exactly 1. Verified broken on 10 committed
    cells. Do not derive one from the other."""
    return Frac(sum(s["true_verdict"] in s["reachable"] for s in sessions), len(sessions))


def indeterminate_output_size(sessions: Sequence[Mapping]) -> Frac:
    """VM-06 · mean |V(S)| over WITHHELD sessions — the verdict-layer set size,
    the credal protocol's "indeterminate output size" and the natural re-ask
    target. NOT the item-set size |S_i| (decree C9)."""
    withheld = [s for s in sessions if not s["certified"]]
    return Frac(sum(len(s["reachable"]) for s in withheld), len(withheld))


def discounted_utility(sessions: Sequence[Mapping], *, w: float,
                       unsanctioned: bool = False) -> Frac:
    """Utility-discounted accuracy: ZCM's quadratic interpolation through
    u(1) = 1, u(2) = w — u_w(k) = a/k + b/k², a = 4w−1, b = 2−4w (Zaffalon,
    Corani & Mauá 2012). A released verdict scores 1 if correct, 0 if wrong; a
    withheld session scores u_w(|V|) if the truth is in V(S), 0 otherwise. Only
    w = 0.65 and w = 0.80 are sanctioned. Two stated limits (docs/metrics.md
    §4): the utility prices withholding as a TERMINAL set output — this
    system's withheld sessions instead enter the recovery loop, which the score
    does not credit — and it is verdict-symmetric, pricing all wrong releases
    identically. Assumes release ⟺ |V| = 1; a withheld row with |V| < 2 means
    the caller's release rule disagrees with the certificate's, and scoring it
    u(1) = 1 would award full credit for releasing nothing — refused."""
    if not unsanctioned and w not in (0.65, 0.80):
        # Only these two weights are sanctioned (docs/metrics.md §4); a caller
        # passing 65 instead of 0.65 previously got silent garbage (u=65.0).
        # Research sweeps outside the sanction pass unsanctioned=True, which is
        # greppable — audit wave 2, finding 19.
        raise ValueError(f"w={w!r} is not a sanctioned weight (0.65 or 0.80); "
                         "pass unsanctioned=True for research sweeps")
    a, b = 4 * w - 1, 2 - 4 * w
    total = 0.0
    for s in sessions:
        if s["certified"]:
            total += 0.0 if _released_wrong(s) else 1.0
            continue
        k = len(s["reachable"])
        if k < 2:
            raise ValueError(
                f"withheld session with |V(S)| = {k}: the release rule that "
                "produced this row disagrees with |V|=1 release; "
                "discounted_utility's scoring would be wrong on it")
        if s["true_verdict"] in s["reachable"]:
            total += a / k + b / (k * k)
    return Frac(total, len(sessions))


@dataclass(frozen=True)
class CredalSuite:
    """VM-07 · the Corani–Zaffalon protocol, on our verdict sets."""

    determinacy: Frac
    single_accuracy: Frac
    set_accuracy: Frac
    indeterminate_output_size: Frac
    u65: Frac
    u80: Frac

    def jsonable(self) -> dict:
        return {k: getattr(self, k).jsonable() for k in (
            "determinacy", "single_accuracy", "set_accuracy",
            "indeterminate_output_size", "u65", "u80")}


def credal_suite(sessions: Sequence[Mapping]) -> CredalSuite:
    return CredalSuite(
        determinacy=release_rate(sessions),
        single_accuracy=accuracy_given_release(sessions),
        set_accuracy=set_accuracy(sessions),
        indeterminate_output_size=indeterminate_output_size(sessions),
        u65=discounted_utility(sessions, w=0.65),
        u80=discounted_utility(sessions, w=0.80),
    )


# ---------------------------------------------------------------------------
# Session-run (burden) metrics (rows: re-ask traces)
# ---------------------------------------------------------------------------


def time_to_verdict(runs: Sequence[Mapping]) -> Frac:
    """VM-08 · re-asks needed per RESOLVED session-run. Rows need ``certified``
    and ``n_reasks``. (Wall-clock time is data-blocked — spec §6.)"""
    resolved = [r for r in runs if r["certified"]]
    return Frac(sum(r["n_reasks"] for r in resolved), len(resolved))


def burden_exchange(runs: Sequence[Mapping]) -> Frac:
    """VM-09 · re-asks per avoided wrong verdict (proposal E3's exchange rate).
    "Avoided" := the run would have released a wrong verdict without the gate
    (``forced_wrong``) and did not release one with it (decree C10). The
    numerator is ALL re-asks — a gross ratio, no baseline subtraction: runs
    that never risked a wrong verdict still pay into it. Rows need
    ``n_reasks``, ``forced_wrong``, and the released-wrong fields."""
    avoided = sum(bool(r["forced_wrong"]) and not _released_wrong(r) for r in runs)
    return Frac(sum(r["n_reasks"] for r in runs), avoided)


# ---------------------------------------------------------------------------
# Record (item-elicitation) metrics
# ---------------------------------------------------------------------------


def argmax_option(dist: Mapping, options: Sequence = OPTIONS):
    """Deterministic argmax: ties break to the FIRST option in ``options``
    order (decree C5). Pass the decision function's own option space for
    non-{a,b,c} domains (e.g. M6's {1, 0}) — the default is scorer-specific.
    A dist sharing no key with ``options`` (empty, or foreign-keyed through a
    wrong ``options``) would silently return ``options[0]`` — refused."""
    if not any(o in dist for o in options):
        raise ValueError(f"dist keys {sorted(map(str, dist))[:4]} share no "
                         f"option with {tuple(options)}")
    return max(options, key=lambda o: dist.get(o, 0.0))


def extractor_accuracy(records: Sequence[Mapping], tag: str) -> Frac:
    """VM-10 · argmax-of-distribution vs gold, for one extractor tag."""
    return Frac(sum(argmax_option(r["dists"][tag]) == r["gold"] for r in records),
                len(records))


def phi(a: Sequence[bool], b: Sequence[bool]) -> float:
    """VM-11 · phi coefficient between two error-indicator vectors; NaN when a
    margin is empty. (Canonical home of ``m2_calibration._phi`` /
    ``s3_independence.phi``.)"""
    if len(a) != len(b):
        raise ValueError("error vectors differ in length")
    n11 = sum(x and y for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum((not x) and y for x, y in zip(a, b))
    n00 = sum((not x) and (not y) for x, y in zip(a, b))
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / den if den else NAN


def coverage_set(records: Sequence[Mapping],
                 build: Callable[[Mapping], frozenset[str]]) -> Frac:
    """VM-12a · gold ∈ build(record) — POLICY coverage, through the caller's set
    builder INCLUDING its empty-set policy. Pass a builder over
    ``calibration.conformal_set`` for empty→full semantics, over
    the quarantined legacy empty→argmin builder, or M1's union rule —
    all three are committed semantics and must stay distinguishable (decree C3)."""
    return Frac(sum(r["gold"] in build(r) for r in records), len(records))


def coverage_event(records: Sequence[Mapping],
                   score: Callable[[Mapping], Mapping[str, float]],
                   threshold: float) -> Frac:
    """VM-12b · the raw conformal event score(gold) ≤ t — what a calibrated
    threshold actually controls, before any empty-set policy (the M2 phase-4/5
    semantics; rationale at ``m2_calibration.py:1079``)."""
    return Frac(sum(score(r)[r["gold"]] <= threshold for r in records), len(records))


def informative_coverage(records: Sequence[Mapping],
                         build: Callable[[Mapping], frozenset],
                         *, n_options: int = len(OPTIONS)) -> Frac:
    """VM-12c · coverage conditional on the set being informative
    (|S| < ``n_options``). Pass the decision function's own option count for
    non-3-option domains — with the default, a 2-option (M6-style) full set
    would wrongly count as informative."""
    hits = n = 0
    for r in records:
        s = build(r)
        if len(s) < n_options:
            n += 1
            hits += r["gold"] in s
    return Frac(hits, n)


def floor_alpha(n_groups: int) -> float:
    """VM-13 · the per-item conformal floor 1/(n+1); n counts distinct
    exchangeability GROUPS, never rows (decree C6)."""
    return 1.0 / (n_groups + 1)


def sigma_floor(per_item_groups: Mapping[str, int]) -> float:
    """VM-14 · Σᵢ 1/(nᵢ+1) — the smallest honest Σᵢαᵢ the pool can license."""
    return sum(floor_alpha(n) for n in per_item_groups.values())


def mondrian_floor(records: Sequence[Mapping],
                   group_of: Callable[[Mapping], str],
                   unit_of: Callable[[Mapping], str] = lambda r: r["utterance"],
                   ) -> dict[str, float]:
    """VM-15 · per-group calibration floor, POOLED ACROSS ITEMS ONLY (decree C6;
    rewrite-spec §5.5: per (item, group) the floor is vacuous by construction —
    6 units per cell on the M3 pool gives Σ ≥ 1.57)."""
    units: dict[str, set[str]] = {}
    for r in records:
        units.setdefault(group_of(r), set()).add(unit_of(r))
    return {g: floor_alpha(len(u)) for g, u in sorted(units.items())}


# ---------------------------------------------------------------------------
# Grader-agreement metrics (rows: plain dicts with ``gold`` and ``grades``;
# this module NEVER imports the M6 harness — the harness adapts its rows)
# ---------------------------------------------------------------------------


def grader_accuracy(rows: Sequence[Mapping], g: int) -> Frac:
    """VM-16 · one human grader's agreement with the majority gold."""
    return Frac(sum(r["grades"][g] == r["gold"] for r in rows), len(rows))


def grader_coverage(rows_with_sets: Sequence[Mapping], g: int) -> Frac:
    """VM-17 · grader g's label ∈ the conformal set (rows carry ``set``) — does
    the set straddle human disagreement, not just the majority vote?"""
    return Frac(sum(r["grades"][g] in r["set"] for r in rows_with_sets),
                len(rows_with_sets))


def human_pairwise_agreement(rows: Sequence[Mapping]) -> Frac:
    """VM-18 · fraction of grader pairs agreeing, pooled over rows — the
    human–human ceiling any machine-agreement number must be read against."""
    hits = n = 0
    for r in rows:
        grades = r["grades"]
        for i in range(len(grades)):
            for j in range(i + 1, len(grades)):
                n += 1
                hits += grades[i] == grades[j]
    return Frac(hits, n)


def fleiss_kappa(tables: Sequence[Sequence[int]]) -> float:
    """VM-19 · Fleiss' κ from per-subject category-count rows (each row sums to
    the constant rater count). NaN when chance agreement is 1."""
    if not tables:
        return NAN
    n = sum(tables[0])
    if n < 2:
        raise ValueError("Fleiss' kappa needs at least two ratings per subject")
    big_n = len(tables)
    n_cat = len(tables[0])
    totals = [0] * n_cat
    p_i = []
    for row in tables:
        if sum(row) != n:
            raise ValueError("every subject needs the same number of ratings")
        for c, x in enumerate(row):
            totals[c] += x
        p_i.append(sum(x * (x - 1) for x in row) / (n * (n - 1)))
    p_bar = sum(p_i) / big_n
    p_e = sum((t / (big_n * n)) ** 2 for t in totals)
    return (p_bar - p_e) / (1 - p_e) if p_e != 1 else NAN


@dataclass(frozen=True)
class GraderVerdicts:
    """VM-20 · session-level verdicts under each grader's labels."""

    n_subjects: int
    pairwise_agreement: Frac   # grader-pair verdict agreement (the ceiling)
    unanimous: Frac            # all graders' verdicts coincide
    per_grader: tuple[dict, ...]  # verdict counts per grader (labels from verdict_fn)

    def jsonable(self) -> dict:
        return {"n_subjects": self.n_subjects,
                "pairwise_agreement": self.pairwise_agreement.jsonable(),
                "unanimous": self.unanimous.jsonable(),
                "per_grader": list(self.per_grader)}


def grader_verdict_agreement(
    per_subject_grades: Mapping[str, Mapping[str, Sequence[int]]],
    verdict_fn: Callable[[Mapping[str, int]], str],
) -> GraderVerdicts:
    """VM-20 · re-derive each subject's verdict under EACH grader's labels and
    measure how often humans would have agreed on the verdict itself. The
    decision function is injected (``verdict_fn``) so this module stays free of
    any harness import. Caller contract: every subject must carry the SAME
    grader count (enforced) and a COMPLETE item profile for ``verdict_fn``
    (not checkable here — item sets may legitimately differ per decision
    function; pre-filter to complete profiles as the a2 driver does)."""
    subjects = sorted(per_subject_grades)
    n_graders = None
    pair_hits = pair_n = unanimous = 0
    counts: list[dict[str, int]] = []
    for subj in subjects:
        grades = per_subject_grades[subj]
        gs = {len(v) for v in grades.values()}
        if len(gs) != 1:
            raise ValueError(f"{subj}: inconsistent grader counts {gs}")
        g_count = gs.pop()
        if n_graders is None:
            n_graders = g_count
            counts = [{} for _ in range(g_count)]
        elif g_count != n_graders:
            # Mixing subjects with different grader counts silently skews the
            # pair denominator and truncates per_grader — refuse instead.
            raise ValueError(f"{subj}: {g_count} graders, earlier subjects had "
                             f"{n_graders}")
        verdicts = [verdict_fn({it: v[g] for it, v in grades.items()})
                    for g in range(g_count)]
        for g, v in enumerate(verdicts):
            counts[g][v] = counts[g].get(v, 0) + 1
        for i in range(g_count):
            for j in range(i + 1, g_count):
                pair_n += 1
                pair_hits += verdicts[i] == verdicts[j]
        unanimous += len(set(verdicts)) == 1
    return GraderVerdicts(
        n_subjects=len(subjects),
        pairwise_agreement=Frac(pair_hits, pair_n),
        unanimous=Frac(unanimous, len(subjects)),
        per_grader=tuple(counts),
    )


# ---------------------------------------------------------------------------
# Binding decomposition (VM-21, rewrite-spec §5.6)
# ---------------------------------------------------------------------------

#: The pinned scorer's band edges, re-declared because scorer.py is untouchable
#: (it is the verbatim copy of the deployed rule). tests/test_metrics.py probes
#: every total 0..22 against scorer.overall_band so these cannot drift.
CUT_HIGH = 18
CUT_LOW = 11


def verdict_margin(total: int) -> int:
    """Minimal joint shift of both band edges that changes ``overall_band(total)``.

    Margin form: shifting both cuts by d puts the high edge at CUT_HIGH+d and
    the low edge at CUT_LOW+d; the band flips for some |d| ≤ δ iff the total
    sits within δ of an edge in the flipping direction. Equivalence with the
    exhaustive shifted-cut evaluation is proved by brute force in the tests.
    """
    if total >= CUT_HIGH:
        return total - CUT_HIGH + 1
    if total <= CUT_LOW:
        return CUT_LOW - total + 1
    return min(CUT_HIGH - total, total - CUT_LOW)


def cut_sensitive(total: int, *, delta: int) -> bool:
    """Would the argmax profile's band change under some cut shift in ±delta?"""
    return verdict_margin(total) <= delta


@dataclass(frozen=True)
class BindingDecomposition:
    """VM-21 · the 2×2 per δ: (withheld vs released) × (cut-sensitive vs
    cut-stable at the judge-argmax profile). The psychometric diagnostic:
    ``withheld_cut_stable / withheld`` is the share of our withholding a plain
    distance-from-threshold rule could NOT reproduce — the elicitation-binding
    fraction. δ is a sensitivity band, not a cut-score SE estimate.

    ⚠ Every cell is ``Frac(count, n_ALL_sessions)``, so ``.rate`` is NOT that
    quotient — the elicitation-binding fraction must be computed as
    ``withheld_cut_stable.num / (withheld_cut_stable.num + withheld_cut_sensitive.num)``.
    Taking ``.rate`` at face value gives it scaled by the withhold rate (on A2's
    M1 arm: 1.7% instead of 25%). Both consumers do this correctly; the note is
    here because the wording above invited the error (audit wave 2, 2026-08-15)."""

    n: int
    by_delta: Mapping[int, Mapping[str, Frac]]

    def jsonable(self) -> dict:
        return {"n": self.n,
                "by_delta": {str(d): {k: f.jsonable() for k, f in cells.items()}
                             for d, cells in self.by_delta.items()}}


def binding_decomposition(
    sessions: Sequence[Mapping],
    argmax_total_of: Callable[[Mapping], int | None],
    *,
    deltas: Sequence[int] = (1, 2),
) -> BindingDecomposition:
    """``argmax_total_of`` maps a session to its judge-argmax total score, or
    None when the argmax profile is blocked — the integrity gate has no cut, so
    blocked profiles are cut-stable by decree (C10). Pairing the sessions with
    the right extraction pool is the CALLER's obligation: a mismatched pool
    surfaces as whatever ``argmax_total_of`` raises (typically a KeyError on a
    transcript id) — this function does not catch it."""
    n = len(sessions)
    totals = [argmax_total_of(s) for s in sessions]
    by_delta: dict[int, dict[str, Frac]] = {}
    for d in deltas:
        cells = {"withheld_cut_sensitive": 0, "withheld_cut_stable": 0,
                 "released_cut_sensitive": 0, "released_cut_stable": 0}
        for s, total in zip(sessions, totals):
            sens = total is not None and cut_sensitive(total, delta=d)
            row = "released" if s["certified"] else "withheld"
            col = "cut_sensitive" if sens else "cut_stable"
            cells[f"{row}_{col}"] += 1
        by_delta[d] = {k: Frac(v, n) for k, v in cells.items()}
    return BindingDecomposition(n=n, by_delta=by_delta)


# ---------------------------------------------------------------------------
# The group axis
# ---------------------------------------------------------------------------


def voice_of(transcript_id: str) -> str:
    """The voice code embedded in an M3 transcript id (``p01:0:IN:tel:s1`` → ``IN``).

    S3 ids carry no voice (``p01:0:tel:s1``, four fields) and would silently
    yield the CHANNEL LEVEL as a voice — a per-group table that looks right and
    is wrong. Arity-checked instead."""
    parts = transcript_id.split(":")
    if len(parts) != 5:
        raise ValueError(
            f"{transcript_id!r} is not a voiced (M3-style) transcript id: "
            f"{len(parts)} fields, expected 5 (item:idx:VOICE:level:seed). "
            "The S3 pool has no voice axis.")
    return parts[2]


def seed_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    """Cross-seed dispersion for one metric (decree C11): mean, sd, min, max
    over the DEFINED (non-NaN) per-seed values, plus ``n_defined`` — the count
    of seeds contributing. For frontier-style metrics an undefined seed means
    no feasible operating point existed, so ``n_defined`` is the feasible-seed
    count (f9's ``n_feasible``); for plain rates it means an empty denominator.
    Never averages NaN away silently — the count always travels with the mean."""
    defined = [v for v in values if v == v]
    if not defined:
        return {"mean": None, "sd": None, "min": None, "max": None, "n_defined": 0}
    mean = sum(defined) / len(defined)
    # One defined value: the sample-sd denominator (n-1) is EMPTY, so decree C1
    # applies -- null, never 0.0. Reporting 0.0 asserts zero dispersion from a
    # single observation, which is exactly the silent-zero class C1 exists to ban.
    # Corrected 2026-08-15 (audit wave 2); three committed frontier cells carried it.
    sd = (sum((v - mean) ** 2 for v in defined) / (len(defined) - 1)) ** 0.5 \
        if len(defined) > 1 else None
    return {"mean": mean, "sd": sd, "min": min(defined), "max": max(defined),
            "n_defined": len(defined)}


def mean_wer(records: Sequence[Mapping]) -> Frac:
    """VM-24 · mean word error rate over records carrying ``wer`` (num = the
    WER sum, so ``rate`` is the mean; the channel-descriptor sibling of the
    extraction metrics — B1's dose–response x-axis)."""
    return Frac(sum(r["wer"] for r in records), len(records))


def by_group(rows: Sequence[Mapping], key: str | Callable[[Mapping], str],
             metric: Callable, *args) -> dict[str, object]:
    """Apply any metric per group. ``key`` is a field name or a callable; group
    order is sorted, so output is deterministic."""
    key_fn = (lambda r: r[key]) if isinstance(key, str) else key
    groups: dict[str, list[Mapping]] = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    return {g: metric(groups[g], *args) for g in sorted(groups)}


# ---------------------------------------------------------------------------
# Registries and bundles (the COMBINERS pattern)
# ---------------------------------------------------------------------------

SESSION_METRICS: dict[str, Callable[[Sequence[Mapping]], object]] = {
    "release_rate": release_rate,
    "verdict_errors": verdict_errors,
    "accuracy_given_release": accuracy_given_release,
    "escalation_rate": escalation_rate,
    "set_accuracy": set_accuracy,
    "verdict_set_coverage": verdict_set_coverage,
    "indeterminate_output_size": indeterminate_output_size,
}

RUN_METRICS: dict[str, Callable[[Sequence[Mapping]], object]] = {
    "time_to_verdict": time_to_verdict,
    "burden_exchange": burden_exchange,
}

#: docs/metrics.md definitions table ↔ module bijection (tested).
METRIC_IDS: dict[str, str] = {
    "VM-01": "release_rate",
    "VM-02": "verdict_errors",
    "VM-03": "accuracy_given_release",
    "VM-04": "escalation_rate",
    "VM-05": "set_accuracy",
    "VM-06": "indeterminate_output_size",
    "VM-07": "credal_suite",
    "VM-08": "time_to_verdict",
    "VM-09": "burden_exchange",
    "VM-10": "extractor_accuracy",
    "VM-11": "phi",
    "VM-12a": "coverage_set",
    "VM-12b": "coverage_event",
    "VM-12c": "informative_coverage",
    "VM-13": "floor_alpha",
    "VM-14": "sigma_floor",
    "VM-15": "mondrian_floor",
    "VM-16": "grader_accuracy",
    "VM-17": "grader_coverage",
    "VM-18": "human_pairwise_agreement",
    "VM-19": "fleiss_kappa",
    "VM-20": "grader_verdict_agreement",
    "VM-21": "binding_decomposition",
    "VM-22": "verdict_set_coverage",
    "VM-23": "risk_among_released",
    "VM-24": "mean_wer",
    "VM-25": "miscovering_transcripts",
}


#: The session-row schema the suite consumes (docs/metrics.md §7). Producers:
#: ``verdict_cert.sessions.draw_sessions`` emits a superset; note that
#: ``m2_calibration._simulate`` rows LACK ``reachable``/``true_verdict`` and
#: cannot feed the full suite.
SESSION_ROW_FIELDS = ("certified", "true_verdict", "reachable")


def miscovering_transcripts(sessions: Sequence[Mapping]) -> list[str]:
    """VM-25 · the distinct transcript ids whose failed coverage produced this
    cell's wrong releases — decree C11's effective-information count, which must
    travel with any pooled denominator. A cell reporting 2,400 draws and 3
    miscovering transcripts is making a statement about 3 utterances.

    Rows need per-item ``true`` / ``set`` / ``transcript_id`` (the
    ``sessions.draw_sessions`` schema)."""
    ids: set[str] = set()
    for s in sessions:
        if not _released_wrong(s):
            continue
        for d in s["items"].values():
            if d["true"] not in d["set"]:
                ids.add(d["transcript_id"])
    return sorted(ids)


@dataclass(frozen=True)
class ArmDescriptor:
    """One sweep arm's provenance (docs/metrics.md §7). The released
    ``Certificate`` carries no α, threshold, or calibration provenance, so this
    descriptor is the ONLY attribution a sweep row has — the FULL descriptor,
    not the :attr:`seating_id` shorthand. Emitted beside every arm's metrics.

    ``tags`` is stored sorted because the combiners consume it as a SET —
    ``("judge","evaluator")`` and its reverse are the same arm and must not
    produce two descriptors."""

    pool: str
    group_key: str
    tags: tuple[str, ...]
    combiner: str
    builder: str
    baseline_tag: str
    alpha: float | None
    threshold: float | None
    spend: bool
    voice: str | None
    base_seed: int
    n_seeds: int
    sessions_per_seed: int
    generator: str = "verdict_cert.sessions.draw_sessions"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(sorted(self.tags)))

    @property
    def seating_id(self) -> str:
        """Order-independent name for the SEATING: pool·combiner·sorted-tags.

        Deliberately NOT a full-provenance key — it omits threshold, alpha,
        voice, spend and builder, so a sweep over thresholds produces one
        seating_id per threshold. Use the whole descriptor for provenance; this
        is for grouping arms that differ only in operating point."""
        return f"{self.pool}|{self.combiner}|{'+'.join(self.tags)}"

    def jsonable(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)} | {
            "seating_id": self.seating_id}


def summarize_sessions(sessions: Sequence[Mapping]) -> dict[str, object]:
    """The whole session-metric suite on one arm; JSON-ready via :func:`jsonable`.

    Rows must carry :data:`SESSION_ROW_FIELDS` plus one of ``released_correct``
    / ``released_wrong`` — validated up front so a partial-schema row (e.g. an
    ``m2_calibration._simulate`` row) fails with the missing fields named
    instead of a bare KeyError deep in one metric."""
    if sessions:
        missing = [f for f in SESSION_ROW_FIELDS if f not in sessions[0]]
        if "released_correct" not in sessions[0] and "released_wrong" not in sessions[0]:
            missing.append("released_correct|released_wrong")
        if missing:
            raise ValueError(
                f"session rows lack {missing}; the full suite needs the "
                "verdict_cert.sessions.draw_sessions schema (docs/metrics.md §7)")
    out: dict[str, object] = {name: fn(sessions) for name, fn in SESSION_METRICS.items()}
    out["credal"] = credal_suite(sessions)
    out["n"] = len(sessions)
    return out


def summarize_sessions_by_group(sessions: Sequence[Mapping],
                                key: str | Callable[[Mapping], str]) -> dict[str, dict]:
    """B1's one-call-per-arm entry point: the full suite, per group."""
    return by_group(sessions, key, summarize_sessions)
