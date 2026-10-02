"""Constrained-choice extraction against an OpenAI-compatible vLLM endpoint.

This is the load-bearing serving mechanism (build-guide §2): the model answers a
mapping question under ``guided_choice`` and returns logprobs; the renormalized
option distribution is the conformal score input downstream.

Operational facts this module encodes (measured on vLLM 0.25.1, 2026-07-16):

* ``guided_choice`` constrains the *output*, but the returned logprobs are the
  raw PRE-MASK next-token distribution — disallowed tokens come back with
  finite logprobs. The distribution must be filtered to the allowed options
  and renormalized here; it is not automatic.
* Option tokens carry leading-space variants (``' a'`` vs ``'a'``); probability
  mass is aggregated across variants, never keyed naively.
* Multi-token options (e.g. ``unclear`` under Gemma's tokenizer) may be absent
  from any top-k. The argmax choice is still trustworthy (it is constrained);
  the missing option simply contributes ~0 recovered mass. Exact per-option
  scoring is deferred to the M2 calibration harness.
* Qwen3 is a hybrid reasoning model: without ``enable_thinking: False`` its
  ``<think>`` tokens displace the decision token entirely.
* ``/v1/chat/completions`` is used throughout — the raw completions endpoint
  has a known ``-inf`` JSON-serialization failure under guided_choice on some
  models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from openai import OpenAI

from .items import EXTRACTION_OPTIONS, Item

MAPPING_PROMPT = """\
You are the independent evaluator in a job-readiness screening interview. \
Your task is to map the candidate's answer onto exactly one predefined option.

Question the candidate was asked:
{question}

Option definitions:
(a) {a}
(b) {b}
(c) {c}

Candidate's answer: "{answer}"

Which option definition best matches the candidate's answer? If the answer does \
not clearly match any option, choose "unclear". Reply with exactly one of: \
a, b, c, unclear."""

RENDER_PROMPT = """\
You are a friendly voice interviewer screening trade workers for overseas jobs. \
Ask the following assessment question in a natural, conversational way, in one \
or two short sentences of simple spoken English. Do not suggest any answer, do \
not list options, do not add new scenario details.

Question: {question}"""

#: The scored options — the canonical extraction competes ONLY these. ``unclear``
#: was removed after S1: as a competing token it is tokenizer-asymmetric (one
#: token for Mistral, invisible multi-token for Gemma) and produced spurious
#: abstentions. Abstention is a property of the distribution's shape, handled by
#: conformal thresholding in M2 — never by a competing token.
CANONICAL_OPTIONS: tuple[str, ...] = ("a", "b", "c")

CANONICAL_PROMPT = """\
You are the independent evaluator in a job-readiness screening interview. \
Your task is to map the candidate's answer onto exactly one predefined option.

Question the candidate was asked:
{question}

Option definitions:
(a) {a}
(b) {b}
(c) {c}

Candidate's answer: "{answer}"

Which option definition best matches the candidate's answer? \
Reply with exactly one of: a, b, c."""

#: The v2 pool's binary template (prereg §6.4, committed before any extraction
#: runs). Exactly two slots and an instruction naming exactly the live options —
#: the three-slot template renders a dangling empty "(c)" and licenses a phantom
#: reply under a binary guided_choice, leaking pre-mask mass into the one
#: diagnostic the v2 pool exists to measure (wave-3 finding 3, both reviewers).
#: `canonical_per_option` refuses non-{a,b,c} option sets without an explicit
#: template; this is that template. The framing mirrors CANONICAL_PROMPT so the
#: only changed variables against v1 are the option arity and the grading task.
CANONICAL_PROMPT_BINARY = """\
You are the independent evaluator grading one answer in a US naturalization \
civics check. Your task is to decide whether the candidate's answer is correct \
for the question, judged against the official acceptable answers.

Question the candidate was asked:
{question}

Official acceptable answers:
{a}

Candidate's answer: "{answer}"

Is the candidate's answer correct? Reply with exactly one of: a, b.
(a) correct — the answer matches an acceptable answer in meaning
(b) incorrect — it does not"""


def recover_mass(
    top_logprobs: list, options: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    """Recover per-option probability mass from a decision-token top-k.

    vLLM's logprobs under guided_choice are the PRE-mask model distribution, and
    option tokens appear in leading-space variants (``' a'`` vs ``'a'``) — so
    mass is aggregated per option across variants and then filtered to the
    allowed options. Returns (option_mass, raw_token_mass); option_mass is
    UN-normalized.

    ⚠ **Truncation semantics — this is the whole of the score's provenance.** The
    caller passes a top-k list (k=20 at every call site), so an option that does
    not appear there is assigned mass 0.0 — *asserted, not measured*. When only
    one option survives, the downstream renormalization yields exactly 1.0 by
    arithmetic rather than by confidence. Measured on the committed pools that is
    52.3%/51.4% of judge rows and 80.1%/79.5% of evaluator rows, and it is what
    drives ~90% of true-option nonconformity scores to exactly zero
    (`research/experiment-audit.md`). The recorded object is therefore *the
    renormalized mass over options surviving the top-k cut*, not the model's
    distribution over the options.
    """
    mass: dict[str, float] = {}
    raw: dict[str, float] = {}
    for alt in top_logprobs:
        p = math.exp(alt.logprob) if alt.logprob != float("-inf") else 0.0
        raw[alt.token] = raw.get(alt.token, 0.0) + p
        key = alt.token.strip().lower()
        if key in options:
            mass[key] = mass.get(key, 0.0) + p
    return mass, raw


@dataclass(frozen=True)
class Extraction:
    """One constrained mapping: the argmax choice plus the recovered
    (renormalized, post-filter) option distribution."""

    choice: str
    dist: dict[str, float]
    raw_top: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalExtraction:
    """THE extraction object (S2) — used identically by every family, and both
    the decision and the conformal score input downstream.

    ``choice`` is the argmax of ``dist`` (the renormalized {a,b,c} decision-token
    distribution) — NOT the constrained-decode emitted token. ``emitted`` is kept
    as a diagnostic only; rows where emitted != choice track the S1 Mistral
    anomaly (constrained output diverging from the model's own distribution).
    """

    choice: str
    dist: dict[str, float]
    emitted: str
    #: The raw decision-token top-k, kept so the "no option in top-k" fallback
    #: below is diagnosable after the fact. Defaulted and additive: the committed
    #: harnesses neither set nor serialize it, so their outputs are unchanged.
    raw_top: dict[str, float] = field(default_factory=dict)

    @property
    def diverged(self) -> bool:
        return self.emitted != self.choice


class Extractor:
    """Thin client for one served model. The same prompts are used for every
    family — S3's independence measurement requires an identical extraction
    path with only the model swapped."""

    def __init__(self, base_url: str, *, disable_thinking: bool | None = None):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = self.client.models.list().data[0].id
        # Qwen3's <think> tokens break guided_choice; auto-detect unless forced.
        if disable_thinking is None:
            disable_thinking = "qwen3" in self.model.lower()
        self.disable_thinking = disable_thinking

    def _extra_body(self) -> dict:
        extra: dict = {"guided_choice": list(EXTRACTION_OPTIONS)}
        if self.disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        return extra

    def map_answer(self, item: Item, answer: str) -> Extraction:
        prompt = MAPPING_PROMPT.format(
            question=item.question,
            a=item.options["a"].meaning,
            b=item.options["b"].meaning,
            c=item.options["c"].meaning,
            answer=answer,
        )
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            extra_body=self._extra_body(),
        )
        choice = r.choices[0].message.content.strip()
        mass, raw = recover_mass(r.choices[0].logprobs.content[0].top_logprobs, EXTRACTION_OPTIONS)
        z = sum(mass.values())
        dist = {k: v / z for k, v in mass.items()} if z > 0 else {}
        return Extraction(choice=choice, dist=dist, raw_top=raw)

    def canonical_measured(self, item: Item, answer: str,
                           notes: str | None = None,
                           options: tuple[str, ...] = CANONICAL_OPTIONS,
                           meanings: dict[str, str] | None = None,
                           prompt_template: str | None = None,
                           top_k: int = 20) -> "PerOptionExtraction":
        """The §7.2 transport: ONE call, per-option mass aggregated over top-k
        variants, every option measured or the call raises.

        Why this shape (prereg §7.2, §8.5). The registered transport was one
        ``guided_choice=[single option]`` call per option; on this box's vLLM a
        single-option grammar is INERT — forcing "b" emits "a" — so every pass
        returned the same token's logprob. Multi-option guided decoding works
        normally, and its top-k logprobs are PRE-mask, so one call carries the
        model's own distribution over the decision token.

        The aggregation is over variants: a model spells the same choice ``a``,
        ``A``, `` (a)``, ``a.`` and each spelling is a distinct token carrying
        part of that option's mass. Summing them is what makes P(option) a
        measurement rather than a sample of one spelling.

        Two guards, both hard:

        - **Absence raises.** If an option has no variant anywhere in the top-k,
          this raises instead of recording 0.0. F-11 was exactly the habit of
          treating "absent from the window" as "probability zero"; here the fix
          survives as a guard rather than an assumption.
        - **The residual is kept.** ``mass_unnormalized`` is the option variants'
          share of the whole next-token distribution BEFORE renormalizing — how
          much of the model's mass the options actually carry.
        """
        opts = tuple(options)
        meanings = meanings if meanings is not None else {
            k: item.options[k].meaning for k in opts}
        template = prompt_template or CANONICAL_PROMPT
        prompt = _render_prompt(template, item, meanings, answer)
        if notes:
            marker = "Candidate's answer:"
            prompt = prompt.replace(marker, notes.rstrip() + "\n\n" + marker, 1)
        extra: dict = {"guided_choice": list(opts)}
        if self.disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top_k,
            extra_body=extra,
        )
        try:
            top = r.choices[0].logprobs.content[0].top_logprobs
        except (AttributeError, IndexError, TypeError) as e:
            raise PerOptionScoringError(
                "response carries no top_logprobs — the endpoint must be called "
                f"with logprobs=True and top_logprobs={top_k}, and the server's "
                f"--max-logprobs must be at least that. Underlying: {e}") from e
        return variant_aggregated_scores(top, opts,
                                         emitted=r.choices[0].message.content)

    def canonical(self, item: Item, answer: str,
                  notes: str | None = None) -> CanonicalExtraction:
        """The canonical extraction (S2): one guided call over {a,b,c} only;
        decision = argmax of the renormalized decision-token distribution.

        ``notes`` is the M2.5 anchored-loop hook (proposal §3.1: prompt-level
        updates, item-specific confusion notes): extra reference material
        inserted before the answer. Callers derive it per agent from GROUND
        TRUTH on held-out data, never from the other agent — passing anything
        else is the consensus trap. Default None reproduces the S2/S3/M1/M3
        prompt byte-for-byte.
        """
        prompt = CANONICAL_PROMPT.format(
            question=item.question,
            a=item.options["a"].meaning,
            b=item.options["b"].meaning,
            c=item.options["c"].meaning,
            answer=answer,
        )
        if notes:
            marker = "Candidate's answer:"
            prompt = prompt.replace(marker, notes.rstrip() + "\n\n" + marker, 1)
        extra: dict = {"guided_choice": list(CANONICAL_OPTIONS)}
        if self.disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            extra_body=extra,
        )
        emitted = r.choices[0].message.content.strip()
        mass, raw = recover_mass(r.choices[0].logprobs.content[0].top_logprobs,
                                 CANONICAL_OPTIONS)
        z = sum(mass.values())
        if z > 0:
            dist = {k: mass.get(k, 0.0) / z for k in CANONICAL_OPTIONS}
        else:
            # All three options outside the top-20 (never observed in S1; the
            # decision token is constrained to one of them) — trust the emitted
            # token and record a degenerate distribution.
            dist = {k: (1.0 if k == emitted else 0.0) for k in CANONICAL_OPTIONS}
        choice = max(dist, key=dist.get)
        return CanonicalExtraction(choice=choice, dist=dist, emitted=emitted, raw_top=raw)

    def render_question(self, item: Item) -> str:
        """Judge-side: administer the item conversationally (S1a)."""
        extra: dict = {}
        if self.disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": RENDER_PROMPT.format(question=item.question)}],
            max_tokens=90,
            temperature=0.0,
            extra_body=extra or None,
        )
        return r.choices[0].message.content.strip()


    def canonical_per_option(self, item: Item, answer: str,
                             notes: str | None = None,
                             options: tuple[str, ...] = CANONICAL_OPTIONS,
                             meanings: dict[str, str] | None = None,
                             prompt_template: str | None = None,
                             ) -> "PerOptionExtraction":
        """v2 extraction: one forced call per option; every probability measured.

        The prompt is byte-identical to :meth:`canonical`'s (including the
        ``notes`` hook and its consensus-trap warning) — only the scoring
        transport differs, so a v1-vs-v2 score comparison isolates the F-11
        fix. ``options``/``meanings`` generalize to the v2 pool's binary
        {correct, incorrect} single-token mapping.
        """
        # Wave-3 finding 3 (both reviewers, independently): CANONICAL_PROMPT
        # hardcodes three slots and the instruction "one of: a, b, c" — with a
        # non-{a,b,c} option set it renders a dangling empty slot and licenses a
        # phantom reply, leaking pre-mask mass and corrupting mass_unnormalized.
        # A non-default option set therefore REQUIRES its own template.
        if tuple(options) != CANONICAL_OPTIONS and prompt_template is None:
            raise ValueError(
                f"options {options!r} need an explicit prompt_template — "
                "CANONICAL_PROMPT names exactly a/b/c and would render a "
                "malformed prompt (wave-3 finding 3)")
        mean = meanings or {o: item.options[o].meaning for o in options}
        prompt = (prompt_template or CANONICAL_PROMPT).format(
            question=item.question,
            a=mean.get("a", ""), b=mean.get("b", ""), c=mean.get("c", ""),
            answer=answer,
        )
        if notes:
            marker = "Candidate's answer:"
            prompt = prompt.replace(marker, notes.rstrip() + "\n\n" + marker, 1)

        def call(option: str):
            extra: dict = {"guided_choice": [option]}
            if self.disable_thinking:
                extra["chat_template_kwargs"] = {"enable_thinking": False}
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2,
                temperature=0.0,
                logprobs=True,
                extra_body=extra,
            )

        return per_option_scores(call, options)


# ---------------------------------------------------------------------------
# Per-option extraction — the v2 score path (F-11 FIXED here, not scoped)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerOptionExtraction:
    """The v2 extraction object: every option's probability MEASURED, none asserted.

    ``dist`` is renormalized over the per-option masses; ``raw_logprob`` holds
    each option's measured pre-mask log-probability (the diagnostic F-11 found
    was thrown away); ``mass_unnormalized`` is Σ exp(raw) BEFORE renormalizing —
    how much of the model's next-token mass the option tokens actually carry,
    the honest residual F-11's truncated path could not even express.

    ``window`` is the raw ``(token, logprob)`` list the readout inspected, in
    server order. Discarding it was itself a measurement-integrity cost: it made
    the option's RANK unrecoverable, so the paper could not say whether options
    actually left the top-k window or were merely rounded away at write time —
    the two routes are indistinguishable once the window is gone. Optional and
    defaulted so existing callers and caches stay valid (audit wave 8).
    """

    choice: str
    dist: dict[str, float]
    raw_logprob: dict[str, float]
    mass_unnormalized: float
    window: tuple[tuple[str, float], ...] = ()


class PerOptionScoringError(RuntimeError):
    """Raised when the served model's logprob semantics are not the pre-mask
    distribution this path requires — never silently worked around (that
    silent fallback was exactly F-11's second fabrication path)."""


def _norm_token(tok: str) -> str:
    """Option-letter normalization: 'A)', ' (b) ', 'c.' -> 'a'/'b'/'c'."""
    return tok.strip().strip("(). \t\n").lower()


def _forced_token_emission(resp) -> tuple[str, float]:
    """The emitted decision token AND its own logprob.

    The token is returned alongside the number because the number alone cannot
    tell you WHICH option it belongs to: if the server ignores the constraint,
    every forced pass returns the same emitted token's logprob and the caller
    would silently score one option's mass as another's (wave-5 F2).

    A degenerate response shape (logprobs disabled or dropped by the server)
    raises :class:`PerOptionScoringError` with the configuration hint, not a
    bare AttributeError/IndexError (wave-3 finding 6)."""
    try:
        content = resp.choices[0].logprobs.content[0]
        lp = float(content.logprob)
        tok = getattr(content, "token", None)
    except (AttributeError, IndexError, TypeError) as e:
        raise PerOptionScoringError(
            "response carries no decision-token logprob — the endpoint must be "
            "called with logprobs=True and must return them (some proxies strip "
            f"the field). Underlying: {type(e).__name__}: {e}") from e
    if tok is None:
        raise PerOptionScoringError(
            "response carries a logprob but no emitted token — without the token "
            "this path cannot verify the constraint was honored, and an "
            "unverified forced pass is exactly the wave-5 F2 failure")
    return tok, lp


def _render_prompt(template: str, item, meanings: dict[str, str], answer: str) -> str:
    """Fill a canonical template from an item + per-option meanings.

    Wave-3 finding 3: CANONICAL_PROMPT hardcodes three slots, so a two-option
    domain must use CANONICAL_PROMPT_BINARY or it renders a dangling empty slot
    and licenses a phantom reply. The template decides the arity; this only
    supplies what the template asks for.
    """
    fields = {"question": item.question, "answer": answer}
    for k, v in meanings.items():
        fields[k] = v
    return template.format(**fields)


def variant_aggregated_scores(top_logprobs, options: tuple[str, ...],
                              emitted: str | None = None) -> PerOptionExtraction:
    """§7.2's arithmetic, separated from transport so it is unit-testable.

    ``top_logprobs`` is one decision token's top-k list (objects carrying
    ``.token``/``.logprob``, or ``(token, logprob)`` pairs). Mass for an option
    is the SUM over every token that normalizes to it — ``"A"``, ``" (a)"`` and
    ``"a."`` are the same choice spelled three ways, and reading only one
    spelling understates the option.
    """
    mass: dict[str, float] = {}
    seen: dict[str, list[str]] = {}
    for entry in top_logprobs:
        tok = getattr(entry, "token", None)
        lp = getattr(entry, "logprob", None)
        if tok is None and isinstance(entry, (tuple, list)):
            tok, lp = entry[0], entry[1]
        if tok is None or lp is None:
            continue
        key = _norm_token(tok)
        if key in options:
            mass[key] = mass.get(key, 0.0) + math.exp(float(lp))
            seen.setdefault(key, []).append(tok)
    missing = [o for o in options if o not in mass]
    if missing:
        raise PerOptionScoringError(
            f"options {missing} have no variant in the top-{len(list(top_logprobs))} "
            "window — refusing to record absence as probability 0.0, which is "
            "precisely the F-11 fabrication this transport exists to remove "
            "(prereg §7.2 guard i). Widen the window (--max-logprobs) or treat "
            "the record as unscoreable; do NOT default it to zero.")
    underflowed = [o for o in options if mass[o] <= 0.0]
    if underflowed:
        # Present in the window but exp(logprob) underflowed to 0.0 — vanishingly
        # rare at top-100, and it would hand back a probability of exactly zero
        # for an option the model DID rank, i.e. the F-11 artifact by float limit
        # rather than by truncation. Refuse it for the same reason.
        raise PerOptionScoringError(
            f"options {underflowed} underflowed to zero mass despite being present "
            "in the top-k window — the score would be exactly 0.0 by float limit, "
            "not by measurement; record the row as unscoreable instead")
    z = sum(mass.values())
    if z <= 0.0:
        raise PerOptionScoringError(
            "option variants carry zero mass — no evidence to renormalize")
    dist = {o: mass[o] / z for o in options}
    raw = {o: math.log(mass[o]) for o in options}
    choice = (_norm_token(emitted) if emitted and _norm_token(emitted) in options
              else max(dist, key=dist.get))
    # Both entry shapes, exactly as the aggregation loop above accepts them —
    # the window must not be the one place that assumes objects.
    def _pair(entry):
        tok = getattr(entry, "token", None)
        lp = getattr(entry, "logprob", None)
        if tok is None and isinstance(entry, (tuple, list)):
            tok, lp = entry[0], entry[1]
        return (str(tok), float(lp))

    return PerOptionExtraction(
        choice=choice, dist=dist, raw_logprob=raw, mass_unnormalized=z,
        window=tuple(_pair(e) for e in top_logprobs))


def per_option_scores(call, options: tuple[str, ...]) -> PerOptionExtraction:
    """Assemble a :class:`PerOptionExtraction` from one forced call per option.

    ``call(option) -> response`` performs a ``guided_choice=[option]`` chat
    call with ``logprobs=True`` (the caller owns prompts and transport; this
    function owns the arithmetic so it is unit-testable without a server).

    Why this is exact where ``canonical()`` was not (F-11): the single-call
    path could only see options that survived the shared top-20 — an absent
    option was ASSERTED to have mass 0.0, and on 52–80% of committed rows only
    one option survived, making its "probability" 1.0 by arithmetic. Here each
    option's probability is read from its own forced pass via the emitted
    token's logprob, which vLLM reports unconditionally. Nothing is truncated;
    the ``z == 0`` one-hot fabrication path cannot exist because every option
    always yields a measured number.

    Two runtime guards, both learned from live stacks:

    **Constraint honored (wave-5 F2).** The emitted token must BE the forced
    option. On this box's vLLM, ``guided_choice=[single option]`` is inert —
    forcing "b" emits "a" — so every pass returns the same token's logprob and
    the assembled ``dist`` is a fabrication that looks entirely plausible
    (a near-uniform split with a believable residual) whenever that token's
    probability is below ~0.5. Silent in exactly the borderline band that
    decides calibration, so it raises here.

    Semantics guard: under guided decoding the POST-mask probability of
    a single forced option is ~1.0 by construction. If the server reported
    post-mask logprobs, every option would come back at ~0.0 logprob and the
    unnormalized masses would sum to ~len(options) — impossible for a real
    pre-mask distribution (Σ ≤ 1). That condition raises
    :class:`PerOptionScoringError` rather than producing garbage scores.
    """
    raw: dict[str, float] = {}
    for opt in options:
        tok, lp = _forced_token_emission(call(opt))
        if _norm_token(tok) != _norm_token(opt):
            raise PerOptionScoringError(
                f"forced option {opt!r} but the server emitted {tok!r} — the "
                "decoding constraint was NOT honored, so this pass measures the "
                "wrong option's probability. Every option would be scored from "
                "the same emitted token, fabricating a plausible distribution "
                "(wave-5 F2). Do not 'work around' by trusting the number: use "
                "the measured single-call transport (prereg §7.2) instead.")
        if math.isnan(lp):
            raise PerOptionScoringError(
                f"option {opt!r}: NaN logprob — refusing to score (a NaN passes "
                "every comparison guard silently; wave-3 finding 5)")
        if lp > 0.0:
            raise PerOptionScoringError(
                f"option {opt!r}: positive logprob {lp} — a probability above 1 "
                "is itself proof of broken logprob semantics (wave-3 finding 5)")
        raw[opt] = lp
    mass = {o: math.exp(lp) if lp != float("-inf") else 0.0 for o, lp in raw.items()}
    z = sum(mass.values())
    if z > 1.0 + 1e-3 and all(m > 1.0 / (len(options) + 1) for m in mass.values()):
        raise PerOptionScoringError(
            f"unnormalized option masses sum to {z:.3f} with every option above "
            f"{1.0 / (len(options) + 1):.2f} — the server is reporting POST-mask "
            "logprobs (forced token ≈ certainty), not the pre-mask distribution "
            "this score requires. Check the vLLM version/config before rerunning.")
    if z <= 0.0:
        raise PerOptionScoringError(
            "every option carries zero pre-mask mass — refusing to fabricate a "
            "distribution from no evidence (the F-11 second-path shape).")
    dist = {o: m / z for o, m in mass.items()}
    choice = max(dist, key=dist.get)
    return PerOptionExtraction(choice=choice, dist=dist, raw_logprob=raw,
                               mass_unnormalized=z)

