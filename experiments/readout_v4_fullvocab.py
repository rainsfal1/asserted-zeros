"""V2/V3/V4: check the corrected readout against a full-vocabulary reconstruction.

Design and predictions: ``docs/readout-validation-preregistration.md`` (frozen
2026-08-18, amendment 1 same day). The standing objection is that
``canonical_measured`` is checked only against itself: it reads a top-100 window
under ``guided_choice``, aggregates variants with ``_norm_token``, renormalizes
over options, and nothing outside that code path has confirmed the result.

Three subcommands, in this order (prereg §5; serving is sequential):

  served   Needs the model SERVED. Draws the stratified sample, records the
           server's own tokenization of each prompt (the positive control), and
           re-extracts ~20 committed records to prove the server has not drifted
           since the cache was written (amendment 1).

  offline  Needs the GPU FREE. Loads the model outside any server, no
           guided_choice, no top-k window, and takes the softmax over the FULL
           vocabulary at the decision token. Asserts prompt-token identity
           against the served stage before touching a distribution.

  compare  No GPU. Scores V2/V3/V4 against the registered thresholds.

The prompt is built by the SAME code the extraction path uses (``_render_prompt``
with the same template). That is deliberate: the object under test is the
readout, not the prompt, so the prompt is held fixed and proven identical.
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from verdict_cert.extraction import (  # noqa: E402
    CANONICAL_PROMPT_BINARY, _norm_token, _render_prompt,
)

_s = ilu.spec_from_file_location("v2p", REPO / "experiments" / "v2_pool.py")
v2p = ilu.module_from_spec(_s)
_s.loader.exec_module(v2p)

MODELS = {"judge": "Qwen/Qwen3-32B-AWQ",
          "evaluator": "gaunernst/gemma-3-27b-it-qat-autoawq"}
OPTIONS = ("a", "b")
OUT = REPO / "results" / "readout_validation"
N_DECILE = 300          # prereg §2.1: the sample V2/V3/V4 are scored on
N_LOWMASS = 50          # amendment 1: the adversarial stratum, reported apart
N_DRIFT = 20            # amendment 1: served-vs-cache drift control
WINDOW = 2048           # amendment 3: wide enough that every option token is inside


def _fnv(s: str) -> int:
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def load_pool(tag: str):
    """(records by id, committed extraction by id) for the binary v2 corpus."""
    recs = {json.loads(x)["id"]: json.loads(x)
            for x in (REPO / "results/v2_pool/records.jsonl").open() if x.strip()}
    ex = {json.loads(x)["id"]: json.loads(x)
          for x in (REPO / f"results/v2_pool/extract-{tag}.jsonl").open() if x.strip()}
    return recs, ex


def build_prompt(rec: dict, key: dict) -> str:
    """Byte-identical to what v2_pool.cmd_extract sent for this record."""
    from types import SimpleNamespace
    return _render_prompt(
        CANONICAL_PROMPT_BINARY,
        SimpleNamespace(question=v2p.QUESTIONS[rec["item"]]),
        {"a": "; ".join(key.get(rec["item"], [])), "b": ""},
        rec["transcript"])


def sample(ex: dict) -> tuple[list[str], list[str]]:
    """(decile sample, low-mass stratum) — deterministic, seed-free."""
    ids = sorted(ex)
    scored = [(i, min(ex[i]["dist"].values())) for i in ids]
    live = [(i, m) for i, m in scored if m > 0]
    live.sort(key=lambda t: t[1])
    # decile bands of log10(min_p), equal draw per band, bottom band oversampled
    bands: list[list[str]] = [[] for _ in range(10)]
    for rank, (i, _m) in enumerate(live):
        bands[min(9, rank * 10 // len(live))].append(i)
    per = N_DECILE // 11          # 11 shares: bottom band gets two
    decile: list[str] = []
    for b, members in enumerate(bands):
        want = per * 2 if b == 0 else per
        members = sorted(members, key=lambda i: _fnv(f"{i}:v4"))
        decile.extend(members[:want])
    lowmass = [i for i, _ in sorted(
        ((i, ex[i]["mass_unnormalized"]) for i in ids), key=lambda t: t[1])][:N_LOWMASS]
    return decile, lowmass


def cmd_served(args) -> int:
    """Sample, capture the server's tokenization, prove the server has not drifted."""
    import urllib.request
    from types import SimpleNamespace

    from verdict_cert.extraction import Extractor

    v2p._load_questions()
    key = v2p.m6.load_key()
    recs, ex = load_pool(args.tag)
    decile, lowmass = sample(ex)
    model = MODELS[args.tag]
    print(f"[{args.tag}] sample: {len(decile)} decile + {len(lowmass)} low-mass")

    # The control must tokenize the way the EXTRACTION does, or it fails for a
    # reason that has nothing to do with the readout. Extractor auto-sets
    # enable_thinking=False for Qwen3 (extraction.py:188); gemma has no such
    # switch, which is why the evaluator run did not need this.
    tmpl_kwargs = ({"enable_thinking": False}
                   if "qwen3" in model.lower() else None)

    def served_tokens(prompt: str) -> list[int]:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if tmpl_kwargs:
            body["chat_template_kwargs"] = tmpl_kwargs
        req = urllib.request.Request(
            f"{args.endpoint.rsplit('/v1', 1)[0]}/tokenize",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]

    rows = []
    for n, rid in enumerate(decile + lowmass, 1):
        p = build_prompt(recs[rid], key)
        rows.append({"id": rid, "stratum": "decile" if rid in set(decile) else "lowmass",
                     "prompt": p, "served_tokens": served_tokens(p),
                     "gold": recs[rid]["gold"]})
        if n % 100 == 0:
            print(f"  tokenized {n}/{len(decile) + len(lowmass)}")

    # Drift control (amendment 1): the cache must still be reproducible NOW.
    exr = Extractor(args.endpoint)
    drift, worst = [], 0.0
    for rid in sorted(ex, key=lambda i: _fnv(f"{i}:drift"))[:N_DRIFT]:
        po = exr.canonical_measured(
            SimpleNamespace(question=v2p.QUESTIONS[recs[rid]["item"]]),
            recs[rid]["transcript"], options=OPTIONS,
            meanings={"a": "; ".join(key.get(recs[rid]["item"], [])), "b": ""},
            prompt_template=CANONICAL_PROMPT_BINARY, top_k=100)
        old = ex[rid]
        d = max(abs(po.dist[o] - old["dist"][o]) for o in OPTIONS)
        worst = max(worst, d)
        drift.append({"id": rid, "max_abs_dist_delta": d,
                      "mass_delta": abs(po.mass_unnormalized - old["mass_unnormalized"])})
    print(f"  drift control: worst |Δdist| over {N_DRIFT} records = {worst:.3e}"
          f"  ({'BIT-IDENTICAL' if worst == 0 else 'DRIFTED'})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"f4_served_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "model": model, "n_decile": len(decile),
         "n_lowmass": len(lowmass), "drift_control": drift,
         "drift_worst_abs": worst, "rows": rows}, indent=1))
    print(f"  wrote f4_served_{args.tag}.json")
    return 0


def cmd_baseline(args) -> int:
    """Re-extract the sampled records through the LIVE server (prereg amendment 2).

    F4's comparator is no longer the committed cache. A freshly launched server
    does not reproduce the cache exactly (154/200 bit-identical, worst 1.27e-2,
    stable within an instance and different across them), so comparing offline
    values against the cache would confound the readout difference under test
    with an instance-level noise floor. This captures the same-instance baseline
    while that instance is still running.
    """
    from types import SimpleNamespace

    from verdict_cert.extraction import Extractor

    key = v2p.m6.load_key()
    recs, _ = load_pool(args.tag)
    served = json.loads((OUT / f"f4_served_{args.tag}.json").read_text())
    exr = Extractor(args.endpoint)
    rows = []
    for n, r in enumerate(served["rows"], 1):
        rec = recs[r["id"]]
        po = exr.canonical_measured(
            SimpleNamespace(question=v2p.QUESTIONS[rec["item"]]), rec["transcript"],
            options=OPTIONS,
            meanings={"a": "; ".join(key.get(rec["item"], [])), "b": ""},
            prompt_template=CANONICAL_PROMPT_BINARY, top_k=100)
        rows.append({"id": r["id"], "dist": dict(po.dist),
                     "mass_unnormalized": po.mass_unnormalized})
        if n % 50 == 0:
            print(f"  baseline {n}/{len(served['rows'])}")
    (OUT / f"f4_baseline_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "note": "same-instance served baseline, prereg amendment 2",
         "rows": rows}, indent=1))
    print(f"  wrote f4_baseline_{args.tag}.json ({len(rows)} rows)")
    return 0


def option_token_ids(tk, options=OPTIONS) -> dict[str, list[int]]:
    """Every vocabulary id whose decoded form normalizes to an option letter."""
    out: dict[str, list[int]] = {o: [] for o in options}
    for tid in range(len(tk)):
        s = tk.decode([tid])
        if s and len(s) <= 8 and _norm_token(s) in out:
            out[_norm_token(s)].append(tid)
    return out


def cmd_offline(args) -> int:
    """Full-vocabulary option mass at the decision token, outside any server.

    Transport: vLLM offline (prereg amendment 3). transformers cannot load these
    AWQ checkpoints without gptqmodel, whose own import chain is broken here. So
    this is independent of the HTTP server, the OpenAI layer, guided_choice, the
    top-100 window and the aggregation code -- but NOT of vLLM's model execution.
    It validates the readout, not the engine, and the paper must say so.

    "Full vocabulary" is exact rather than approximate: vLLM reports ABSOLUTE
    log-probabilities from the full softmax, so summing the option-normalizing
    tokens inside a wide window gives the true full-vocabulary option mass --
    provided every such token is inside the window. That is checked per record;
    a record failing it is EXCLUDED and counted, never summed anyway, since a
    silently truncated sum is precisely this paper's own defect.
    """
    import math as _m

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    served = json.loads((OUT / f"f4_served_{args.tag}.json").read_text())
    model_id = served["model"]
    tk = AutoTokenizer.from_pretrained(model_id)
    opt_ids = option_token_ids(tk)
    v1 = json.loads((OUT / "v1_variant_set.json").read_text())[args.tag]["candidate_misses"]
    cand_strs = {s for v in v1.values() for s in v}
    cand_ids = sorted({t for t in range(len(tk)) if tk.decode([t]) in cand_strs})
    print("  option token ids: " + ", ".join(f"{o}={len(v)}" for o, v in opt_ids.items())
          + f" | candidate-miss ids={len(cand_ids)}")

    need = {t for v in opt_ids.values() for t in v}
    llm = LLM(model=model_id, max_logprobs=WINDOW, gpu_memory_utilization=0.90,
              max_model_len=4096, enforce_eager=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=WINDOW)

    prompts = []
    for r in served["rows"]:
        text = tk.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True, tokenize=False,
            **({"enable_thinking": False} if "Qwen" in model_id else {}))
        ids = tk(text, add_special_tokens=False)["input_ids"]
        if ids != r["served_tokens"]:
            raise SystemExit(
                f"POSITIVE CONTROL FAILED on {r['id']}: offline prompt is {len(ids)} "
                f"tokens, the server said {len(r['served_tokens'])}. Comparing "
                "distributions over different inputs is uninformative; prereg §4 says "
                "no F4 number is reportable in this case.")
        prompts.append(ids)
    print(f"  positive control: all {len(prompts)} prompts tokenize identically to the server")

    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    rows, excluded = [], 0
    for r, o in zip(served["rows"], outs):
        lp = o.outputs[0].logprobs[0]           # {token_id: Logprob}, ABSOLUTE
        prob = {t: _m.exp(v.logprob) for t, v in lp.items()}
        floor = min(prob.values())              # nothing outside the window exceeds this
        present = need & set(prob)
        # Do NOT drop absent option tokens to zero -- that is this paper's own
        # defect. Sum what is measured and BOUND what is outside the window:
        # every missing id has probability at most `floor`, so the true option
        # mass lies in [m_full, m_full + n_missing*floor]. The bound is reported
        # per record and must be negligible for the comparison to be exact.
        p_full = {op: sum(prob[t] for t in v if t in present)
                  for op, v in opt_ids.items()}
        m_full = sum(p_full.values())
        n_missing = len(need - present)
        bound = n_missing * floor
        order = sorted(prob, key=lambda t: -prob[t])
        keep = set(order[:100])
        m_window = sum(prob[t] for v in opt_ids.values() for t in v if t in keep)
        rows.append({
            "id": r["id"], "stratum": r["stratum"], "gold": r["gold"], "excluded": False,
            "p_full": p_full, "m_full": m_full, "m_window": m_window,
            "missing_option_tokens": n_missing, "window_floor": floor,
            "missing_mass_bound": bound, "bound_relative": bound / m_full if m_full else None,
            "cand_miss_mass": sum(prob.get(t, 0.0) for t in cand_ids),
            # the constraint was silently dropped, so the "decision token" need
            # not be an option at all -- record what actually leads, and each
            # option's rank, the quantity the paper says it cannot report
            "top1_str": tk.decode([order[0]]), "top1_p": prob[order[0]],
            "top1_is_option": _norm_token(tk.decode([order[0]])) in opt_ids,
            "option_ranks": {op: (min(order.index(t) for t in v if t in present) + 1
                                  if any(t in present for t in v) else None)
                             for op, v in opt_ids.items()},
        })
    worst = max(r["bound_relative"] for r in rows)
    print(f"  {len(rows)} records; worst missing-mass bound {worst:.2e} relative "
          f"({'negligible' if worst < 1e-9 else 'NOT negligible -- report it'})")

    (OUT / f"f4_offline_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "model": model_id, "transport": "vllm-offline",
         "window": WINDOW, "vocab": len(tk),
         "n_option_tokens": {o: len(v) for o, v in opt_ids.items()},
         "n_candidate_miss_tokens": len(cand_ids), "n_excluded": excluded,
         "rows": rows}, indent=1))
    print(f"  wrote f4_offline_{args.tag}.json")
    return 0


def cmd_compare(args) -> int:
    """Score V2/V3/V4 against the registered thresholds."""
    off = json.loads((OUT / f"f4_offline_{args.tag}.json").read_text())
    # prereg amendment 2: compare against the same-instance served baseline,
    # not the committed cache, whose instance is gone.
    ex = {r["id"]: r for r in json.loads(
        (OUT / f"f4_baseline_{args.tag}.json").read_text())["rows"]}
    res: dict = {"tag": args.tag, "transport": off["transport"], "strata": {}}

    for stratum in ("decile", "lowmass"):
        rows = [r for r in off["rows"]
                if r["stratum"] == stratum and not r.get("excluded")]
        if not rows:
            continue
        ratios, score_ratios, zeros, cand = [], [], 0, []
        for r in rows:
            c = ex[r["id"]]
            ratios.append(r["m_full"] / c["mass_unnormalized"])
            dist_full = {o: p / r["m_full"] for o, p in r["p_full"].items()}
            gold = "a" if r["gold"] in (1, "1", True) else "b"
            s_full = 1.0 - dist_full[gold]
            s_cache = 1.0 - c["dist"][gold]
            if s_full == 0.0:
                zeros += 1
            if s_cache > 0 and s_full > 0:
                score_ratios.append(s_full / s_cache)
            cand.append(r["cand_miss_mass"])
        within2 = sum(1 for x in score_ratios if 0.5 <= x <= 2.0) / len(score_ratios)
        res["strata"][stratum] = {
            "n": len(rows),
            "V2_mass_ratio": {"median": st.median(ratios), "min": min(ratios),
                              "max": max(ratios),
                              "p99_abs_dev": sorted(abs(1 - x) for x in ratios)[
                                  max(0, math.ceil(0.99 * len(ratios)) - 1)]},
            "V3_score_ratio": {"within_factor_2": within2,
                               "median": st.median(score_ratios),
                               "p05": sorted(score_ratios)[max(0, len(score_ratios) // 20)],
                               "p95": sorted(score_ratios)[min(len(score_ratios) - 1,
                                                               19 * len(score_ratios) // 20)]},
            "V4_exact_zeros": zeros,
            "V1_candidate_miss_mass": {"max": max(cand), "median": st.median(cand)},
        }
    d = res["strata"].get("decile")
    if d:
        m = d["V2_mass_ratio"]
        res["verdict"] = {
            "V1_held": d["V1_candidate_miss_mass"]["max"] < 1e-6,
            "V2_held": 0.99 <= m["median"] <= 1.01 and m["p99_abs_dev"] < 0.05,
            "V2_falsified": m["median"] > 1.05,
            "V3_held": d["V3_score_ratio"]["within_factor_2"] >= 0.95,
            "V4_held": d["V4_exact_zeros"] == 0,
        }
    (OUT / f"f4_verdict_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("served")
    s.add_argument("--tag", required=True, choices=tuple(MODELS))
    s.add_argument("--endpoint", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--tag", required=True, choices=tuple(MODELS))
    b.add_argument("--endpoint", required=True)
    o = sub.add_parser("offline")
    o.add_argument("--tag", required=True, choices=tuple(MODELS))
    c = sub.add_parser("compare")
    c.add_argument("--tag", required=True, choices=tuple(MODELS))
    args = ap.parse_args()
    if args.cmd != "compare":
        v2p._load_questions()
    return {"served": cmd_served, "baseline": cmd_baseline,
            "offline": cmd_offline, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
