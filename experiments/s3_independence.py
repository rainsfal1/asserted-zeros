"""S3 — independence measurement + seating (build-guide §4, the first real result).

Three stages, run separately so the GPU serves only during `extract`:

  corrupt  : all 77 gold-labeled utterances × levels × seeds through the channel
             (edge-tts → telephony → MUSAN → whisper-CPU) → transcripts.jsonl
             (the FROZEN pool — committed; edge-tts is a cloud dep and can drift).
  extract  : one served family canonical-extracts every pooled transcript.
  analyze  : per-pair error correlation (phi), P(both wrong), and
             P(both wrong ∧ same wrong option) — the case disagreement cannot
             catch — plus per-mechanism breakdown; seats the least-correlated
             pair. Seating is measured on ONE native TTS voice; the accent axis
             is E3's, and the seat is NOT claimed accent-robust.

    python experiments/s3_independence.py corrupt  --out results/s3
    python experiments/s3_independence.py extract  --endpoint URL --tag judge --out results/s3
    python experiments/s3_independence.py analyze  --out results/s3
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verdict_cert.items import load_items  # noqa: E402

SEEDS = (1, 2)
TAGS = ("judge", "evaluator", "third")
FAMILY = {"judge": "Qwen3-32B", "evaluator": "Gemma-3-27B", "third": "Mistral-24B"}

#: Tags this script may EXTRACT for. Deliberately wider than TAGS, which drives
#: cmd_analyze's pairwise loop and the seating gate — widening TAGS itself would
#: change S3's committed analysis and demand an extract file per tag. cmd_extract
#: is pool-agnostic (it reads whatever transcripts.jsonl is under --out), which is
#: how results/m3/extract-third.jsonl was produced by this script. B2 adds the
#: fourth extractor under its MODEL name: "weak" is a conclusion, not an id.
EXTRACT_TAGS = TAGS + ("qwen8b",)


def cases():
    for it in load_items():
        for gold, utt in it.clean_examples():
            yield it, gold, utt, "bank"
        fp = it.fragile_probe
        yield it, fp.clean_maps_to, fp.utterance, "probe_clean"


def _corrupt_case(payload: dict) -> list[dict]:
    """Worker: synthesize one utterance once, then degrade+transcribe each pending
    (level, seed) variant. Runs in its own process (whisper instance per worker)."""
    from verdict_cert.channel import degrade, synthesize, transcribe, wer

    wav, sr = synthesize(payload["utterance"])
    out = []
    for level, seed, uid in payload["variants"]:
        dwav, dsr = degrade(wav, sr, level, seed)
        hyp = transcribe(dwav, dsr)
        out.append(
            {
                "id": uid,
                "item": payload["item"],
                "kind": payload["kind"],
                "gold": payload["gold"],
                "level": level,
                "seed": seed,
                "utterance": payload["utterance"],
                "transcript": hyp,
                "wer": round(wer(payload["utterance"], hyp), 4),
                "mechanisms": payload["mechanisms"],
            }
        )
    return out


def cmd_corrupt(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from verdict_cert.channel import LEVELS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool_path = out / "transcripts.jsonl"
    done = set()
    if pool_path.exists():  # idempotent resume
        done = {json.loads(line)["id"] for line in pool_path.open()}
        print(f"resuming: {len(done)} entries already pooled")

    payloads = []
    for idx, (it, gold, utt, kind) in enumerate(cases()):
        variants = [
            (level, seed, f"{it.id}:{idx}:{level}:s{seed}")
            for level in LEVELS
            for seed in SEEDS
            if f"{it.id}:{idx}:{level}:s{seed}" not in done
        ]
        if variants:
            payloads.append(
                {
                    "item": it.id,
                    "kind": kind,
                    "gold": gold,
                    "utterance": utt,
                    "mechanisms": list(it.mechanisms),
                    "variants": variants,
                }
            )
    print(f"[corrupt] {len(payloads)} utterances to process ({args.workers} workers)")

    n_new = 0
    with pool_path.open("a") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_corrupt_case, p) for p in payloads]
        for i, fut in enumerate(as_completed(futures)):
            for rec in fut.result():
                f.write(json.dumps(rec) + "\n")
                n_new += 1
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(payloads)} utterances done ({n_new} new entries)")

    pool_rows = [json.loads(line) for line in pool_path.open()]
    for level in LEVELS:
        ws = [r["wer"] for r in pool_rows if r["level"] == level]
        print(f"[corrupt] level={level}: n={len(ws)} mean WER {sum(ws)/len(ws):.1%}")
    print(f"[corrupt] pool size {len(pool_rows)} -> {pool_path}")
    return 0


def cmd_extract(args) -> int:
    from verdict_cert.extraction import Extractor

    out = Path(args.out)
    items = {it.id: it for it in load_items()}
    pool = [json.loads(line) for line in (out / "transcripts.jsonl").open()]
    ex = Extractor(args.endpoint)
    path = out / f"extract-{args.tag}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.open()}
    # Qwen3 needs enable_thinking=False or <think> tokens displace the decision
    # token entirely. The flag is auto-detected; record it, because a run whose
    # validity depends on it should not leave that to inference.
    if "qwen3" in ex.model.lower():
        assert ex.disable_thinking, f"{ex.model} needs enable_thinking=False"
    print(f"[{args.tag}] model={ex.model}; thinking_disabled={ex.disable_thinking}; "
          f"{len(pool)} transcripts, {len(done)} cached")
    with path.open("a") as f:
        for i, rec in enumerate(pool):
            if rec["id"] in done:
                continue
            r = ex.canonical(items[rec["item"]], rec["transcript"])
            f.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "pred": r.choice,
                        "emitted": r.emitted,
                        "diverged": r.diverged,
                        "dist": {k: round(v, 5) for k, v in r.dist.items()},
                    }
                )
                + "\n"
            )
            if (i + 1) % 50 == 0:
                f.flush()
                print(f"  {i + 1}/{len(pool)}")
    print(f"[{args.tag}] done -> {path}")
    return 0


def phi(x: list[bool], y: list[bool]) -> float:
    n11 = sum(a and b for a, b in zip(x, y))
    n00 = sum((not a) and (not b) for a, b in zip(x, y))
    n10 = sum(a and (not b) for a, b in zip(x, y))
    n01 = sum((not a) and b for a, b in zip(x, y))
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / denom if denom else float("nan")


def cmd_analyze(args) -> int:
    out = Path(args.out)
    pool = {r["id"]: r for r in map(json.loads, (out / "transcripts.jsonl").open())}
    preds = {
        t: {r["id"]: r for r in map(json.loads, (out / f"extract-{t}.jsonl").open())}
        for t in TAGS
    }
    ids_all = sorted(pool)
    lines = [
        "# S3 — independence measurement + seating",
        "",
        f"Pool: {len(ids_all)} channel-corrupted transcripts "
        "(77 gold utterances × levels × seeds; edge-tts en-US-GuyNeural → 8 kHz μ-law "
        "→ MUSAN → whisper-large-v3). Canonical extraction for all three families.",
        "",
        "**Scope caveat:** measured on ONE native TTS voice + noise. The accent axis is",
        "deferred to E3; this seating is NOT claimed accent-robust.",
        "",
    ]

    # WER dial (recomputed consistently from stored utterance+transcript).
    from verdict_cert.channel import wer as _wer

    order = ("tel", "snr5", "snr0", "snr_m5", "snr_m10")
    lines += ["## Channel WER dial (build-guide §3)", "",
              "| level | n | mean WER | median | max |", "|---|---|---|---|---|"]
    for level in order:
        ws = sorted(_wer(pool[i]["utterance"], pool[i]["transcript"])
                    for i in ids_all if pool[i]["level"] == level)
        if ws:
            med = ws[len(ws) // 2]
            lines.append(f"| {level} | {len(ws)} | {sum(ws)/len(ws):.1%} | {med:.1%} | {max(ws):.0%} |")
    lines.append("")

    for level in ("snr_m10", "snr_m5", "snr0", "snr5", "tel"):
        ids = [i for i in ids_all if pool[i]["level"] == level]
        if not ids:
            continue
        err = {t: [preds[t][i]["pred"] != pool[i]["gold"] for i in ids] for t in TAGS}
        acc = {t: 1 - sum(e) / len(e) for t, e in err.items()}
        primary = " (PRIMARY)" if level == "snr_m10" else ""
        lines += [
            f"## Level `{level}`{primary} — n={len(ids)}",
            "",
            "accuracy: " + " · ".join(f"{FAMILY[t]} {acc[t]:.1%}" for t in TAGS),
            "",
            "| pair | phi(errors) | P(both wrong) | P(both wrong ∧ same option) |",
            "|---|---|---|---|",
        ]
        stats = {}
        for a, b in itertools.combinations(TAGS, 2):
            both = [ea and eb for ea, eb in zip(err[a], err[b])]
            same_wrong = [
                ea and eb and preds[a][i]["pred"] == preds[b][i]["pred"]
                for ea, eb, i in zip(err[a], err[b], ids)
            ]
            stats[(a, b)] = {
                "phi": phi(err[a], err[b]),
                "p_both": sum(both) / len(ids),
                "p_same_wrong": sum(same_wrong) / len(ids),
            }
            s = stats[(a, b)]
            lines.append(
                f"| {FAMILY[a]} × {FAMILY[b]} | {s['phi']:.3f} | {s['p_both']:.1%} "
                f"| {s['p_same_wrong']:.1%} |"
            )
        lines.append("")
        if level == "snr_m10":
            # mechanism breakdown at the primary level
            lines += ["### Per-mechanism error rates (snr_m10)", "",
                      "| mechanism | n | " + " | ".join(FAMILY[t] for t in TAGS) + " |",
                      "|---|---|" + "---|" * len(TAGS)]
            mechs = sorted({m for i in ids for m in pool[i]["mechanisms"]})
            for m in mechs:
                mids = [i for i in ids if m in pool[i]["mechanisms"]]
                rates = [
                    sum(preds[t][i]["pred"] != pool[i]["gold"] for i in mids) / len(mids)
                    for t in TAGS
                ]
                lines.append(
                    f"| {m} | {len(mids)} | " + " | ".join(f"{r:.0%}" for r in rates) + " |"
                )
            lines.append("")
            # seating on the primary level
            best = min(stats, key=lambda k: (stats[k]["phi"], stats[k]["p_same_wrong"]))
            expected = ("judge", "evaluator")
            seat_ok = set(best) == set(expected)
            seating = {
                "least_correlated_pair": [FAMILY[t] for t in best],
                "phi": stats[best]["phi"],
                "p_same_wrong": stats[best]["p_same_wrong"],
                "expected": seat_ok,
                "seated_judge": "Qwen3-32B" if seat_ok else None,
                "seated_evaluator": "Gemma-3-27B" if seat_ok else None,
            }
            (out / "seating.json").write_text(json.dumps(seating, indent=2))
            lines += [
                "### Seating",
                "",
                f"Least-correlated pair: **{FAMILY[best[0]]} × {FAMILY[best[1]]}** "
                f"(phi {stats[best]['phi']:.3f}, joint-same-wrong {stats[best]['p_same_wrong']:.1%}).",
                "",
            ]
            if seat_ok:
                lines.append(
                    "Seated: **judge = Qwen3-32B** (better conversationalist per S1a renders), "
                    "**evaluator = Gemma-3-27B**. Third family (Mistral-24B) sits out, kept "
                    "for the E1b/robustness ablation."
                )
            else:
                lines.append(
                    "**GATE S3: SURPRISE — least-correlated pair is not Qwen×Gemma. "
                    "STOP: do not build M1 on this seating without review.**"
                )
    div = {
        t: sum(1 for i in ids_all if preds[t][i].get("diverged")) for t in TAGS
    }
    lines += ["", "Diagnostic — emitted≠argmax rows (constrained-decode anomaly tracking): "
              + " · ".join(f"{FAMILY[t]} {div[t]}" for t in TAGS), ""]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-25:]))
    print(f"wrote {out / 'REPORT.md'}")
    seating = json.loads((out / "seating.json").read_text())
    return 0 if seating["expected"] else 3  # rc=3 signals the seating surprise


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("corrupt", "extract", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="results/s3")
        if name == "corrupt":
            p.add_argument("--workers", type=int, default=10)
        if name == "extract":
            p.add_argument("--endpoint", required=True)
            p.add_argument("--tag", required=True, choices=EXTRACT_TAGS)
    args = ap.parse_args()
    return {"corrupt": cmd_corrupt, "extract": cmd_extract, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
