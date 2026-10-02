"""M3 — voice in the loop: the ACCENT axis the S3 null result pointed at.

S3 found that telephony + noise over one native TTS voice induces ~no extraction
errors — whisper is too robust and the transcripts arrive clean. The real threat
named there (and in build-guide §3 layer 1) is accented speech. M3 turns that
dial two ways, keeping S3's frozen-pool discipline throughout:

  arctic   REAL accented speech: L2-ARCTIC recordings (24 non-native speakers,
           6 L1s, ground-truth transcripts) through the same telephony/noise
           channel → whisper. Measures the per-L1 WER dose-response — the
           evidence that accent moves the dial synthetic speech could not.
           (L2-ARCTIC reads CMU ARCTIC prompts, not our items, so this measures
           the CHANNEL, not extraction.)
  corrupt  OUR items in accented voices: the 77 utterances (66 bank + 11 probes)
           synthesized in 8 accented English voices (edge-tts: IN NG PH ZA KE
           SG HK TZ) × 3 channel levels → whisper → the frozen M3 transcript
           pool. Same record schema as S3, plus a ``voice`` field — which is
           what lets every downstream harness (M2 calibration included) run on
           this pool unchanged.
  extract  one served family canonical-extracts every pooled transcript
           (identical to S3's extract; run once per family, GPU sequential).
  analyze  per-accent WER + extraction accuracy, channel-induced vs semantic
           errors, and the pool's exchangeability inventory (groups by
           utterance and by utterance×voice — the M2 sample-size wall is the
           reason this pool exists).

Accented TTS is respondent SIMULATION (same caveat as S3): synthetic accents are
a proxy for L2 speech, not a claim about real speakers — `arctic` is what ties
the proxy to real recordings. XTTS-v2 voice cloning of the L2-ARCTIC speakers
(a closer proxy) needs a separate venv (coqui-tts conflicts with vLLM) and is
deliberately not part of this pass.

    python experiments/m3_accent_channel.py arctic   --out results/m3
    python experiments/m3_accent_channel.py corrupt  --out results/m3
    python experiments/m3_accent_channel.py extract  --endpoint URL --tag judge --out results/m3
    python experiments/m3_accent_channel.py analyze  --out results/m3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verdict_cert.items import load_items  # noqa: E402

ARCTIC = Path.home() / "vc-lab" / "data" / "l2arctic"

#: One voice per accented-English locale edge-tts offers, gender-balanced.
VOICES = {
    "IN": "en-IN-PrabhatNeural", "NG": "en-NG-EzinneNeural",
    "PH": "en-PH-JamesNeural", "ZA": "en-ZA-LeahNeural",
    "KE": "en-KE-ChilembaNeural", "SG": "en-SG-LunaNeural",
    "HK": "en-HK-SamNeural", "TZ": "en-TZ-ImaniNeural",
}
#: M3 operating points: telephony floor, hard noise, harder noise. S3 showed the
#: clean/tel end of the dial is flat; the accent axis is the new variable.
M3_LEVELS = ("tel", "snr0", "snr_m5")
SEED = 1
TAGS = ("judge", "evaluator")

ARCTIC_CONDS = ("clean", "tel", "snr0")
ARCTIC_UTTS_PER_SPK = 12


def cases():
    for it in load_items():
        for gold, utt in it.clean_examples():
            yield it, gold, utt, "bank"
        fp = it.fragile_probe
        yield it, fp.clean_maps_to, fp.utterance, "probe_clean"


# --------------------------------------------------------------------- arctic


def speaker_l1() -> dict[str, str]:
    """Parse the speaker→native-language table out of the corpus README."""
    txt = (ARCTIC / "README.md").read_text()
    table = {}
    for m in re.finditer(r"^\|([A-Z]{3,5})\|[MF]\|([A-Za-z]+)\|\d+\|", txt, re.M):
        table[m.group(1)] = m.group(2)
    if len(table) != 24:
        raise RuntimeError(f"expected 24 speakers in README table, parsed {len(table)}")
    return table


def _arctic_speaker(payload: dict) -> list[dict]:
    """Worker: one speaker's sampled utterances through every condition."""
    import soundfile as sf

    from verdict_cert.channel import degrade, transcribe, wer

    out = []
    for utt_id, ref in payload["utts"]:
        wav, sr = sf.read(str(ARCTIC / payload["spk"] / "wav" / f"{utt_id}.wav"),
                          dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        for cond in payload["conds"]:
            if cond == "clean":
                dwav, dsr = wav, sr
            else:
                dwav, dsr = degrade(wav, sr, cond, SEED)
            hyp = transcribe(dwav, dsr)
            out.append({
                "id": f"{payload['spk']}:{utt_id}:{cond}",
                "speaker": payload["spk"], "l1": payload["l1"], "cond": cond,
                "ref": ref, "hyp": hyp, "wer": round(wer(ref, hyp), 4),
            })
    return out


def cmd_arctic(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "arctic.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(x)["id"] for x in path.open()}
        print(f"resuming: {len(done)} rows present")

    l1 = speaker_l1()
    rng = random.Random(7)
    payloads = []
    for spk in sorted(l1):
        tdir = ARCTIC / spk / "transcript"
        utt_ids = sorted(p.stem for p in tdir.glob("arctic_*.txt"))
        picks = rng.sample(utt_ids, ARCTIC_UTTS_PER_SPK)
        utts = []
        for u in picks:
            if all(f"{spk}:{u}:{c}" in done for c in ARCTIC_CONDS):
                continue
            utts.append((u, (tdir / f"{u}.txt").read_text().strip()))
        if utts:
            payloads.append({"spk": spk, "l1": l1[spk], "conds": ARCTIC_CONDS, "utts": utts})
    print(f"[arctic] {len(payloads)} speakers to process ({args.workers} workers)")

    with path.open("a") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_arctic_speaker, p) for p in payloads]
        for i, fut in enumerate(as_completed(futs)):
            for rec in fut.result():
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  speaker {i + 1}/{len(payloads)} done")

    rows = [json.loads(x) for x in path.open()]
    for cond in ARCTIC_CONDS:
        ws = [r["wer"] for r in rows if r["cond"] == cond]
        print(f"[arctic] {cond}: n={len(ws)} mean WER {statistics.mean(ws):.1%}")
    return 0


# -------------------------------------------------------------------- corrupt


def _corrupt_case(payload: dict) -> list[dict]:
    """Worker: synthesize one (utterance, voice) once, then degrade+transcribe
    each pending level. One whisper instance per worker process.

    Everything is caught INSIDE the worker: edge-tts failures raise exceptions
    carrying live network objects (SSLContext) that cannot be pickled back
    through the ProcessPoolExecutor, so an escaped exception kills the whole
    run rather than one clip. Synthesis gets retries (cloud flake); a clip that
    still fails is reported as a skip marker and picked up by the next resume.
    """
    import time
    import traceback

    from verdict_cert.channel import degrade, synthesize, transcribe, wer

    wav = sr = None
    for attempt in range(4):
        try:
            wav, sr = synthesize(payload["utterance"], voice=payload["voice_name"])
            break
        except Exception:  # noqa: BLE001 — must not cross the process boundary
            if attempt == 3:
                return [{"_skip": f"{payload['item']}:{payload['voice']}",
                         "_why": traceback.format_exc(limit=1)}]
            time.sleep(2.0 * (attempt + 1))
    out = []
    try:
        for level, uid in payload["variants"]:
            dwav, dsr = degrade(wav, sr, level, SEED)
            hyp = transcribe(dwav, dsr)
            out.append({
                "id": uid, "item": payload["item"], "kind": payload["kind"],
                "gold": payload["gold"], "level": level, "seed": SEED,
                "voice": payload["voice"],
                "utterance": payload["utterance"], "transcript": hyp,
                "wer": round(wer(payload["utterance"], hyp), 4),
                "mechanisms": payload["mechanisms"],
            })
    except Exception:  # noqa: BLE001
        out.append({"_skip": f"{payload['item']}:{payload['voice']}",
                    "_why": traceback.format_exc(limit=1)})
    return out


def cmd_corrupt(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool_path = out / "transcripts.jsonl"
    done = set()
    if pool_path.exists():
        done = {json.loads(x)["id"] for x in pool_path.open()}
        print(f"resuming: {len(done)} entries already pooled")

    payloads = []
    for idx, (it, gold, utt, kind) in enumerate(cases()):
        for acc, voice_name in VOICES.items():
            variants = [
                (level, f"{it.id}:{idx}:{acc}:{level}:s{SEED}")
                for level in M3_LEVELS
                if f"{it.id}:{idx}:{acc}:{level}:s{SEED}" not in done
            ]
            if variants:
                payloads.append({
                    "item": it.id, "kind": kind, "gold": gold, "utterance": utt,
                    "voice": acc, "voice_name": voice_name,
                    "mechanisms": list(it.mechanisms), "variants": variants,
                })
    print(f"[corrupt] {len(payloads)} (utterance × voice) clips to process "
          f"({args.workers} workers)")

    n_new = n_skip = 0
    with pool_path.open("a") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_corrupt_case, p) for p in payloads]
        for i, fut in enumerate(as_completed(futs)):
            for rec in fut.result():
                if "_skip" in rec:
                    n_skip += 1
                    print(f"  ! skipped {rec['_skip']}: {rec['_why'].splitlines()[-1]}")
                    continue
                f.write(json.dumps(rec) + "\n")
                n_new += 1
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(payloads)} clips done ({n_new} new rows, {n_skip} skips)")
    if n_skip:
        print(f"[corrupt] {n_skip} clips skipped after retries — RE-RUN this command to retry them")

    rows = [json.loads(x) for x in pool_path.open()]
    for acc in VOICES:
        ws = [r["wer"] for r in rows if r["voice"] == acc]
        if ws:
            print(f"[corrupt] {acc}: n={len(ws)} mean WER {statistics.mean(ws):.1%}")
    print(f"[corrupt] pool size {len(rows)} -> {pool_path}")
    return 0


# -------------------------------------------------------------------- extract


def cmd_extract(args) -> int:
    from verdict_cert.extraction import Extractor

    out = Path(args.out)
    items = {it.id: it for it in load_items()}
    pool = [json.loads(x) for x in (out / "transcripts.jsonl").open()]
    ex = Extractor(args.endpoint)
    path = out / f"extract-{args.tag}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(x)["id"] for x in path.open()}
    print(f"[{args.tag}] model={ex.model}; {len(pool)} transcripts, {len(done)} cached")
    with path.open("a") as f:
        for i, rec in enumerate(pool):
            if rec["id"] in done:
                continue
            r = ex.canonical(items[rec["item"]], rec["transcript"])
            f.write(json.dumps({
                "id": rec["id"], "pred": r.choice, "emitted": r.emitted,
                "diverged": r.diverged,
                "dist": {k: round(v, 5) for k, v in r.dist.items()},
            }) + "\n")
            if (i + 1) % 100 == 0:
                f.flush()
                print(f"  {i + 1}/{len(pool)}")
    print(f"[{args.tag}] done -> {path}")
    return 0


# -------------------------------------------------------------------- analyze


def cmd_analyze(args) -> int:
    out = Path(args.out)
    arctic = [json.loads(x) for x in (out / "arctic.jsonl").open()]
    pool = [json.loads(x) for x in (out / "transcripts.jsonl").open()]
    preds = {
        t: {r["id"]: r for r in map(json.loads, (out / f"extract-{t}.jsonl").open())}
        for t in TAGS
    }

    L = [
        "# M3 — voice in the loop: the accent axis",
        "",
        f"Two frozen pools. **arctic**: {len(arctic)} transcriptions of real non-native "
        f"speech (L2-ARCTIC, {ARCTIC_UTTS_PER_SPK} sampled utterances × 24 speakers × "
        f"{len(ARCTIC_CONDS)} conditions) — the channel measured on real accents. "
        f"**items**: {len(pool)} elicitations of our 11-item instrument "
        f"(88 texts × {len(VOICES)} accented TTS voices × {len(M3_LEVELS)} levels) — "
        "the instrument measured under the accent proxy. Accented TTS is respondent "
        "simulation; the arctic table is what ties the proxy to real recordings.",
        "",
        "## The channel on REAL accented speech (L2-ARCTIC)",
        "",
        "| L1 | clean WER | tel | tel+0 dB | n/cond |",
        "|---|---|---|---|---|",
    ]
    by_l1: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in arctic:
        by_l1[r["l1"]][r["cond"]].append(r["wer"])
    for l1 in sorted(by_l1):
        c = by_l1[l1]
        L.append(f"| {l1} | {statistics.mean(c['clean']):.1%} "
                 f"| {statistics.mean(c['tel']):.1%} | {statistics.mean(c['snr0']):.1%} "
                 f"| {len(c['clean'])} |")
    allw = {cond: [r["wer"] for r in arctic if r["cond"] == cond] for cond in ARCTIC_CONDS}
    L += [
        f"| **all** | **{statistics.mean(allw['clean']):.1%}** "
        f"| **{statistics.mean(allw['tel']):.1%}** | **{statistics.mean(allw['snr0']):.1%}** "
        f"| {len(allw['clean'])} |",
        "",
        "S3's native-voice channel sat at ~3.5% mean WER at every level. Real accented "
        "speech starts higher and the channel compounds it — accent is the variable that "
        "moves the dial.",
        "",
        "## Our items under the accent proxy",
        "",
        "| voice | mean WER | extraction err (judge) | (evaluator) | n |",
        "|---|---|---|---|---|",
    ]
    ids = [r["id"] for r in pool]
    err = {t: {r["id"]: (preds[t][r["id"]]["pred"] != r["gold"]) for r in pool} for t in TAGS}
    for acc in VOICES:
        rows = [r for r in pool if r["voice"] == acc]
        if not rows:
            continue
        L.append(
            f"| {acc} | {statistics.mean(r['wer'] for r in rows):.1%} "
            f"| {sum(err['judge'][r['id']] for r in rows) / len(rows):.1%} "
            f"| {sum(err['evaluator'][r['id']] for r in rows) / len(rows):.1%} "
            f"| {len(rows)} |"
        )
    tot = {t: sum(err[t].values()) for t in TAGS}
    L += [
        "",
        f"Totals: judge {tot['judge']}/{len(ids)} "
        f"({tot['judge']/len(ids):.2%}), evaluator {tot['evaluator']}/{len(ids)} "
        f"({tot['evaluator']/len(ids):.2%}).",
        "",
        "### Channel-induced vs semantic errors (the S3-null retest)",
        "",
    ]
    # split errors by whether the channel altered the transcript: an error on a
    # VERBATIM transcript is semantic (the wording itself splits the extractor);
    # an error on an altered transcript is the channel reaching the verdict
    ident = {t: [r for r in pool if r["transcript"].strip().lower() ==
                 r["utterance"].strip().lower() and err[t][r["id"]]] for t in TAGS}
    corrupted = {t: [r for r in pool if r["transcript"].strip().lower() !=
                     r["utterance"].strip().lower()] for t in TAGS}
    corr_err = {t: [r for r in corrupted[t] if err[t][r["id"]]] for t in TAGS}
    for t in TAGS:
        n_c = len(corrupted[t])
        L.append(
            f"- **{t}**: {len(ident[t])} errors on VERBATIM transcripts (semantic — the "
            f"wording itself splits the extractor); {len(corr_err[t])} errors on the "
            f"{n_c} channel-altered transcripts "
            f"({len(corr_err[t])/max(n_c,1):.2%} of altered rows)."
        )
    L += [
        "",
        "S3 (native voice) found the channel induced ~no extraction errors. Under accented "
        "voices the altered-transcript share and its error rate are the retest: errors on "
        "altered rows are the channel reaching the verdict, which is the failure mode the "
        "certificate exists to catch.",
        "",
        "## Exchangeability inventory (why this pool exists)",
        "",
    ]
    utts = {r["utterance"] for r in pool}
    uv = {(r["utterance"], r["voice"]) for r in pool}
    trans = {(r["item"], r["transcript"]) for r in pool}
    per_item_uv = min(
        len({(r["utterance"], r["voice"]) for r in pool if r["item"] == it.id})
        for it in load_items()
    )
    L += [
        f"- {len(pool)} rows · {len(utts)} distinct utterances · **{len(uv)} distinct "
        f"utterance×voice units** · {len(trans)} distinct (item, transcript) pairs.",
        f"- Per item: ≥{per_item_uv} utterance×voice units → per-item conformal floor "
        f"α ≥ 1/{per_item_uv + 1} = {1/(per_item_uv+1):.4f} → Σᵢαᵢ floor "
        f"{11/(per_item_uv+1):.3f} under the utterance×voice exchangeability reading.",
        "- M2 on the S3 pool had 7 utterances/item (Σᵢαᵢ ≥ 1.375, vacuous). Whether "
        "utterance×voice is the honest unit — voices break transcript dependence, but the "
        "wording is still shared — is reported both ways by the M2 rerun on this pool.",
        "",
        "## Caveats",
        "",
        "- Accented TTS ≠ real L2 speech. The proxy's WER is tied to reality only through "
        "the arctic table; claims about real speakers wait for cloned or collected voice "
        "(XTTS-v2 in a separate venv, or a cohort).",
        "- One seed per level (the S3 pool showed seed-to-seed variation is second-order "
        "next to voice/level).",
        "- Judge and evaluator share whisper as the ASR front-end here — threat T1's "
        "ASR-sharing arm is NOT exercised; a second ASR family (Canary/Parakeet) is the "
        "remaining E2 infrastructure item.",
    ]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"wrote {out/'REPORT.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("arctic", cmd_arctic), ("corrupt", cmd_corrupt),
                     ("extract", cmd_extract), ("analyze", cmd_analyze)):
        p = sub.add_parser(name)
        p.add_argument("--out", default="results/m3")
        if name in ("arctic", "corrupt"):
            p.add_argument("--workers", type=int, default=8)
        if name == "extract":
            p.add_argument("--endpoint", required=True)
            p.add_argument("--tag", required=True, choices=TAGS)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
