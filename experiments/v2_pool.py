"""The v2 pool — render and extract, exactly as pre-registered.

Design authority: ``docs/v2-pool-preregistration.md`` §2 + §6 (frozen; every parameter
here is pinned there — change nothing without a dated amendment). Two subcommands:

  render    350 students × 10 items × 4 channel draws: edge-tts (one accent voice per
            student, round-robin by student-hash order) → degrade_v2 (per-record noise,
            noise-before-codec, active-speech SNR) → faster-whisper large-v3 (CPU).
            Network + CPU only; resume-by-id; safe to interrupt and rerun.

  extract   blind extraction over the binary {a: correct, b: incorrect} domain via
            ``canonical_measured`` + ``CANONICAL_PROMPT_BINARY`` — ONE call per
            record, per-option mass aggregated over top-k variants, absence RAISES
            (prereg §7.2/§8.5; the registered forced-call transport is inert on
            this box's vLLM, so it was amended before any extraction ran).
            Needs a served model; run once with the judge endpoint, once with the
            evaluator's (sequential serving per ops rules).

    python experiments/v2_pool.py render  --out results/v2_pool [--workers 8]
    python experiments/v2_pool.py extract --out results/v2_pool \\
        --endpoint http://localhost:8001/v1 --tag judge

Record ids: ``q{item}:{student}:{VOICE}:{level}:d{draw}`` — 5 fields, the arity
``metrics.voice_of`` requires; student ids verified colon-free (698/698).
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verdict_cert.channel import degrade_v2, synthesize, transcribe, wer  # noqa: E402

#: Frozen m6 module — for load_rows / load_key / normalize / split_students /
#: hash_str, so the v2 pool's student split and wording normalization are
#: BYTE-IDENTICAL to the scored M6 record's. Loaded by path (m6 is frozen; this
#: is a read-only reuse, the metric_suite pattern).
_spec = _ilu.spec_from_file_location("m6_frozen", REPO / "experiments" / "m6_powergrading.py")
m6 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(m6)

#: Prereg §2 pins — voices from the m3 table, order fixed; levels one per draw.
VOICES = {"IN": "en-IN-PrabhatNeural", "NG": "en-NG-EzinneNeural",
          "PH": "en-PH-JamesNeural", "ZA": "en-ZA-LeahNeural",
          "KE": "en-KE-ChilembaNeural", "SG": "en-SG-LunaNeural",
          "HK": "en-HK-SamNeural", "TZ": "en-TZ-ImaniNeural"}
VOICE_ORDER = tuple(VOICES)  # dict order == the m3 table order, pinned
DRAW_LEVELS = ("tel", "snr0", "snr_m5", "snr_m10")  # draw d = index+1
N_STUDENTS = 350


def pool_students(rows: list[dict]) -> list[str]:
    """The first 350 students by FNV-1a hash of id (prereg §2; seed-free)."""
    students = sorted({r["student"] for r in rows}, key=lambda s: (m6.hash_str(s), s))
    return students[:N_STUDENTS]


def voice_for(rank: int) -> str:
    """Round-robin by student-hash order (prereg §2)."""
    return VOICE_ORDER[rank % len(VOICE_ORDER)]


def plan(rows: list[dict]) -> tuple[list[dict], list[tuple]]:
    """(records to render, empty-answer responses excluded), deterministic.

    Unspeakable answers are excluded HERE and listed in the manifest — a
    disclosure, not a silent drop. "Unspeakable" = no word characters: empty
    strings AND punctuation-only answers ('...', '???'), which edge-tts rejects
    with NoAudioReceived — found by the first full render (2 of 3,500)."""
    chosen = pool_students(rows)
    rank = {s: i for i, s in enumerate(chosen)}
    by = {(r["student"], r["item"]): r for r in rows if r["student"] in rank}
    out, empty = [], []
    for (s, it), r in sorted(by.items(), key=lambda kv: (rank[kv[0][0]], kv[0][1])):
        answer = r["answer"].strip()
        if not any(ch.isalnum() for ch in answer):
            empty.append((s, it))
            continue
        acc = voice_for(rank[s])
        for d, level in enumerate(DRAW_LEVELS, start=1):
            out.append({
                "id": f"q{it}:{s}:{acc}:{level}:d{d}",
                "student": s, "item": it, "gold": r["gold"],
                "answer": answer, "wording": m6.normalize(r["answer"]),
                "voice": VOICES[acc], "voice_code": acc,
                "level": level, "draw": d, "kind": "bank",
            })
    return out, empty


def _render_case(case: dict) -> dict:
    """One (student, item): synthesize ONCE, degrade per level, transcribe.
    Exceptions become skip-markers (edge-tts errors carry unpicklable state —
    the m3 lesson)."""
    try:
        wav, sr = None, None
        for attempt in range(4):
            try:
                wav, sr = synthesize(case["records"][0]["answer"], case["records"][0]["voice"])
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2.0 * (attempt + 1))
        out = []
        for rec in case["records"]:
            dwav, dsr = degrade_v2(wav, sr, rec["level"], rec["id"])
            transcript = transcribe(dwav, dsr)
            row = dict(rec)
            row["transcript"] = transcript
            row["wer"] = wer(rec["answer"], transcript)
            out.append(row)
        return {"ok": out}
    except Exception as e:  # noqa: BLE001 — worker boundary; reported, not raised
        return {"_skip": case["records"][0]["id"], "_why": f"{type(e).__name__}: {e}"}


def cmd_render(args) -> int:
    rows = m6.load_rows()
    todo, empty = plan(rows)
    chosen = pool_students(rows)

    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    pool_path = outdir / "records.jsonl"
    done = set()
    if pool_path.exists():
        done = {json.loads(x)["id"] for x in pool_path.open() if x.strip()}
    # group by (student, item): one TTS per response, 4 degrades
    cases: dict[tuple, dict] = {}
    for rec in todo:
        if rec["id"] in done:
            continue
        cases.setdefault((rec["student"], rec["item"]), {"records": []})["records"].append(rec)
    manifest = {"n_students": len(chosen), "students_hash_first": chosen[:3],
                "planned_records": len(todo), "already_done": len(done),
                "empty_answers_excluded": sorted(f"{s}/{it}" for s, it in empty),
                "voices": VOICES, "draw_levels": DRAW_LEVELS}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"render: {len(cases)} responses to go ({len(done)} records already done, "
          f"{len(manifest['empty_answers_excluded'])} empty answers excluded)")
    if not cases:
        return 0

    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_skip = n_ok = 0
    with pool_path.open("a") as sink, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_render_case, c): k for k, c in cases.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if "_skip" in res:
                n_skip += 1
                print(f"  SKIP {res['_skip']}: {res['_why']}")
                continue
            for row in res["ok"]:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            n_ok += 1
            if i % 50 == 0:
                print(f"  {i}/{len(futs)} responses ({n_ok} ok, {n_skip} skipped)")
    print(f"render done: {n_ok} responses rendered, {n_skip} skipped")
    return 0 if n_skip == 0 else 1


def cmd_extract(args) -> int:
    from types import SimpleNamespace

    from verdict_cert.extraction import CANONICAL_PROMPT_BINARY, Extractor

    key = m6.load_key()
    outdir = REPO / args.out
    pool_path = outdir / "records.jsonl"
    recs = [json.loads(x) for x in pool_path.open() if x.strip()]
    out_path = outdir / f"extract-{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(x)["id"] for x in out_path.open() if x.strip()}
    ex = Extractor(args.endpoint)
    print(f"extract[{args.tag}]: model={ex.model}, {len(recs) - len(done)} of {len(recs)} to go")
    with out_path.open("a") as sink:
        for i, r in enumerate(recs):
            if r["id"] in done:
                continue
            item_q = QUESTIONS[r["item"]]
            acceptable = "; ".join(key.get(r["item"], []))
            po = ex.canonical_measured(
                SimpleNamespace(question=item_q), r["transcript"],
                options=("a", "b"),
                meanings={"a": acceptable, "b": ""},
                prompt_template=CANONICAL_PROMPT_BINARY,
                top_k=args.top_k,
            )
            sink.write(json.dumps({
                "id": r["id"], "choice": po.choice,
                # FULL PRECISION, deliberately. The whole point of the §7.2
                # transport is that P(option) is measured, so a measured 4e-7
                # must not become 0.0 on the way to disk — rounding to 6dp
                # re-manufactures F-11's exactly-zero spike at serialization
                # time, after the transport got it right (audit wave 5).
                "dist": dict(po.dist),
                "raw_logprob": dict(po.raw_logprob),
                "mass_unnormalized": po.mass_unnormalized,
            }) + "\n")
            if (i + 1) % 200 == 0:
                sink.flush()
                print(f"  {i + 1}/{len(recs)}")
    print("extract done")
    return 0


#: The ten USCIS questions carrying gold, verbatim from the corpus questions
#: file — item id -> question text. Filled at import from the key TSV's sibling.
QUESTIONS: dict[str, str] = {}


def _load_questions() -> None:
    qpath = m6.DATA / "questions_answer_key.tsv"
    import csv
    with open(qpath) as f:
        r = csv.DictReader(f, delimiter="\t")
        fields = r.fieldnames
        qcol = next(c for c in fields if "question" in c.lower() and "id" not in c.lower())
        idcol = next(c for c in fields if c.strip().lower() in ("q#", "id", "qid", "question_id"))
        for row in r:
            QUESTIONS[row[idcol]] = row[qcol].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", default="results/v2_pool")
    r.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("extract")
    e.add_argument("--out", default="results/v2_pool")
    e.add_argument("--endpoint", required=True)
    e.add_argument("--tag", required=True, choices=("judge", "evaluator"))
    e.add_argument("--top-k", type=int, default=100, dest="top_k",
                   help="top_logprobs width; the server must allow it "
                        "(--max-logprobs). Prereg §7.2 pins the server maximum.")
    args = ap.parse_args()
    _load_questions()
    return {"render": cmd_render, "extract": cmd_extract}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
