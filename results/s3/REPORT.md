# S3 — independence measurement: NULL RESULT (channel does not threaten extraction)

> **Gate outcome: no valid seating produced — and that is itself the finding.**
> The native-TTS + MUSAN + whisper-large-v3 channel does not induce extraction
> errors, so there is no decorrelation signal to seat on. Details below; the
> auto-generated tables follow but must be read in this light.

**What happened.** 770 transcripts across a 5-point SNR dial down to −10 dB.
Of these, only **40 reached WER > 20%**, and on those 40 genuinely-corrupted
transcripts **all three families extracted with 0% error**. Every extraction
error in the entire pool (18 cases) falls on the two clean-text boundary items
(p07 "phone home", p11 sentiment trap) — instrument boundary, **not** channel —
and all three families make the *same* error there (P(both wrong ∧ same option)
== P(both wrong) at every level). The phi values in the table below are computed
on 3–8 errors per family and are statistical noise.

**Why (the real insight).** Option-mapping is *semantic*: the evaluator reads a
garbled transcript and still recovers the correct a/b/c option, because the gist
survives word-level ASR errors. So at these noise levels, on these
fairly-unambiguous scripted answers, the ASR channel does not move the verdict —
there is nothing for the certificate to catch, and nothing to decorrelate.
whisper-large-v3 is also simply very robust to stationary MUSAN noise (it strips
it), so even −10 dB SNR leaves a ~4% mean WER.

**What this does NOT do.** It does not measure T1 independence, and it does not
seat the judge/evaluator pair. Seating on these numbers would be measuring
clean-text instrument boundary, not channel-induced error correlation.

**Paths to a real S3 (a methodological choice — see handoff):**
1. **Ambiguous/hedged answer bank** — the signal that *did* appear is at genuinely
   near-boundary answers (p07/p11). An answer bank written to sit near option
   boundaries, where ASR corruption tips interpretation, would produce real
   channel-induced flips. Instrument+channel co-design.
2. **Accented speech (the real E3 axis)** — L2-ARCTIC / accented TTS causes
   *systematic* (not random) errors that survive as semantic shifts. Needs voice
   cloning (coqui blocked) or accented TTS voices.
3. **Babble at low SNR** — MUSAN `speech` (overlapping talkers) is the ASR
   Achilles heel; untested here (load-time timed out) but promising.
4. **Weaker judge-side ASR** — realistic for field deployment, but changes a
   pinned model choice.

Recommendation: proceed to M1 on the **a-priori** seating (judge = Qwen3-32B, the
better conversationalist per S1a; evaluator = Gemma-3-27B, different family/
tokenizer per build-guide §1) — S3 found no reason to override it — and defer the
*measured* decorrelation-seating until the channel actually induces errors
(option 1 or 2). M1 still demonstrates the end-to-end certificate loop.

---

# S3 — independence measurement + seating

Pool: 770 channel-corrupted transcripts (77 gold utterances × levels × seeds; edge-tts en-US-GuyNeural → 8 kHz μ-law → MUSAN → whisper-large-v3). Canonical extraction for all three families.

**Scope caveat:** measured on ONE native TTS voice + noise. The accent axis is
deferred to E3; this seating is NOT claimed accent-robust.

## Channel WER dial (build-guide §3)

| level | n | mean WER | median | max |
|---|---|---|---|---|
| tel | 154 | 3.6% | 0.0% | 33% |
| snr5 | 154 | 3.6% | 0.0% | 50% |
| snr0 | 154 | 3.7% | 0.0% | 50% |
| snr_m5 | 154 | 3.5% | 0.0% | 50% |
| snr_m10 | 154 | 3.8% | 0.0% | 50% |

## Level `snr_m10` (PRIMARY) — n=154

accuracy: Qwen3-32B 97.4% · Gemma-3-27B 95.5% · Mistral-24B 98.1%

| pair | phi(errors) | P(both wrong) | P(both wrong ∧ same option) |
|---|---|---|---|
| Qwen3-32B × Gemma-3-27B | 0.552 | 1.9% | 1.9% |
| Qwen3-32B × Mistral-24B | 0.863 | 1.9% | 1.9% |
| Gemma-3-27B × Mistral-24B | 0.646 | 1.9% | 1.9% |

### Per-mechanism error rates (snr_m10)

| mechanism | n | Qwen3-32B | Gemma-3-27B | Mistral-24B |
|---|---|---|---|---|
| clause_deletion | 28 | 0% | 0% | 0% |
| hedge_grading | 42 | 5% | 10% | 5% |
| list_speech | 14 | 0% | 0% | 0% |
| minimal_pair | 42 | 5% | 10% | 5% |
| mixed_proposition | 28 | 0% | 0% | 0% |
| negation_drop | 56 | 0% | 4% | 0% |
| number_confusion | 28 | 7% | 4% | 4% |
| politeness_mask | 28 | 0% | 7% | 0% |
| sentiment_trap | 14 | 14% | 7% | 7% |
| two_correct | 14 | 0% | 0% | 0% |

### Seating

Least-correlated pair: **Qwen3-32B × Gemma-3-27B** (phi 0.552, joint-same-wrong 1.9%).

Seated: **judge = Qwen3-32B** (better conversationalist per S1a renders), **evaluator = Gemma-3-27B**. Third family (Mistral-24B) sits out, kept for the E1b/robustness ablation.
## Level `snr_m5` — n=154

accuracy: Qwen3-32B 97.4% · Gemma-3-27B 95.5% · Mistral-24B 98.1%

| pair | phi(errors) | P(both wrong) | P(both wrong ∧ same option) |
|---|---|---|---|
| Qwen3-32B × Gemma-3-27B | 0.552 | 1.9% | 1.9% |
| Qwen3-32B × Mistral-24B | 0.863 | 1.9% | 1.9% |
| Gemma-3-27B × Mistral-24B | 0.646 | 1.9% | 1.9% |

## Level `snr0` — n=154

accuracy: Qwen3-32B 97.4% · Gemma-3-27B 94.8% · Mistral-24B 97.4%

| pair | phi(errors) | P(both wrong) | P(both wrong ∧ same option) |
|---|---|---|---|
| Qwen3-32B × Gemma-3-27B | 0.698 | 2.6% | 2.6% |
| Qwen3-32B × Mistral-24B | 1.000 | 2.6% | 2.6% |
| Gemma-3-27B × Mistral-24B | 0.698 | 2.6% | 2.6% |

## Level `snr5` — n=154

accuracy: Qwen3-32B 97.4% · Gemma-3-27B 94.8% · Mistral-24B 97.4%

| pair | phi(errors) | P(both wrong) | P(both wrong ∧ same option) |
|---|---|---|---|
| Qwen3-32B × Gemma-3-27B | 0.698 | 2.6% | 2.6% |
| Qwen3-32B × Mistral-24B | 1.000 | 2.6% | 2.6% |
| Gemma-3-27B × Mistral-24B | 0.698 | 2.6% | 2.6% |

## Level `tel` — n=154

accuracy: Qwen3-32B 97.4% · Gemma-3-27B 94.8% · Mistral-24B 97.4%

| pair | phi(errors) | P(both wrong) | P(both wrong ∧ same option) |
|---|---|---|---|
| Qwen3-32B × Gemma-3-27B | 0.698 | 2.6% | 2.6% |
| Qwen3-32B × Mistral-24B | 1.000 | 2.6% | 2.6% |
| Gemma-3-27B × Mistral-24B | 0.698 | 2.6% | 2.6% |


Diagnostic — emitted≠argmax rows (constrained-decode anomaly tracking): Qwen3-32B 0 · Gemma-3-27B 0 · Mistral-24B 0

