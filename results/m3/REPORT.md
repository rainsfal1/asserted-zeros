# M3 — voice in the loop: the accent axis

Two frozen pools. **arctic**: 864 transcriptions of real non-native speech (L2-ARCTIC, 12 sampled utterances × 24 speakers × 3 conditions) — the channel measured on real accents. **items**: 1848 elicitations of our 11-item instrument (88 texts × 8 accented TTS voices × 3 levels) — the instrument measured under the accent proxy. Accented TTS is respondent simulation; the arctic table is what ties the proxy to real recordings.

## The channel on REAL accented speech (L2-ARCTIC)

| L1 | clean WER | tel | tel+0 dB | n/cond |
|---|---|---|---|---|
| Arabic | 9.3% | 9.8% | 16.2% | 48 |
| Chinese | 12.4% | 12.2% | 23.4% | 48 |
| Hindi | 6.0% | 5.7% | 10.0% | 48 |
| Korean | 6.2% | 6.2% | 14.5% | 48 |
| Spanish | 7.1% | 7.8% | 13.2% | 48 |
| Vietnamese | 17.7% | 17.9% | 23.6% | 48 |
| **all** | **9.8%** | **9.9%** | **16.8%** | 288 |

S3's native-voice channel sat at ~3.5% mean WER at every level. Real accented speech starts higher and the channel compounds it — accent is the variable that moves the dial.

## Our items under the accent proxy

| voice | mean WER | extraction err (judge) | (evaluator) | n |
|---|---|---|---|---|
| IN | 3.7% | 3.0% | 5.2% | 231 |
| NG | 5.0% | 3.5% | 5.2% | 231 |
| PH | 3.9% | 4.3% | 6.5% | 231 |
| ZA | 3.8% | 3.9% | 6.5% | 231 |
| KE | 6.2% | 4.8% | 6.5% | 231 |
| SG | 5.1% | 2.6% | 3.9% | 231 |
| HK | 3.8% | 2.6% | 5.2% | 231 |
| TZ | 5.1% | 3.0% | 5.6% | 231 |

Totals: judge 64/1848 (3.46%), evaluator 103/1848 (5.57%).

### Channel-induced vs semantic errors (the S3-null retest)

- **judge**: 6 errors on VERBATIM transcripts (semantic — the wording itself splits the extractor); 58 errors on the 955 channel-altered transcripts (6.07% of altered rows).
- **evaluator**: 27 errors on VERBATIM transcripts (semantic — the wording itself splits the extractor); 76 errors on the 955 channel-altered transcripts (7.96% of altered rows).

S3 (native voice) found the channel induced ~no extraction errors. Under accented voices the altered-transcript share and its error rate are the retest: errors on altered rows are the channel reaching the verdict, which is the failure mode the certificate exists to catch.

## Exchangeability inventory (why this pool exists)

- 1848 rows · 77 distinct utterances · **616 distinct utterance×voice units** · 349 distinct (item, transcript) pairs.
- Per item: ≥56 utterance×voice units → per-item conformal floor α ≥ 1/57 = 0.0175 → Σᵢαᵢ floor 0.193 under the utterance×voice exchangeability reading.
- M2 on the S3 pool had 7 utterances/item (Σᵢαᵢ ≥ 1.375, vacuous). Whether utterance×voice is the honest unit — voices break transcript dependence, but the wording is still shared — is reported both ways by the M2 rerun on this pool.

## Caveats

- Accented TTS ≠ real L2 speech. The proxy's WER is tied to reality only through the arctic table; claims about real speakers wait for cloned or collected voice (XTTS-v2 in a separate venv, or a cohort).
- One seed per level (the S3 pool showed seed-to-seed variation is second-order next to voice/level).
- Judge and evaluator share whisper as the ASR front-end here — threat T1's ASR-sharing arm is NOT exercised; a second ASR family (Canary/Parakeet) is the remaining E2 infrastructure item.
