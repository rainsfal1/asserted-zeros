"""The degraded voice channel (build-guide §3, layers 2+3) — text → noisy transcript.

Pipeline: edge-tts synthesis → 8 kHz resample + μ-law codec (telephony) → MUSAN
background noise at a target SNR → faster-whisper large-v3 transcription (CPU;
the GPU belongs to vLLM). Audio I/O is **soundfile only** — ``torchaudio.load``
is broken on this box (no system FFmpeg); ``torchaudio.functional`` tensor ops
are fine and are what we use for resampling and μ-law.

edge-tts is a cloud service (the only TTS that can share this venv — coqui-tts
conflicts with vLLM's transformers pin). It is respondent simulation, NOT part
of the certified path, so a cloud dependency is acceptable (build-guide §1
lists edge-tts for TTS respondents); the corrupted-transcript pool is frozen
into the repo so downstream results stay reproducible if the service drifts.
"""

from __future__ import annotations

import asyncio
import io
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

MUSAN_NOISE = Path.home() / "vc-lab" / "data" / "musan" / "noise"
VOICE = "en-US-GuyNeural"  # one fixed native voice for S3 — accent axis is E3's job

#: Channel operating points — a WER dose-response dial (build-guide §0/§3). "tel" =
#: telephony only; "snrN" = telephony + MUSAN at N dB SNR. Measured 2026-07-18: whisper
#: large-v3 is very robust on clean TTS speech (tel 5.0%, +5 dB 5.9%, 0 dB 6.7% mean WER),
#: so the channel must be pushed into NEGATIVE SNR (noise louder than speech) to turn the
#: dial. Negative levels are the primary regime for the independence measurement.
LEVELS = ("tel", "snr5", "snr0", "snr_m5", "snr_m10")
_SNR_DB = {"snr5": 5.0, "snr0": 0.0, "snr_m5": -5.0, "snr_m10": -10.0}


def synthesize(text: str, voice: str = VOICE) -> tuple[np.ndarray, int]:
    """Text → mono float32 waveform via edge-tts (mp3 stream decoded by libsndfile)."""
    import edge_tts

    async def _run() -> bytes:
        buf = b""
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return buf

    audio_bytes = asyncio.run(_run())
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def degrade(wav: np.ndarray, sr: int, level: str, seed: int) -> tuple[np.ndarray, int]:
    """Apply the telephony (+ optional MUSAN) channel; deterministic per seed."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    t = torch.from_numpy(wav).unsqueeze(0)
    t8 = torchaudio.functional.resample(t, sr, 8000).clamp(-1, 1)
    mu = torchaudio.functional.mu_law_encoding(t8, 256)
    dec = torchaudio.functional.mu_law_decoding(mu, 256)
    out = dec.numpy()[0]

    if level in _SNR_DB:
        from audiomentations import AddBackgroundNoise

        snr = _SNR_DB[level]
        random.seed(seed)
        np.random.seed(seed)
        aug = AddBackgroundNoise(
            sounds_path=str(MUSAN_NOISE), min_snr_db=snr, max_snr_db=snr, p=1.0
        )
        out = aug(samples=out, sample_rate=8000)
        peak = float(np.abs(out).max())
        if peak > 1.0:  # keep the mix in range — uncontrolled clipping is not part of the channel
            out = out / peak
    return out, 8000


_WHISPER = None


def transcribe(wav: np.ndarray, sr: int) -> str:
    """faster-whisper large-v3, CPU int8 (leaves the GPU to vLLM)."""
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        _WHISPER = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segments, _info = _WHISPER.transcribe(wav if sr == 16000 else _resample(wav, sr), language="en")
    return " ".join(s.text.strip() for s in segments).strip()


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    t = torch.from_numpy(wav).unsqueeze(0)
    return torchaudio.functional.resample(t, sr, 16000).numpy()[0]


def _norm_words(s: str) -> list[str]:
    """Lowercase, strip surrounding punctuation per token (so 'No,' == 'no' and
    trailing '.' doesn't count as a word error). Formatting differences like
    'four thousand' vs '$4,000' remain real word errors — that is the channel."""
    return [w.strip(".,;:!?\"'()").lower() for w in s.split() if w.strip(".,;:!?\"'()")]


def wer(ref: str, hyp: str) -> float:
    """Word error rate via Levenshtein distance (stdlib only, for the dial log)."""
    r = _norm_words(ref)
    h = _norm_words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(r), len(h)]) / max(len(r), 1)


# ---------------------------------------------------------------------------
# degrade_v2 — the v2 channel (audit wave 2 findings fixed, v1 kept verbatim)
# ---------------------------------------------------------------------------
#
# What v1 got wrong (research/audit-wave-2.md, measured):
#   1. `random.seed(seed)` before AddBackgroundNoise made the noise file a
#      function of the SEED alone — all of S3 drew from 2 of 930 MUSAN files
#      and all of M3 from one 0.55 s clip, tiled (hence periodic).
#   2. Noise was added AFTER the μ-law codec — "telephony speech + studio
#      noise", not telephony(speech + noise).
#   3. Nominal SNR was global-RMS over the whole waveform including TTS
#      lead-in/out silence — ~2 dB optimistic vs active speech.
#
# v2 fixes all three: the noise draw is a pure function of (record_id, level)
# via FNV-1a over a sorted file list (every record gets its own clip and
# offset, deterministically, with no global RNG); noise is mixed BEFORE the
# telephony chain at the speech sample rate; and the target SNR is computed
# over ACTIVE-SPEECH frames (energy VAD at −40 dB rel peak). `degrade()` above
# is byte-untouched so the committed v1 pools reproduce.

_NOISE_CACHE: dict[str, list[Path]] = {}


def _fnv1a(s: str) -> int:
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _noise_files(sounds_dir: Path) -> list[Path]:
    key = str(sounds_dir.resolve())
    if key not in _NOISE_CACHE:
        files = sorted(sounds_dir.rglob("*.wav"))
        if not files:
            raise FileNotFoundError(f"no .wav noise files under {sounds_dir}")
        _NOISE_CACHE[key] = files
    return _NOISE_CACHE[key]


def _active_rms(wav: np.ndarray, sr: int, *, floor_db: float = -40.0) -> float:
    """RMS over active-speech frames: 25 ms frames whose energy clears
    ``floor_db`` relative to the loudest frame. Falls back to global RMS if
    nothing clears (pure silence input)."""
    if len(wav) == 0:
        return 1e-12  # np.mean of empty is NaN, and NaN is truthy (wave-3 f.10)
    frame = max(1, int(sr * 0.025))
    n = (len(wav) // frame) * frame
    if n == 0:
        return float(np.sqrt(np.mean(np.square(wav)))) or 1e-12
    frames = wav[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return 1e-12
    active = rms[rms > peak * (10.0 ** (floor_db / 20.0))]
    if len(active) == 0:
        active = rms
    return float(np.sqrt(np.mean(np.square(active))))


def degrade_v2(
    wav: np.ndarray,
    sr: int,
    level: str,
    record_id: str,
    *,
    sounds_dir: Path = MUSAN_NOISE,
) -> tuple[np.ndarray, int]:
    """The v2 channel: noise (per-record, active-SNR-targeted) → telephony.

    Deterministic in ``(record_id, level)`` alone — no global RNG is touched,
    so callers cannot accidentally couple every record to one clip the way v1's
    seed plumbing did.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    if len(wav) == 0:
        raise ValueError("empty waveform — nothing to degrade (wave-3 finding 10)")

    out = wav.astype(np.float32, copy=True)
    if level in _SNR_DB:
        files = _noise_files(sounds_dir)
        h = _fnv1a(f"{record_id}:{level}")
        nf = files[h % len(files)]
        noise, nsr = sf.read(nf, dtype="float32")
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if nsr != sr:
            t = torch.from_numpy(noise).unsqueeze(0)
            noise = torchaudio.functional.resample(t, nsr, sr).numpy()[0]
        if len(noise) == 0:
            raise ValueError(f"empty noise file {nf}")
        # Deterministic start offset. The offset window is the clip's OWN
        # length (capped at the speech length), so even a clip tiled to exactly
        # the speech length keeps offset diversity — without this, short clips
        # would always start at 0 and two records sharing a short file would be
        # identical, the exact degeneracy v2 exists to remove.
        orig_len = len(noise)
        window = max(1, min(orig_len, len(out)))
        need = len(out) + window
        if len(noise) < need:
            noise = np.tile(noise, int(np.ceil(need / len(noise))))
        start = (h // max(1, len(files))) % window
        noise = noise[start:start + len(out)]

        target_db = _SNR_DB[level]
        s_rms = _active_rms(out, sr)
        n_rms = float(np.sqrt(np.mean(np.square(noise)))) or 1e-12
        gain = s_rms / (n_rms * (10.0 ** (target_db / 20.0)))
        out = out + gain * noise

    # telephony AFTER the acoustic mix — the physical order v1 inverted
    t = torch.from_numpy(out).unsqueeze(0)
    t8 = torchaudio.functional.resample(t, sr, 8000).clamp(-1, 1)
    mu = torchaudio.functional.mu_law_encoding(t8, 256)
    dec = torchaudio.functional.mu_law_decoding(mu, 256)
    out = dec.numpy()[0]
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = out / peak
    return out, 8000
