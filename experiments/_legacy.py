"""QUARANTINE — the legacy set builder, removed from the library (F-3, 2026-08-16).

This module exists so the committed v1 harnesses stay byte-reproducible after
`build_set` was deleted from ``verdict_cert.certificate``. It is deliberately
OUTSIDE ``src/`` — it is not part of the package, it is not importable as
``verdict_cert.*``, and nothing in the v2 instrument may import it.

Why it was removed from the library (architecture finding F-3, measured by B1,
ledger E-106..E-108): the empty→argmin fallback converts "no option is
plausible" into a confident singleton — the exact shape that certifies a wrong
verdict. Measured: **no feasible operating point at a 0% risk tolerance in any
of 30 arm×seed cells on either pool** (against 30/30 for the normative
``calibration.conformal_set``); it releases wrong verdicts the normative
builder withholds while withholding none in return; and its release rate barely
responds to the threshold, so the α machinery is a no-op under it.

The verbatim body below is byte-for-byte the deleted ``certificate.build_set``
(minus the deprecation block). ``tests/test_hazard_ratchets.py`` enforces that
this file is the ONLY definition and freezes its importer list.
"""

from __future__ import annotations

from typing import Mapping


def build_set(option_scores: Mapping[str, float], threshold: float) -> frozenset[str]:
    """Conformal prediction set: every option whose nonconformity score ≤ threshold.

    Guarantees non-empty (falls back to the single best option) so downstream
    subset-sum never sees an empty item. ``option_scores`` maps each of a/b/c to
    its nonconformity score (lower = more plausible).
    """
    chosen = frozenset(o for o, s in option_scores.items() if s <= threshold)
    if chosen:
        return chosen
    return frozenset({min(option_scores, key=option_scores.get)})
