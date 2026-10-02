"""The lab item instrument — typed loader for ``items/en_v1.yaml``.

The instrument re-renders the deployed assessment's 11 constructs as English
scenario questions with authored option semantics (the *answer bank*). The
scoring key is NOT authored here: every option's score is validated against the
pinned deployed scorer (:mod:`verdict_cert.scorer`) at load time, so the
instrument can never drift from the decision function the certificate reasons
about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .scorer import ITEMS, OPTIONS, SCORE_MAP

#: What the extractor may return: the scored options plus an abstention bucket.
EXTRACTION_OPTIONS: tuple[str, ...] = ("a", "b", "c", "unclear")

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "items" / "en_v1.yaml"


@dataclass(frozen=True)
class Option:
    key: str
    score: int
    meaning: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class FragileProbe:
    """A natural utterance whose *predicted* channel corruption maps to a
    different option than its clean reading (a controlled flip probe)."""

    utterance: str
    predicted_corruption: str
    clean_maps_to: str
    corrupted_maps_to: str


@dataclass(frozen=True)
class Item:
    id: str
    construct: str
    label: str
    question: str
    options: Mapping[str, Option]
    fragile_probe: FragileProbe
    mechanisms: tuple[str, ...]

    def clean_examples(self) -> list[tuple[str, str]]:
        """All (gold_option, utterance) pairs for this item's answer bank."""
        return [(o.key, ex) for o in self.options.values() for ex in o.examples]


def load_items(path: Path | str = DEFAULT_PATH, *, exact_examples: int | None = 2) -> list[Item]:
    """Load the instrument and validate it against the pinned scorer."""
    doc = yaml.safe_load(Path(path).read_text())
    items: list[Item] = []
    for raw in doc["items"]:
        pid = raw["id"]
        unknown = set(raw["options"]) - set(OPTIONS)
        if unknown:  # v1 silently DROPPED these (audit wave 2, finding 16)
            raise ValueError(f"{pid}: unknown option keys {sorted(unknown)}")
        options = {
            key: Option(key=key,
                        score=raw["options"][key]["score"],
                        meaning=raw["options"][key]["meaning"].strip(),
                        examples=tuple(raw["options"][key]["examples"]))
            for key in OPTIONS
        }
        for key, opt in options.items():
            pinned = SCORE_MAP[pid][key]
            if opt.score != pinned:
                raise ValueError(
                    f"{pid}.{key}: instrument score {opt.score} != pinned scorer {pinned}"
                )
        fp = raw["fragile_probe"]
        probe = FragileProbe(
            utterance=fp["utterance"],
            predicted_corruption=fp["predicted_corruption"],
            clean_maps_to=fp["clean_maps_to"],
            corrupted_maps_to=fp["corrupted_maps_to"],
        )
        items.append(
            Item(
                id=pid,
                construct=raw["construct"],
                label=raw["label"],
                question=raw["question"].strip(),
                options=options,
                fragile_probe=probe,
                mechanisms=tuple(raw["mechanisms"]),
            )
        )
    if tuple(i.id for i in items) != ITEMS:
        raise ValueError(f"instrument items {[i.id for i in items]} != scorer ITEMS")
    _validate_bank(items, exact_examples=exact_examples)
    return items


def _validate_bank(items: list[Item], *, exact_examples: int | None) -> None:
    """The answer-bank invariants v1 never enforced (audit wave 2, finding 16).

    Everything here was previously either test-only (`tests/test_items.py`) or
    entirely unchecked: wave 2 loaded banks with zero examples, duplicate
    utterances across options, a non-option ``clean_maps_to`` and a bogus
    option key without a complaint. The two-wordings invariant is load-bearing
    for the v1 re-ask draw and for the calibration arithmetic (66 = 11×3×2),
    so it is enforced at load time, parameterized for future banks.
    """
    seen: dict[str, str] = {}
    for it in items:
        if not it.question.strip():
            raise ValueError(f"{it.id}: empty question")
        for key, opt in it.options.items():
            where = f"{it.id}.{key}"
            if not opt.meaning:
                raise ValueError(f"{where}: empty meaning")
            if not opt.examples:
                raise ValueError(f"{where}: no example utterances")
            if exact_examples is not None and len(opt.examples) != exact_examples:
                raise ValueError(
                    f"{where}: {len(opt.examples)} examples, bank declares "
                    f"{exact_examples} wordings per (item, option)")
            for u in opt.examples:
                if not u.strip():
                    raise ValueError(f"{where}: empty example utterance")
                if u in seen:
                    raise ValueError(
                        f"utterance {u!r} appears under both {seen[u]} and {where} "
                        "— gold would be ambiguous")
                seen[u] = where
    # Second pass: probes, against the COMPLETE bank and each other — the v1
    # single pass never added probes to `seen`, so probe-vs-probe and
    # probe-vs-later-item collisions loaded silently (wave-3 finding 2).
    for it in items:
        fp = it.fragile_probe
        if fp.clean_maps_to not in OPTIONS:
            raise ValueError(f"{it.id}: probe clean_maps_to {fp.clean_maps_to!r} not an option")
        if fp.corrupted_maps_to not in EXTRACTION_OPTIONS:
            raise ValueError(
                f"{it.id}: probe corrupted_maps_to {fp.corrupted_maps_to!r} "
                f"not in {EXTRACTION_OPTIONS}")
        if fp.clean_maps_to == fp.corrupted_maps_to:
            raise ValueError(f"{it.id}: probe does not flip (clean == corrupted)")
        if fp.utterance in seen:
            raise ValueError(
                f"{it.id}: probe utterance {fp.utterance!r} collides with "
                f"{seen[fp.utterance]}")
        seen[fp.utterance] = f"{it.id}.probe"
