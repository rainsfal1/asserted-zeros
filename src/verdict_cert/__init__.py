"""verdict-cert — runtime verdict-referenced fidelity certification (M0 core)."""

from .certificate import (
    Certificate,
    UncertaintySets,
    calibrate_threshold,
    certify,
    full_sets,
    reachable_verdicts,
    reask_ranking,
    singleton_sets,
    verdict_distribution,
)
from .scorer import (
    ITEMS,
    OPTIONS,
    PSYCHOMETRIC_MAX,
    SCORE_MAP,
    VERDICTS,
    placement_decision,
    total_score,
)

__version__ = "0.1.0"

__all__ = [
    "Certificate",
    "UncertaintySets",
    "calibrate_threshold",
    "certify",
    "full_sets",
    "reachable_verdicts",
    "reask_ranking",
    "singleton_sets",
    "verdict_distribution",
    "ITEMS",
    "OPTIONS",
    "PSYCHOMETRIC_MAX",
    "SCORE_MAP",
    "VERDICTS",
    "placement_decision",
    "total_score",
]
