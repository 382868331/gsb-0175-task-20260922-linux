"""wayplan — cache way-partition planning with performance isolation.

Public interface (fixed contract, see README.md):

- :class:`AppTrace`          — validated (app_id, blocks) trace
- :func:`simulate_lru_misses`— exact LRU miss count for one configuration
- :func:`miss_curve`         — miss counts for every way count 1..max_ways
- :func:`plan_allocation`    — exact DP way allocation -> :class:`PlanResult`
  (optional per-app hard miss caps and paid deactivation since v1.1)
- :func:`replay_isolated`    — replay stats on isolated caches -> ReplayStats
- :func:`verify_plan`        — replay-based re-check of every hard
  constraint of a PlanResult -> :class:`PlanVerification`
- :class:`WayPlanError`, :class:`ValidationError`,
  :class:`InfeasibleBudgetError` — error taxonomy
- ``MAX_APPS`` / ``MAX_SETS`` / ``MAX_WAYS`` — interface bounds (6/16/12)
"""

from .core import (
    MAX_APPS,
    MAX_SETS,
    MAX_WAYS,
    AppTrace,
    PlanResult,
    PlanVerification,
    ReplayStats,
    miss_curve,
    plan_allocation,
    replay_isolated,
    simulate_lru_misses,
    verify_plan,
)
from .errors import InfeasibleBudgetError, ValidationError, WayPlanError

__all__ = [
    "MAX_APPS",
    "MAX_SETS",
    "MAX_WAYS",
    "AppTrace",
    "PlanResult",
    "PlanVerification",
    "ReplayStats",
    "miss_curve",
    "plan_allocation",
    "replay_isolated",
    "simulate_lru_misses",
    "verify_plan",
    "WayPlanError",
    "ValidationError",
    "InfeasibleBudgetError",
]

__version__ = "1.1.0"
