"""wayplan — cache way-partition planning with performance isolation.

Public interface (fixed contract, see README.md):

- :class:`AppTrace`          — validated (app_id, blocks) trace
- :func:`simulate_lru_misses`— exact LRU miss count for one configuration
- :func:`miss_curve`         — miss counts for every way count 1..max_ways
- :func:`plan_allocation`    — exact constrained DP way allocation with
  optional per-app hard miss caps and disable penalties -> PlanResult
- :func:`replay_isolated`    — replay stats on isolated caches -> ReplayStats
- :func:`verify_plan`        — independent replay re-checking every hard
  constraint of a plan -> :class:`VerificationReport`
- :class:`WayPlanError`, :class:`ValidationError`,
  :class:`InfeasibleBudgetError` — error taxonomy
- ``MAX_APPS`` / ``MAX_SETS`` / ``MAX_WAYS`` — interface bounds (6/16/12)
"""

from .core import (
    MAX_APPS,
    MAX_SETS,
    MAX_WAYS,
    AppTrace,
    ConstraintCheck,
    PlanResult,
    ReplayStats,
    VerificationReport,
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
    "ConstraintCheck",
    "PlanResult",
    "ReplayStats",
    "VerificationReport",
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
