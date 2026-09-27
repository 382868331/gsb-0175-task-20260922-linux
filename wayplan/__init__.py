"""wayplan — cache way-partition planning with performance isolation.

Public interface (fixed contract, see README.md):

- :class:`AppTrace`          — validated (app_id, blocks) trace
- :func:`simulate_lru_misses`— exact cold or warm LRU miss count
- :func:`miss_curve`         — miss counts at every way count 1..max_ways
- :func:`plan_allocation`    — exact DP from one shared initial condition
- :func:`replay_isolated`    — warm/cold replay stats on isolated caches
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
    ReplayStats,
    miss_curve,
    plan_allocation,
    replay_isolated,
    simulate_lru_misses,
)
from .errors import InfeasibleBudgetError, ValidationError, WayPlanError

__all__ = [
    "MAX_APPS",
    "MAX_SETS",
    "MAX_WAYS",
    "AppTrace",
    "PlanResult",
    "ReplayStats",
    "miss_curve",
    "plan_allocation",
    "replay_isolated",
    "simulate_lru_misses",
    "WayPlanError",
    "ValidationError",
    "InfeasibleBudgetError",
]

__version__ = "1.0.0"
