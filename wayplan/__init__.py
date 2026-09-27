"""wayplan — cache way-partition planning with performance isolation.

Public interface (fixed contract, see README.md):

- :class:`AppTrace`          — validated (app_id, blocks) trace
- :func:`simulate_lru_misses`— exact LRU miss count for one configuration
- :func:`miss_curve`         — miss counts for every way count 1..max_ways
- :func:`plan_allocation`    — exact DP way allocation -> :class:`PlanResult`
- :func:`replay_isolated`    — replay stats on isolated caches -> ReplayStats
- :class:`WayPlanError`, :class:`ValidationError`,
  :class:`InfeasibleBudgetError` — error taxonomy
- ``MAX_APPS`` / ``MAX_SETS`` / ``MAX_WAYS`` — interface bounds (6/16/12)

The three simulation entry points and the planner accept an optional
``initial_state`` (one row per set, blocks ordered LRU -> MRU).  Every
candidate run starts from the same normalised, immutable copy of it; rows
longer than the candidate way count are shrunk LRU-first, so candidates
never share or inherit cache contents.
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
