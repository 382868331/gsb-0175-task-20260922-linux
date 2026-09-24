"""wayplan — cache way-partition planning with performance isolation.

Public interface (fixed contract, see README.md):

- :class:`AppTrace`          — validated (app_id, blocks) trace
- :func:`simulate_lru_misses`— exact LRU miss count for one configuration
- :func:`miss_curve`         — miss counts for every way count 1..max_ways
- :func:`plan_allocation`    — exact DP way allocation -> :class:`PlanResult`
- :func:`replay_isolated`    — replay stats on isolated caches -> ReplayStats
- :class:`TaskSpec`          — one task's known staged loss curves
- :func:`plan_stages`        — exact DP over a known stage sequence with
                               re-routing migration costs -> StagePlanResult
- :class:`WayPlanError`, :class:`ValidationError`,
  :class:`InfeasibleBudgetError` — error taxonomy
- ``MAX_APPS`` / ``MAX_SETS`` / ``MAX_WAYS`` — allocation bounds (6/16/12)
- ``MAX_TASKS`` / ``MAX_STAGES`` / ``MAX_DYN_WAYS`` — staged-planning
  bounds (3/4/8)
"""

from .core import (
    MAX_APPS,
    MAX_DYN_WAYS,
    MAX_SETS,
    MAX_STAGES,
    MAX_TASKS,
    MAX_WAYS,
    AppTrace,
    PlanResult,
    ReplayStats,
    StagePlanResult,
    TaskSpec,
    miss_curve,
    plan_allocation,
    plan_stages,
    replay_isolated,
    simulate_lru_misses,
)
from .errors import InfeasibleBudgetError, ValidationError, WayPlanError

__all__ = [
    "MAX_APPS",
    "MAX_SETS",
    "MAX_WAYS",
    "MAX_TASKS",
    "MAX_STAGES",
    "MAX_DYN_WAYS",
    "AppTrace",
    "PlanResult",
    "ReplayStats",
    "TaskSpec",
    "StagePlanResult",
    "miss_curve",
    "plan_allocation",
    "plan_stages",
    "replay_isolated",
    "simulate_lru_misses",
    "WayPlanError",
    "ValidationError",
    "InfeasibleBudgetError",
]

__version__ = "1.1.0"
