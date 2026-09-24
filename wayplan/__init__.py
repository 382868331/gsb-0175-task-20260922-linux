"""wayplan — cache way-partition planning with performance isolation.

Public interface (fixed contract, see README.md):

- :class:`AppTrace`          — validated (app_id, blocks) trace
- :func:`simulate_lru_misses`— exact LRU miss count for one configuration
- :func:`miss_curve`         — miss counts for every way count 1..max_ways
- :func:`plan_allocation`    — exact DP way allocation -> :class:`PlanResult`
- :func:`replay_isolated`    — replay stats on isolated caches -> ReplayStats
- :class:`Task` / :class:`Stage`           — multi-stage planner inputs
- :func:`plan_task_stages`   — exact DP across stages with migration cost
                               and per-boundary max change -> MultiStagePlanResult
- :class:`WayPlanError`, :class:`ValidationError`,
  :class:`InfeasibleBudgetError`, :class:`InfeasibleStageError`
                               — error taxonomy
- ``MAX_APPS`` / ``MAX_SETS`` / ``MAX_WAYS`` — old bounds (6/16/12)
- ``MAX_TASKS`` / ``MAX_STAGES`` / ``MAX_TASK_WAYS`` — multi-stage
                               bounds (3/4/8)
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
from .errors import (
    InfeasibleBudgetError,
    InfeasibleStageError,
    ValidationError,
    WayPlanError,
)
from .stages import (
    MAX_STAGES,
    MAX_TASKS,
    MAX_TASK_WAYS,
    MultiStagePlanResult,
    Stage,
    Task,
    TaskPlan,
    plan_task_stages,
)

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
    "MAX_TASKS",
    "MAX_STAGES",
    "MAX_TASK_WAYS",
    "Stage",
    "Task",
    "MultiStagePlanResult",
    "TaskPlan",
    "plan_task_stages",
    "WayPlanError",
    "ValidationError",
    "InfeasibleBudgetError",
    "InfeasibleStageError",
]

__version__ = "1.1.0"
