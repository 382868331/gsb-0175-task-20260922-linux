"""可分配缓存路数的性能隔离规划（离线、纯标准库）。

公开接口见 ``__all__``。模块只实现本题契约声明的有限数学/仿真子集，
不宣称兼容任何完整行业标准缓存模型。
"""

from .errors import (
    CacheIsolationError,
    InvalidInputError,
    InfeasibleAllocationError,
    SearchBudgetExhausted,
)
from .models import (
    AppTrace,
    AllocationPlan,
    SimResult,
    ReplayAppStats,
    ReplayReport,
)
from .curves import MissCurve, build_miss_curves
from .lru import simulate_lru
from .planner import optimize_allocation
from .replay import replay_partition
from .pipeline import plan_isolation, IsolationPlanResult
from .baselines import greedy_marginal_allocation, GreedyResult
from . import samples

__all__ = [
    "AppTrace",
    "MissCurve",
    "AllocationPlan",
    "SimResult",
    "ReplayAppStats",
    "ReplayReport",
    "IsolationPlanResult",
    "simulate_lru",
    "build_miss_curves",
    "optimize_allocation",
    "plan_isolation",
    "replay_partition",
    "greedy_marginal_allocation",
    "GreedyResult",
    "samples",
    "CacheIsolationError",
    "InvalidInputError",
    "InfeasibleAllocationError",
    "SearchBudgetExhausted",
]

__version__ = "1.0.0"
