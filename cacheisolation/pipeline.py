"""端到端流水线：原始序列 -> miss 曲线 -> 精确 DP 最优分配 -> 隔离重放。

一次调用完成题面要求的三个阶段，各阶段结果都保留以便审计；
任何阶段的输入非法都会在产生后续计算前抛出（无可观察的部分变更）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .curves import MissCurve, build_miss_curves
from .models import AllocationPlan, AppTrace, ReplayReport
from .planner import optimize_allocation
from .replay import replay_partition
from .validation import MAX_SETS, MAX_TOTAL_WAYS, check_int


@dataclass(frozen=True, slots=True)
class IsolationPlanResult:
    """三阶段完整结果。"""

    curves: tuple[MissCurve, ...]
    plan: AllocationPlan
    replay: ReplayReport


def plan_isolation(
    apps: Sequence[AppTrace],
    sets: int,
    budget: int,
    *,
    max_ways: int = MAX_TOTAL_WAYS,
    state_budget: int | None = None,
) -> IsolationPlanResult:
    """对一批应用完成曲线构建、最优分配与隔离重放。

    Args:
        apps: 1..6 个 :class:`AppTrace`。
        sets: 缓存组数 1..16。
        budget: 总路数预算 1..12（< 应用数时抛不可行证明）。
        max_ways: 曲线模拟到的最大路数（1..12）。
        state_budget: 透传给 :func:`optimize_allocation` 的可选搜索预算。

    Returns:
        :class:`IsolationPlanResult`。``replay.weighted_miss_total`` 与
        ``plan.weighted_total`` 必然相等（隔离分片无干扰，且曲线与重放
        使用同一 LRU 模型）。
    """
    check_int(sets, "sets", minimum=1, maximum=MAX_SETS)
    check_int(budget, "budget", minimum=1, maximum=MAX_TOTAL_WAYS)
    check_int(max_ways, "max_ways", minimum=1, maximum=MAX_TOTAL_WAYS)
    curves = build_miss_curves(apps, sets, max_ways)
    plan = optimize_allocation(curves, budget, state_budget=state_budget)
    replay = replay_partition(apps, plan.as_dict(), sets)
    # 一致性自检：纯整数等式，无容差。
    if replay.weighted_miss_total != plan.weighted_total:
        raise AssertionError(
            "内部一致性失败：重放加权 miss "
            f"{replay.weighted_miss_total} != 规划值 {plan.weighted_total}"
        )
    return IsolationPlanResult(curves=curves, plan=plan, replay=replay)
