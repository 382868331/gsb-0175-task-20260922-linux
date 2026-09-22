"""贪心边际收益基线（仅作对照，**不是**核心算法）。

本模块刻意实现一个朴素的“逐路分给当前边际收益最大的应用”的贪心过程，
用于在演示/测试中制造并展示反例：miss 曲线非凸时，贪心会被局部
边际收益误导而得到次优分配。核心规划器 :mod:`cacheisolation.planner`
使用精确 DP，不调用本模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .curves import MissCurve


@dataclass(frozen=True, slots=True)
class GreedyResult:
    """贪心基线结果。"""

    allocation: tuple[tuple[int, int], ...]
    weighted_total: int
    ways_used: int


def greedy_marginal_allocation(
    curves: Sequence[MissCurve],
    budget: int,
) -> GreedyResult:
    """朴素边际贪心：每应用先分 1 路，随后逐路分给“再加一路的加权

    miss 下降量”最大的应用；并列时给 app_id 较小者；当所有剩余边际
    收益都 <= 0 时停止（不会把路数浪费在无收益处）。

    这是被演示的**次优基线**：曲线非凸时它无法跨越“先亏后赚”的台阶，
    结果可能显著劣于精确 DP 最优解。
    """
    n = len(curves)
    ways = [1] * n
    used = n
    while used < budget:
        best_i = -1
        best_gain = 0  # 严格正收益才继续
        for i, c in enumerate(curves):
            k = ways[i]
            if k >= c.max_ways:
                continue
            gain = c.weight * (c.at(k) - c.at(k + 1))
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break  # 没有任何正收益的多余路可分
        ways[best_i] += 1
        used += 1

    allocation = tuple((c.app_id, ways[i]) for i, c in enumerate(curves))
    total = sum(c.weight * c.at(ways[i]) for i, c in enumerate(curves))
    return GreedyResult(allocation=allocation, weighted_total=total, ways_used=used)
