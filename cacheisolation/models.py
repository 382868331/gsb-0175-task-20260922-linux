"""公开数据结构（全部为不可变/自描述的数据类）。

设计取舍：使用 ``@dataclass(frozen=True, slots=True)`` 保证规划结果
不会在批次调用中被意外就地修改；序列字段以 tuple 存储，哈希/等值可用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AppTrace:
    """单个应用的合成访存块序列。

    Attributes:
        app_id: 应用 ID，非负整数；在同一批输入中唯一且按其排序。
        blocks: 访存块 ID 的有序序列（块地址 = 非负整数）；允许空序列。
        weight: 目标函数中的正整数权重。
    """

    app_id: int
    blocks: tuple[int, ...]
    weight: int = 1

    @staticmethod
    def of(app_id: int, blocks: Sequence[int], weight: int = 1) -> "AppTrace":
        """便捷构造（把任意序列复制为 tuple）。"""
        return AppTrace(app_id, tuple(blocks), weight)


@dataclass(frozen=True, slots=True)
class SimResult:
    """单次 LRU 仿真结果。"""

    ways: int
    sets: int
    accesses: int
    misses: int
    hits: int

    @property
    def miss_rate(self) -> float:
        """缺失率 misses/accesses（IEEE-754 double）；空序列定义为 0.0。

        精确口径见 README：计数为精确整数，比率仅一次 double 除法，
        调用方不应以宽容差掩盖计数层的边界错误。
        """
        return 0.0 if self.accesses == 0 else self.misses / self.accesses


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    """路数分配规划的最优结果。

    Attributes:
        allocation: 按应用 ID 升序的 (app_id, ways) 元组。
        weighted_total: 目标值 ``sum(weight_i * miss_i(ways_i))``。
        budget: 请求的总路数上限（>= 使用路数）。
        ways_used: 实际使用路数 ``sum(ways_i)``。
        ties: 与最优解同值的其它分配向量（每个为按 app_id 排序的
            (app_id, ways) 元组），不含主解，仅用于审计/教学；可能为空。
        tie_count: 全部并列最优解个数（含主解）。
        iterations: DP 实际访问的（应用, 已用路数）状态计数。
        budget_states: 允许搜索使用的状态预算上限；``iterations <= budget_states``
            时为正常证明；相等且被截断才会抛 ``SearchBudgetExhausted``。
    """

    allocation: tuple[tuple[int, int], ...]
    weighted_total: int
    budget: int
    ways_used: int
    ties: tuple[tuple[tuple[int, int], ...], ...] = ()
    tie_count: int = 1
    iterations: int = 0
    budget_states: int = 0

    def as_dict(self) -> dict[int, int]:
        """{app_id: ways} 形式。"""
        return dict(self.allocation)


@dataclass(frozen=True, slots=True)
class ReplayAppStats:
    """隔离缓存中单个应用分片的重放统计。"""

    app_id: int
    ways: int
    sets: int
    weight: int
    accesses: int
    misses: int
    hits: int
    weighted_misses: int

    @property
    def miss_rate(self) -> float:
        """缺失率，口径同 :attr:`SimResult.miss_rate`。"""
        return 0.0 if self.accesses == 0 else self.misses / self.accesses


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """在隔离分区缓存上重放全部应用的统计汇总。"""

    sets: int
    per_app: tuple[ReplayAppStats, ...]
    total_accesses: int
    total_misses: int
    total_hits: int
    weighted_miss_total: int
    total_ways: int

    def as_plan(self) -> AllocationPlan:
        """还原成与 :meth:`optimize_allocation` 同构的分配结果（无并列信息）。"""
        alloc = tuple((s.app_id, s.ways) for s in self.per_app)
        return AllocationPlan(
            allocation=alloc,
            weighted_total=self.weighted_miss_total,
            budget=self.total_ways,
            ways_used=self.total_ways,
        )
