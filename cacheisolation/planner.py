"""精确动态规划：可分配路数的性能隔离规划。

问题（本题契约）：

    给定 n 个应用的 miss 曲线 c_i(k)（k=1..max_ways）与正整数权重 w_i，
    求整数向量 (k_1,...,k_n)，k_i >= 1，满足 sum(k_i) <= budget，
    最小化  sum_i w_i * c_i(k_i)。

关键约束：**不假定 c_i 凸**（边际收益可以非单调），**不用贪心边际收益
替代最优**。这里实现对每应用可行路数完全枚举的
O(n * budget * max_ways) 动态规划，并通过父指针枚举全部并列最优向量；
并列时按应用 ID 升序的分配向量字典序选主解。

失败语义严格分开（两条不同的代码路径，异常类型不同）：

* 数学上直接可证无解（``budget < n``，每应用至少一路），或 DP 完整表
  跑完无可达终点 -> :class:`InfeasibleAllocationError`（**已证明**无解）；
* 调用方显式给定有限 ``state_budget``，DP 在搜索途中超出预算而中断 ->
  :class:`SearchBudgetExhausted`（**未证明**任何结论）。默认预算等于
  完整状态-转移表的上界，在契约上限（n<=6, budget<=12）内必能跑完，
  因此正常路径只会得到最优解或不可行证明。

纯整数运算，最优值与计数无浮点误差。
"""

from __future__ import annotations

from typing import Sequence

from .curves import MissCurve
from .errors import InfeasibleAllocationError, InvalidInputError, SearchBudgetExhausted
from .models import AllocationPlan
from .validation import MAX_APPS, MAX_TOTAL_WAYS, check_int


def _full_transition_budget(n: int, budget: int, max_ways: int) -> int:
    """完整 DP 表的状态-转移计数上界。

    共 n 层；每层至多遍历 budget+1 个前驱已用量，每个前驱至多尝试
    max_ways 个候选路数。实际可达前驱更少，故真实工作量 <= 此上界。
    """
    return n * (budget + 1) * max_ways


def _run_dp(
    curves: Sequence[MissCurve],
    budget: int,
    state_budget: int | None,
) -> tuple[int, list[list[frozenset[int] | None]], list[int | None], int]:
    """执行精确 DP。

    返回 (层数 n 之后的 parents, 最终 dp, 已执行转移数)。
    搜索途中超出 state_budget 时抛 :class:`SearchBudgetExhausted`。
    """
    dp: list[int | None] = [0] + [None] * budget
    parents: list[list[frozenset[int] | None]] = [
        [frozenset({0})] + [None] * budget
    ]
    work = 0
    for curve in curves:
        ndp: list[int | None] = [None] * (budget + 1)
        nparents: list[frozenset[int] | None] = [None] * (budget + 1)
        max_k = min(curve.max_ways, budget)
        for t_prev in range(budget + 1):
            base = dp[t_prev]
            if base is None:
                continue
            upper = min(max_k, budget - t_prev)
            for k in range(1, upper + 1):
                work += 1
                if state_budget is not None and work > state_budget:
                    # 真实中断：DP 尚未完成，既未证明最优也未证明无解。
                    raise SearchBudgetExhausted(
                        f"搜索预算 {state_budget} 次转移已耗尽（搜索在第 "
                        f"{len(parents)} 个应用处中断，已执行 {work} 次转移）；"
                        f"这不代表无解，放宽预算后重试"
                    )
                t = t_prev + k
                val = base + curve.weight * curve.at(k)
                cur = ndp[t]
                if cur is None or val < cur:
                    ndp[t] = val
                    nparents[t] = frozenset({k})
                elif val == cur:
                    assert nparents[t] is not None
                    nparents[t] = nparents[t] | {k}
        dp = ndp
        parents.append(nparents)
    return len(curves), parents, dp, work


def _backtrack(
    parents: Sequence[Sequence[frozenset[int] | None]],
    end_ts: Sequence[int],
    n: int,
) -> list[tuple[int, ...]]:
    """沿并列父指针枚举所有达到任一最优终点用量的完整分配向量。"""
    vectors: list[tuple[int, ...]] = []

    def rec(i: int, t: int, suffix: tuple[int, ...]) -> None:
        if i == 0:
            vectors.append(tuple(reversed(suffix)))
            return
        choices = parents[i][t]
        assert choices is not None
        for k in sorted(choices):
            rec(i - 1, t - k, suffix + (k,))

    for t in end_ts:
        rec(n, t, ())
    return vectors


def optimize_allocation(
    curves: Sequence[MissCurve],
    budget: int,
    *,
    state_budget: int | None = None,
) -> AllocationPlan:
    """在给定逐路 miss 曲线上求最优路数分配（精确 DP，非贪心）。

    Args:
        curves: 每个应用一条 :class:`MissCurve`（1..6 条，app_id 唯一且升序）。
        budget: 总路数预算，``1 <= budget <= 12``。``budget < 应用数`` 时
            直接给出不可行证明。
        state_budget: 可选的搜索工作量上限（DP 转移计数）。``None``（默认）
            表示使用覆盖完整 DP 表的上界，契约规模内必能证明；显式传入
            过小正整数可在搜索途中触发 :class:`SearchBudgetExhausted`
            （未证明无解，与不可行证明严格区分）。

    Returns:
        :class:`AllocationPlan`：``allocation`` 按 app_id 升序；并列最优时
        主解取分配向量字典序最小者，其余并列向量见 ``ties``。

    Raises:
        InvalidInputError: 曲线/预算类型或取值非法（含 bool 冒充整数）。
        InfeasibleAllocationError: 已证明不存在每应用至少一路且不超预算的分配。
        SearchBudgetExhausted: 搜索预算在证明前耗尽（不代表无解）。
    """
    if not isinstance(curves, (list, tuple)):
        raise InvalidInputError(
            f"curves: 必须是 list/tuple，得到 {type(curves).__name__}"
        )
    if len(curves) < 1 or len(curves) > MAX_APPS:
        raise InvalidInputError(
            f"curves: 应用数必须在 1..{MAX_APPS}，得到 {len(curves)}"
        )
    check_int(budget, "budget", minimum=1, maximum=MAX_TOTAL_WAYS)
    n = len(curves)

    ids: list[int] = []
    for i, c in enumerate(curves):
        if not isinstance(c, MissCurve):
            raise InvalidInputError(
                f"curves[{i}]: 必须是 MissCurve，得到 {type(c).__name__}"
            )
        ids.append(c.app_id)
        if c.max_ways < 1:
            raise InvalidInputError(f"curves[{i}]: miss 曲线为空")
        if any(isinstance(m, bool) or not isinstance(m, int) for m in c.misses):
            raise InvalidInputError(f"curves[{i}].misses: 含非整数项")
        if any(m < 0 for m in c.misses):
            raise InvalidInputError(f"curves[{i}].misses: miss 数不能为负")
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise InvalidInputError(
            f"curves: app_id 必须唯一且按升序排列，得到 {ids}"
        )
    if state_budget is not None:
        check_int(state_budget, "state_budget", minimum=1)

    # 数学直接证明：每应用至少 1 路 => 至少需要 n 路。无需搜索。
    if budget < n:
        raise InfeasibleAllocationError(
            f"预算 {budget} < 应用数 {n}：每应用至少 1 路，已证明无可行分配"
        )

    max_ways = max(c.max_ways for c in curves)
    full_budget = _full_transition_budget(n, budget, max_ways)
    effective_budget = full_budget if state_budget is None else state_budget

    _n, parents, dp, work = _run_dp(curves, budget, effective_budget)

    best: int | None = None
    best_t: list[int] = []
    for t in range(n, budget + 1):
        if dp[t] is None:
            continue
        if best is None or dp[t] < best:
            best = dp[t]
            best_t = [t]
        elif dp[t] == best:
            best_t.append(t)
    if best is None:
        # DP 完整跑完（默认预算下必然如此）仍无可达终点 => 证明无解。
        raise InfeasibleAllocationError(
            f"budget={budget}, n={n}：DP 完整表无可达终点，已证明无可行分配"
        )

    vectors = _backtrack(parents, best_t, n)
    vectors.sort()  # 并列裁决：按应用 ID 升序的分配向量字典序
    primary = vectors[0]
    allocation = tuple((curves[i].app_id, k) for i, k in enumerate(primary))
    ties = tuple(
        tuple((curves[i].app_id, k) for i, k in enumerate(v))
        for v in vectors[1:]
    )
    return AllocationPlan(
        allocation=allocation,
        weighted_total=best,
        budget=budget,
        ways_used=sum(primary),
        ties=ties,
        tie_count=len(vectors),
        iterations=work,
        budget_states=effective_budget,
    )
