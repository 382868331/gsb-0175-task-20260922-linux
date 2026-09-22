"""独立参考实现（测试专用）。

重要：本模块**不导入 cacheisolation**，不调用任何被测核心代码来产生
期望值。LRU 用与核心不同的数据结构（每组一个普通 list，MRU 在尾部）
独立书写；最优分配用 ``itertools.product`` 对所有可行路数向量做完全
暴力枚举。它只作为测试预言机（oracle）。
"""

from __future__ import annotations

from itertools import product
from typing import Sequence


def reference_lru_misses(blocks: Sequence[int], sets: int, ways: int) -> int:
    """独立 LRU 仿真：每组一个 list，LRU 在头部、MRU 在尾部。

    与被测实现的算法等价但代码路径完全独立（无 OrderedDict、
    无 cacheisolation 导入），返回 miss 总数。
    """
    rows: list[list[int]] = [[] for _ in range(sets)]
    misses = 0
    for b in blocks:
        row = rows[b % sets]
        if b in row:
            row.remove(b)      # 命中：摘除旧位置
            row.append(b)      # 放到 MRU 尾
        else:
            misses += 1
            row.append(b)      # 缺失：装入 MRU 尾
            if len(row) > ways:
                row.pop(0)     # 组满：淘汰 LRU 头
    return misses


def reference_lru_trace(blocks: Sequence[int], sets: int, ways: int):
    """逐访问参考：返回 (miss 事件 bool 列表, 最终 miss 数, 最终 hit 数)。

    供手工例逐步核对（与 :func:`reference_lru_misses` 同一实现，
    额外保留逐次事件）。
    """
    rows: list[list[int]] = [[] for _ in range(sets)]
    events: list[bool] = []
    for b in blocks:
        row = rows[b % sets]
        if b in row:
            row.remove(b)
            row.append(b)
            events.append(False)  # hit
        else:
            events.append(True)   # miss
            row.append(b)
            if len(row) > ways:
                row.pop(0)
    misses = sum(events)
    return events, misses, len(blocks) - misses


def brute_force_optimal(
    curves: Sequence[Sequence[int]],
    weights: Sequence[int],
    budget: int,
) -> tuple[int, list[tuple[int, ...]]]:
    """暴力枚举所有 k_i >= 1、sum k_i <= budget 的路数向量。

    Args:
        curves: curves[i][k-1] = 应用 i 在 k 路下的 miss 数。
        weights: 各应用正整数权重。
        budget: 总路数上限。

    Returns:
        (最小加权 miss, 全部最优向量按字典序升序)；无可行解时
        向量列表为空（与“搜索未完成”无关，枚举是完全的）。
    """
    n = len(curves)
    choices = [range(1, len(c) + 1) for c in curves]
    best: int | None = None
    sols: list[tuple[int, ...]] = []
    for vec in product(*choices):
        used = sum(vec)
        if used > budget:
            continue
        val = sum(weights[i] * curves[i][vec[i] - 1] for i in range(n))
        if best is None or val < best:
            best = val
            sols = [vec]
        elif val == best:
            sols.append(vec)
    if best is None:
        return 0, []
    return best, sorted(sols)
