"""Set-associative LRU 缓存仿真（核心算法，从零实现，仅用标准库容器）。

模型（本题契约的有限子集）：

* 缓存分为 ``sets`` 个组，每组 ``ways`` 路，组间完全独立；
* 块地址 ``b`` 映射到组 ``b % sets``，整地址作为 tag（``b1 != b2``
  且同组时即发生争用）；
* 每组严格 LRU：命中移到 MRU 端，缺失且组满时淘汰 LRU 端；
* 冷启动，组间互不影响；不建模预取、写策略、包含性/一致性。

实现使用 ``collections.OrderedDict`` 作为每组的 LRU 链表载体，
LRU 的命中移动/淘汰判定逻辑在本模块内手写，未使用任何第三方缓存实现。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

from .models import SimResult
from .validation import MAX_SETS, MAX_TOTAL_WAYS, check_block_id, check_int


def simulate_lru(blocks: Sequence[int], sets: int, ways: int) -> SimResult:
    """仿真一条访存序列在 ``sets`` 组 ``ways`` 路 LRU 缓存上的命中/缺失。

    Args:
        blocks: 块 ID（非负整数）的有序序列，可为空。
        sets: 缓存组数，``1 <= sets <= 16``。
        ways: 每组路数，``1 <= ways <= 12``。

    Returns:
        :class:`~cacheisolation.models.SimResult`；空序列时三项计数均为 0。

    Raises:
        InvalidInputError: 类型/取值非法（含 bool 冒充整数），错误可定位。
    """
    check_int(sets, "sets", minimum=1, maximum=MAX_SETS)
    check_int(ways, "ways", minimum=1, maximum=MAX_TOTAL_WAYS)
    if not isinstance(blocks, (list, tuple)):
        from .errors import InvalidInputError

        raise InvalidInputError(
            f"blocks: 必须是 list/tuple 序列，得到 {type(blocks).__name__}"
        )
    # 先完成全部校验再仿真：无共享状态，失败不会产生部分结果。
    norm = tuple(check_block_id(b, f"blocks[{i}]") for i, b in enumerate(blocks))
    return _simulate(norm, sets, ways)


def _simulate(blocks: tuple[int, ...], sets: int, ways: int) -> SimResult:
    """内部入口：blocks 已校验，sets/ways 已校验。"""
    cache: list[OrderedDict[int, int]] = [OrderedDict() for _ in range(sets)]
    misses = 0
    hits = 0
    for b in blocks:
        row = cache[b % sets]
        if b in row:
            row.move_to_end(b)  # 命中：刷新为 MRU
            hits += 1
        else:
            misses += 1
            row[b] = 1  # 装入（tag 即整地址）
            if len(row) > ways:
                row.popitem(last=False)  # 组满：淘汰 LRU
    return SimResult(
        ways=ways,
        sets=sets,
        accesses=hits + misses,
        misses=misses,
        hits=hits,
    )
