"""逐路 miss 曲线构建。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .errors import InvalidInputError
from .lru import simulate_lru
from .validation import (
    MAX_APPS,
    MAX_SETS,
    MAX_TOTAL_WAYS,
    check_block_id,
    check_int,
    check_positive_integer_weight,
)


@dataclass(frozen=True, slots=True)
class MissCurve:
    """单个应用在固定组数下的逐路 miss 曲线。

    ``misses[k-1]`` 为分配 ``k`` 路（``1 <= k <= max_ways``）时的 miss 数。
    曲线对路数单调不增，但**不保证凸性**（边际收益可能非单调）。
    """

    app_id: int
    sets: int
    weight: int
    misses: tuple[int, ...]

    def at(self, ways: int) -> int:
        if not 1 <= ways <= len(self.misses):
            raise IndexError(f"ways={ways} 超出曲线范围 [1, {len(self.misses)}]")
        return self.misses[ways - 1]

    @property
    def max_ways(self) -> int:
        return len(self.misses)


def validate_app_trace(app: object, index: int) -> "tuple[int, tuple[int, ...], int]":
    """校验并规范化一个 AppTrace（按列表索引定位错误）。"""
    from .models import AppTrace

    path = f"apps[{index}]"
    if not isinstance(app, AppTrace):
        raise InvalidInputError(
            f"{path}: 必须是 AppTrace，得到 {type(app).__name__}"
        )
    app_id = check_int(app.app_id, f"{path}.app_id", minimum=0)
    weight = check_positive_integer_weight(app.weight, f"{path}.weight")
    blocks = app.blocks
    if not isinstance(blocks, (list, tuple)):
        raise InvalidInputError(
            f"{path}.blocks: 必须是 list/tuple，得到 {type(blocks).__name__}"
        )
    norm_blocks = tuple(
        check_block_id(b, f"{path}.blocks[{i}]") for i, b in enumerate(blocks)
    )
    return app_id, norm_blocks, weight


def normalize_apps(apps: Sequence[object]) -> "list[tuple[int, tuple[int, ...], int]]":
    """校验应用列表：数量上限、ID 唯一，返回按 app_id 升序的规范化列表。"""
    if not isinstance(apps, (list, tuple)):
        raise InvalidInputError(
            f"apps: 必须是 list/tuple，得到 {type(apps).__name__}"
        )
    if len(apps) < 1:
        raise InvalidInputError("apps: 至少需要 1 个应用")
    if len(apps) > MAX_APPS:
        raise InvalidInputError(f"apps: 应用数 {len(apps)} 超过上限 {MAX_APPS}")
    normalized = [validate_app_trace(a, i) for i, a in enumerate(apps)]
    ids = [a[0] for a in normalized]
    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise InvalidInputError(f"apps: app_id 重复: {dupes}")
    normalized.sort(key=lambda t: t[0])
    return normalized


def build_miss_curves(
    apps: Sequence[object],
    sets: int,
    max_ways: int = MAX_TOTAL_WAYS,
) -> tuple[MissCurve, ...]:
    """分别模拟每个应用在每种路数 1..max_ways 下的 LRU miss 数。

    Args:
        apps: 1..6 个 :class:`~cacheisolation.models.AppTrace`（ID 须唯一）。
        sets: 缓存组数，1..16。
        max_ways: 曲线最大路数，1..12（建议 >= 应用数以便后续规划可行）。

    Returns:
        按 app_id 升序排列的 :class:`MissCurve` 元组。

    Raises:
        InvalidInputError: 任一输入非法（在任何仿真开始前完成结构校验，
            本函数无外部可观察状态，不存在部分变更）。
    """
    check_int(sets, "sets", minimum=1, maximum=MAX_SETS)
    check_int(max_ways, "max_ways", minimum=1, maximum=MAX_TOTAL_WAYS)
    normalized = normalize_apps(apps)
    curves: list[MissCurve] = []
    for app_id, blocks, weight in normalized:
        misses = tuple(
            simulate_lru(blocks, sets, w).misses for w in range(1, max_ways + 1)
        )
        curves.append(MissCurve(app_id=app_id, sets=sets, weight=weight, misses=misses))
    return tuple(curves)
