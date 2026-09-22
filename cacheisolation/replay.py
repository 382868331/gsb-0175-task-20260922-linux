"""在按最优分配切分的隔离缓存上重放全部应用。

隔离语义：每个应用独占 ``sets`` 组、每组 ``ways_i`` 路的私有 LRU 分片，
分片之间无争用、无替换干扰。因此重放得到的每应用 miss 数必须与其
miss 曲线上 ``ways_i`` 对应的值一致（本库据此提供一致性自检）。
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .curves import normalize_apps
from .errors import InvalidInputError
from .lru import _simulate
from .models import AppTrace, ReplayAppStats, ReplayReport
from .validation import MAX_APPS, MAX_SETS, MAX_TOTAL_WAYS, check_int


def replay_partition(
    apps: Sequence[AppTrace],
    allocation: Mapping[int, int] | Sequence[tuple[int, int]],
    sets: int,
) -> ReplayReport:
    """按给定路数分配在隔离分区缓存上重放各应用。

    Args:
        apps: 1..6 个 :class:`AppTrace`。
        allocation: ``{app_id: ways}`` 映射，或 ``(app_id, ways)`` 序列；
            必须为每个应用恰好分配一次，ways 为 1..12 的正整数。
        sets: 缓存组数，1..16。

    Returns:
        :class:`ReplayReport`，``per_app`` 按 app_id 升序。

    Raises:
        InvalidInputError: 输入非法；分配与应用集合不匹配（缺漏/多余/0 路）。
    """
    check_int(sets, "sets", minimum=1, maximum=MAX_SETS)
    normalized = normalize_apps(apps)

    if isinstance(allocation, Mapping):
        alloc_items = list(allocation.items())
    elif isinstance(allocation, (list, tuple)):
        alloc_items = list(allocation)
    else:
        raise InvalidInputError(
            f"allocation: 必须是映射或 (app_id, ways) 序列，"
            f"得到 {type(allocation).__name__}"
        )

    alloc_map: dict[int, int] = {}
    for j, item in enumerate(alloc_items):
        if (not isinstance(item, tuple) or len(item) != 2):
            raise InvalidInputError(
                f"allocation[{j}]: 必须是 (app_id, ways) 二元组"
            )
        aid, ways = item
        aid = check_int(aid, f"allocation[{j}].app_id", minimum=0)
        ways = check_int(
            ways, f"allocation[{j}].ways", minimum=1, maximum=MAX_TOTAL_WAYS
        )
        if aid in alloc_map:
            raise InvalidInputError(f"allocation: app_id {aid} 重复分配")
        alloc_map[aid] = ways

    app_ids = [a[0] for a in normalized]
    missing = [a for a in app_ids if a not in alloc_map]
    extra = [a for a in alloc_map if a not in set(app_ids)]
    if missing or extra:
        raise InvalidInputError(
            f"allocation: 必须与应用集合一一对应；缺少 {sorted(missing)}，"
            f"多余 {sorted(extra)}"
        )

    per_app: list[ReplayAppStats] = []
    for app_id, blocks, weight in normalized:
        ways = alloc_map[app_id]
        result = _simulate(blocks, sets, ways)
        per_app.append(
            ReplayAppStats(
                app_id=app_id,
                ways=ways,
                sets=sets,
                weight=weight,
                accesses=result.accesses,
                misses=result.misses,
                hits=result.hits,
                weighted_misses=weight * result.misses,
            )
        )

    return ReplayReport(
        sets=sets,
        per_app=tuple(per_app),
        total_accesses=sum(s.accesses for s in per_app),
        total_misses=sum(s.misses for s in per_app),
        total_hits=sum(s.hits for s in per_app),
        weighted_miss_total=sum(s.weighted_misses for s in per_app),
        total_ways=sum(s.ways for s in per_app),
    )
