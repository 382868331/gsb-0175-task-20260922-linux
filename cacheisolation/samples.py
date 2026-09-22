"""仓库内确定性合成访存序列（无附件、无网络、无随机种子依赖外部状态）。

序列由固定参数的纯函数生成，重复运行逐位一致。包含：

* :func:`cycling_trace` —— 在 ``footprint`` 个块上循环访问，用于构造
  “循环冲突”（路数不足时每次访问都缺失）与“无收益的多余路”
  （路数达到 footprint 后再加路 miss 不变）；
* :func:`demo_workload` —— 正常演示用的 3 应用混合负载；
* :func:`counterexample_workload` —— 非凸曲线导致朴素边际贪心次优、
  而精确 DP 最优的两应用反例（附期望数值，供测试与 demo 核对）。
"""

from __future__ import annotations

from .models import AppTrace


def cycling_trace(app_id: int, footprint: int, rounds: int,
                  *, offset: int = 0, weight: int = 1) -> AppTrace:
    """``[offset, offset+footprint)`` 内块地址循环 ``rounds`` 轮。

    在单组（sets=1）LRU 下：ways < footprint 时每次访问都冲突失效
    （miss = footprint*rounds）；ways >= footprint 时仅首轮冷缺失
    （miss = footprint）。这正是“循环冲突 + 台阶型（非凸）收益”。
    """
    if footprint < 1 or rounds < 1:
        raise ValueError("cycling_trace: footprint 与 rounds 必须 >= 1")
    blocks = tuple(offset + (t % footprint) for t in range(footprint * rounds))
    return AppTrace(app_id=app_id, blocks=blocks, weight=weight)


def demo_workload() -> "tuple[list[AppTrace], int, int]":
    """正常演示负载：3 应用、4 组、预算 8 路。

    各应用由不同足迹的循环段与固定热区段拼接而成，权重为正整数。
    返回 (apps, sets, budget)。
    """
    # 应用 0：大足迹循环（4 组、单组热度高，需要多路才能止住冲突）。
    a0_blocks = tuple(t % 9 for t in range(36)) + (0, 1) * 6
    # 应用 1：两个映射到同一组（地址 %4==0）的块交替 -> 1 路抖动、
    # 2 路即止住冲突，之后平坦。
    a1_blocks = (100, 104) * 12
    # 应用 2：两个落在不同组的块交替，1 路就足够 -> 多余路无收益（平坦）。
    a2_blocks = (200, 201) * 15
    apps = [
        AppTrace(0, a0_blocks, weight=2),
        AppTrace(1, a1_blocks, weight=1),
        AppTrace(7, a2_blocks, weight=3),  # 故意非连续 ID，验证按 ID 排序
    ]
    return apps, 4, 8


def counterexample_workload() -> "tuple[list[AppTrace], int, int]":
    """贪心边际反例负载：2 应用、1 组、预算 4。

    构造（权重见下，N=12 次访问）：

    * 应用 0（w=10）：在 3 个块上循环。miss 曲线为
      k=1: 12, k=2: 12, k=3: 3, k>=3: 3 —— 第 2 路零收益、
      第 3 路才有大收益（非凸台阶）。
    * 应用 1（w=1）：在 2 个块上交替。miss 曲线为
      k=1: 12, k>=2: 2 —— 第 2 路即有大收益，之后无收益。

    朴素边际贪心从 (1,1) 出发：app0 的 1→2 路加权收益为 0，
    app1 的 1→2 路收益为 10，于是分给 app1 后再无正收益，
    停在 (1,2)，加权 miss = 10*12 + 1*2 = 122。

    精确 DP 可取 (3,1)（用满 4 路）：10*3 + 1*12 = 42，严格更优。
    """
    apps = [
        cycling_trace(0, footprint=3, rounds=4, weight=10),
        cycling_trace(1, footprint=2, rounds=6, weight=1),
    ]
    return apps, 1, 4


# 反例的手工核对期望值（sets=1）。
COUNTEREXAMPLE_CURVES: dict[int, tuple[int, ...]] = {
    0: (12, 12, 3, 3),
    1: (12, 2, 2, 2),
}
GREEDY_ALLOCATION = {0: 1, 1: 2}
GREEDY_TOTAL = 122
OPTIMAL_ALLOCATION = {0: 3, 1: 1}
OPTIMAL_TOTAL = 42
