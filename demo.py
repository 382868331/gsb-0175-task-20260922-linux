#!/usr/bin/env python3
"""固定输入演示：真实计算，无 sleep、无预录结果、无网络。

运行：
    python demo.py

内容：
  1) 正常结果：3 应用 / 4 组 / 预算 8 路，打印逐路 miss 曲线、
     精确 DP 最优分配（含并列信息）与隔离缓存重放统计；
  2) 贪心反例：非凸 miss 曲线下，朴素边际贪心严格劣于精确 DP；
  3) 异常边界：实际触发一个“已证明不可行预算”与一个“非法输入”拒绝。

规模刻意小（远低于上限），整个过程毫秒级完成。
"""

from __future__ import annotations

from cacheisolation import (
    InfeasibleAllocationError,
    InvalidInputError,
    AppTrace,
    build_miss_curves,
    greedy_marginal_allocation,
    optimize_allocation,
    plan_isolation,
)
from cacheisolation.samples import counterexample_workload, demo_workload


def hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def show_normal_result() -> None:
    hr("[1/3] 正常结果：逐路曲线 -> 精确 DP 最优分配 -> 隔离重放")
    apps, sets, budget = demo_workload()
    result = plan_isolation(apps, sets, budget, max_ways=8)
    curves, plan, replay = result.curves, result.plan, result.replay

    print(f"缓存组数 sets={sets}，总路数预算 budget={budget}，"
          f"应用数 n={len(apps)}")
    print("\n逐路 miss 曲线（每应用独立模拟，ways=1..8；不假设凸性）：")
    header = "app_id | weight | " + " ".join(f"w={w:<2}" for w in range(1, 9))
    print(header)
    for c in curves:
        print(f"  {c.app_id:>4} | {c.weight:>6} | "
              + " ".join(f"{c.at(w):<3}" for w in range(1, 9)))

    print("\n精确 DP 最优分配（并列时取分配向量字典序最小）：")
    for app_id, ways in plan.allocation:
        print(f"  app {app_id:>2} -> {ways} 路")
    print(f"  使用路数 ways_used = {plan.ways_used} (<= {budget})")
    print(f"  加权 miss 总和 weighted_total = {plan.weighted_total}")
    tie_note = "（无并列）"
    if plan.ties:
        shown = [dict(v) for v in plan.ties[:4]]
        more = "" if plan.tie_count <= 5 else f"，…… 共 {plan.tie_count} 个并列"
        tie_note = f"（这些都是把多余路加到平坦区域的等价向量）: {shown}{more}"
    print(f"  并列最优解个数 tie_count = {plan.tie_count} {tie_note}")
    print(f"  DP 转移计数 iterations = {plan.iterations} "
          f"(<= 完整表预算 {plan.budget_states})")

    print("\n隔离缓存上的重放统计（每应用私有 LRU 分片，互不干扰）：")
    print("  app | ways | accesses | misses | hits | miss_rate | w*miss")
    for s in replay.per_app:
        print(f"  {s.app_id:>3} | {s.ways:>4} | {s.accesses:>8} | "
              f"{s.misses:>6} | {s.hits:>4} | {s.miss_rate:>9.4f} | "
              f"{s.weighted_misses}")
    print(f"  合计: accesses={replay.total_accesses}, "
          f"misses={replay.total_misses}, hits={replay.total_hits}, "
          f"加权 miss={replay.weighted_miss_total}")
    assert replay.weighted_miss_total == plan.weighted_total


def show_greedy_counterexample() -> None:
    hr("[2/3] 贪心反例：非凸 miss 曲线下边际贪心被误导")
    apps, sets, budget = counterexample_workload()
    curves = build_miss_curves(apps, sets, max_ways=4)
    print(f"sets={sets}, budget={budget}")
    for c in curves:
        print(f"  app {c.app_id} (weight={c.weight}) 逐路 miss: "
              + ", ".join(f"w{w}={c.at(w)}" for w in range(1, 5)))
    g = greedy_marginal_allocation(curves, budget)
    opt = optimize_allocation(curves, budget)
    print(f"\n朴素边际贪心分配: {dict(g.allocation)}  "
          f"加权 miss = {g.weighted_total}")
    print(f"精确 DP 最优分配: {opt.as_dict()}  "
          f"加权 miss = {opt.weighted_total}")
    print(f"=> 贪心劣于最优，差值 = {g.weighted_total - opt.weighted_total}")
    assert g.weighted_total > opt.weighted_total


def show_rejected_boundaries() -> None:
    hr("[3/3] 实际触发的拒绝/异常边界（真实调用并捕获）")

    # (a) 合法输入但数学上已证明不可行：3 个应用至少各需 1 路，预算 2。
    apps = [
        AppTrace(0, [0, 1, 0], weight=1),
        AppTrace(1, [4, 5, 4], weight=2),
        AppTrace(2, [8, 9, 8], weight=3),
    ]
    try:
        plan_isolation(apps, sets=2, budget=2)
    except InfeasibleAllocationError as e:
        print(f"(a) 不可行预算 -> InfeasibleAllocationError:\n    {e}")
    else:
        raise AssertionError("应当判定不可行")

    # (b) 非法输入：bool 冒充整数路数预算（True 虽是 int 子类但被拒绝）。
    try:
        plan_isolation(apps, sets=2, budget=True)  # type: ignore[arg-type]
    except InvalidInputError as e:
        print(f"(b) bool 预算 -> InvalidInputError:\n    {e}")
    else:
        raise AssertionError("应当拒绝 bool")

    # (c) 非法输入：负块地址，错误消息可定位到具体下标。
    try:
        plan_isolation([AppTrace(0, [0, 1, -7])], sets=1, budget=2)
    except InvalidInputError as e:
        print(f"(c) 负块地址 -> InvalidInputError:\n    {e}")
    else:
        raise AssertionError("应当拒绝负块地址")


def main() -> None:
    show_normal_result()
    show_greedy_counterexample()
    show_rejected_boundaries()
    hr("演示完成：以上数值均为本次真实计算所得")


if __name__ == "__main__":
    main()
