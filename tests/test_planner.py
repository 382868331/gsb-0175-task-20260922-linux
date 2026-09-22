"""规划器测试：暴力枚举对照 DP、并列字典序、不可行预算、搜索预算、权重。"""

from __future__ import annotations

import unittest

from cacheisolation import (
    InfeasibleAllocationError,
    InvalidInputError,
    MissCurve,
    build_miss_curves,
    optimize_allocation,
)
from cacheisolation.models import AppTrace

from reference import brute_force_optimal, reference_lru_misses


def curve(app_id: int, misses, weight=1, sets=1) -> MissCurve:
    return MissCurve(app_id=app_id, sets=sets, weight=weight, misses=tuple(misses))


class DPvsBruteForceTests(unittest.TestCase):
    """小组合：精确 DP 与独立暴力枚举逐一对照（含全部并列向量）。"""

    def assert_matches_brute(self, curves, budget):
        raw = [list(c.misses) for c in curves]
        weights = [c.weight for c in curves]
        exp_val, exp_vecs = brute_force_optimal(raw, weights, budget)
        plan = optimize_allocation(curves, budget)
        self.assertEqual(plan.weighted_total, exp_val)
        got_vec = tuple(k for _, k in plan.allocation)
        self.assertEqual(got_vec, exp_vecs[0])  # 主解 = 字典序最小
        # 全部并列向量一致（主解 + ties）。
        got_all = [got_vec] + [
            tuple(k for _, k in vec) for vec in plan.ties
        ]
        self.assertEqual(sorted(got_all), exp_vecs)
        self.assertEqual(plan.tie_count, len(exp_vecs))
        self.assertLessEqual(plan.ways_used, budget)

    def test_nonconvex_counterexample_curves(self):
        curves = [curve(0, (12, 12, 3, 3), weight=10),
                  curve(1, (12, 2, 2, 2), weight=1)]
        # budget=4：最优 (3,1)=42；贪心反例见 test_greedy_counterexample。
        self.assert_matches_brute(curves, 4)
        plan = optimize_allocation(curves, 4)
        self.assertEqual(plan.as_dict(), {0: 3, 1: 1})
        self.assertEqual(plan.weighted_total, 42)

    def test_random_like_shapes_vs_brute(self):
        # 由确定性规则生成的非凸/平台/递减混合曲线，覆盖多种预算。
        shapes = [
            [(10, 8, 8, 3, 3), (9, 4, 4, 4, 2), (7, 7, 5, 1, 1)],
            [(5, 5, 5), (6, 3, 0), (4, 4, 2), (8, 2, 2)],
            [(1, 0, 0, 0), (20, 19, 1, 1)],
            [(7, 6, 5, 4, 3, 2, 1)],  # 单应用、严格递减（凸）
            [(9, 9, 0, 9, 0)],        # 故意非单调也能求（仿真不会产生，
                                      # 但 DP 不依赖单调性）
        ]
        weights_sets = [
            [1, 1, 1], [2, 3, 1, 5], [7, 1], [4], [2],
        ]
        for shape, wts in zip(shapes, weights_sets):
            curves = [curve(i, m, weight=w) for i, (m, w) in
                      enumerate(zip(shape, wts))]
            n = len(curves)
            for budget in range(n, 8):
                self.assert_matches_brute(curves, budget)

    def test_curves_from_simulation_vs_brute(self):
        # 端到端：曲线由（被测）仿真产生，但期望值用独立参考重算，
        # 最优值用独立暴力枚举核对。
        apps = [
            AppTrace(0, tuple((t * 3) % 7 for t in range(40)), weight=2),
            AppTrace(1, tuple((t // 2) % 5 for t in range(30)), weight=5),
            AppTrace(3, tuple(range(8)) * 4, weight=1),
        ]
        sets, budget = 3, 7
        curves = build_miss_curves(apps, sets, max_ways=6)
        # 用独立参考验证曲线本身。
        for app, c in zip(apps, curves):
            for w in range(1, 7):
                self.assertEqual(
                    c.at(w), reference_lru_misses(app.blocks, sets, w)
                )
        raw = [list(c.misses) for c in curves]
        weights = [a.weight for a in apps]
        exp_val, exp_vecs = brute_force_optimal(raw, weights, budget)
        plan = optimize_allocation(curves, budget)
        self.assertEqual(plan.weighted_total, exp_val)
        self.assertEqual(tuple(k for _, k in plan.allocation), exp_vecs[0])


class TieBreakTests(unittest.TestCase):
    def test_lexicographic_primary(self):
        # 两条完全相同的曲线 => 多种分配同值；主解必须字典序最小，
        # 且按 app_id 升序的向量比较。
        curves = [curve(3, (10, 5, 0), weight=1),
                  curve(9, (10, 5, 0), weight=1)]
        plan = optimize_allocation(curves, 4)
        # 所有 k1+k2<=4、k>=1 的方案里最小值 0 出现在两者都 3 路(超预算)；
        # budget=4 下可达最小：列举 (1,3)=10、(2,2)=10、(3,1)=10。
        self.assertEqual(plan.weighted_total, 10)
        self.assertEqual(plan.as_dict(), {3: 1, 9: 3})  # 字典序 (1,3)
        self.assertEqual(plan.tie_count, 3)
        self.assertEqual(
            sorted(tuple(k for _, k in v) for v in plan.ties),
            [(2, 2), (3, 1)],
        )

    def test_spare_budget_not_forced(self):
        # 最优用不满预算也可以：多余路无收益时选择字典序最小向量，
        # 而字典序最小意味着前面的应用路数尽量小。
        curves = [curve(0, (5, 5, 5), weight=1),
                  curve(1, (0, 0, 0), weight=1)]
        plan = optimize_allocation(curves, 6)
        self.assertEqual(plan.weighted_total, 5)
        self.assertEqual(plan.as_dict(), {0: 1, 1: 1})
        self.assertLess(plan.ways_used, 6)


class InfeasibleTests(unittest.TestCase):
    def test_budget_below_app_count(self):
        curves = [curve(0, (1,)), curve(1, (1,)), curve(2, (1,))]
        with self.assertRaises(InfeasibleAllocationError):
            optimize_allocation(curves, 2)  # 3 应用、预算 2

    def test_infeasible_is_not_budget_exhausted(self):
        # 两种失败必须是不同异常类型。
        from cacheisolation import SearchBudgetExhausted
        curves = [curve(0, (1,)), curve(1, (1,))]
        with self.assertRaises(InfeasibleAllocationError):
            optimize_allocation(curves, 1)
        # 预算极小导致搜索中断 => SearchBudgetExhausted，而非不可行。
        with self.assertRaises(SearchBudgetExhausted):
            optimize_allocation(
                [curve(0, (10, 9, 8)), curve(1, (10, 9, 8))], 3,
                state_budget=1,
            )

    def test_budget_exhausted_does_not_mutate_inputs(self):
        curves = [curve(0, (10, 9, 8, 7)), curve(1, (10, 9, 8, 7))]
        before = tuple(tuple(c.misses) for c in curves)
        with self.assertRaises(Exception):
            optimize_allocation(curves, 4, state_budget=2)
        after = tuple(tuple(c.misses) for c in curves)
        self.assertEqual(before, after)  # 无部分变更（输入本就不可变）


class PlannerValidationTests(unittest.TestCase):
    def test_bool_rejected_as_budget(self):
        curves = [curve(0, (1, 1))]
        with self.assertRaises(InvalidInputError):
            optimize_allocation(curves, True)

    def test_budget_out_of_range(self):
        curves = [curve(0, (1,))]
        for bad in (0, 13, -1):
            with self.assertRaises(InvalidInputError):
                optimize_allocation(curves, bad)

    def test_unsorted_or_duplicate_ids(self):
        with self.assertRaises(InvalidInputError):
            optimize_allocation([curve(2, (1,)), curve(1, (1,))], 2)
        with self.assertRaises(InvalidInputError):
            optimize_allocation([curve(0, (1,)), curve(0, (1,))], 2)

    def test_weight_validation_on_curves_build(self):
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0, 1], weight=0)], 1)
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0, 1], weight=True)], 1)


if __name__ == "__main__":
    unittest.main()
