"""输入校验、重放、端到端流水线与反例样本测试。"""

from __future__ import annotations

import math
import unittest

from cacheisolation import (
    AppTrace,
    InfeasibleAllocationError,
    InvalidInputError,
    SearchBudgetExhausted,
    build_miss_curves,
    greedy_marginal_allocation,
    optimize_allocation,
    plan_isolation,
    replay_partition,
)
from cacheisolation.samples import (
    COUNTEREXAMPLE_CURVES,
    GREEDY_ALLOCATION,
    GREEDY_TOTAL,
    OPTIMAL_ALLOCATION,
    OPTIMAL_TOTAL,
    counterexample_workload,
    demo_workload,
)


class NumericValidationTests(unittest.TestCase):
    def test_nan_infinity_rejected(self):
        # 数值入口（miss_rate 只产出不接收；权重为 int 契约，这里通过
        # 校验原语与有限数值入口验证 NaN/Inf 拒绝）。
        from cacheisolation.validation import check_finite_number
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(InvalidInputError):
                check_finite_number(bad, "x")
        self.assertEqual(check_finite_number(3, "x"), 3.0)
        self.assertEqual(check_finite_number(3.5, "x"), 3.5)

    def test_bool_rejected_everywhere(self):
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0, 1])], True)          # sets
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0, 1])], 1, True)        # max_ways
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(True, [0, 1])], 1)           # app_id
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0, True, 1])], 1)        # block
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0, 1], weight=False)], 1)

    def test_locatable_messages(self):
        try:
            build_miss_curves([AppTrace(0, [0, -1])], 1)
        except InvalidInputError as e:
            self.assertIn("apps[0].blocks[1]", str(e))
        else:
            self.fail("应拒绝负块地址")
        try:
            build_miss_curves("not-a-list", 1)
        except InvalidInputError as e:
            self.assertIn("apps", str(e))
        else:
            self.fail("应拒绝非序列 apps")

    def test_limits_enforced(self):
        with self.assertRaises(InvalidInputError):
            build_miss_curves([], 1)
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(i, [0]) for i in range(7)], 1)
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0])], 17)
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0])], 1, 13)
        with self.assertRaises(InvalidInputError):
            build_miss_curves([AppTrace(0, [0]), AppTrace(0, [1])], 1)


class ReplayTests(unittest.TestCase):
    def test_replay_matches_curves_and_plan(self):
        apps, sets, budget = demo_workload()
        result = plan_isolation(apps, sets, budget, max_ways=8)
        curves, plan, report = result.curves, result.plan, result.replay
        # 每应用重放 miss == 曲线在分配路数上的值。
        for c, st in zip(curves, report.per_app):
            self.assertEqual(st.misses, c.at(st.ways))
            self.assertEqual(st.weighted_misses, c.weight * c.at(st.ways))
        # 汇总恒等式（精确整数）。
        self.assertEqual(report.weighted_miss_total, plan.weighted_total)
        self.assertEqual(report.total_ways, plan.ways_used)
        self.assertEqual(
            report.total_misses + report.total_hits, report.total_accesses
        )
        # miss_rate 口径：一次 double 除法。
        for st in report.per_app:
            self.assertEqual(
                st.miss_rate,
                0.0 if st.accesses == 0 else st.misses / st.accesses,
            )

    def test_replay_allocation_mismatch(self):
        apps, sets, _ = demo_workload()
        with self.assertRaises(InvalidInputError):
            replay_partition(apps, {0: 2, 1: 2}, sets)       # 缺 app 7
        with self.assertRaises(InvalidInputError):
            replay_partition(apps, {0: 2, 1: 2, 7: 2, 9: 1}, sets)  # 多余
        with self.assertRaises(InvalidInputError):
            replay_partition(apps, {0: 0, 1: 2, 7: 2}, sets)  # 0 路
        # (app_id, ways) 序列形式合法。
        ok = replay_partition(apps, [(0, 2), (1, 2), (7, 2)], sets)
        self.assertEqual(ok.total_ways, 6)

    def test_empty_trace_replay(self):
        apps = [AppTrace(0, [], weight=3), AppTrace(1, [0, 0], weight=1)]
        result = plan_isolation(apps, 2, 3, max_ways=3)
        st0 = result.replay.per_app[0]
        self.assertEqual((st0.accesses, st0.misses, st0.hits), (0, 0, 0))
        self.assertEqual(st0.miss_rate, 0.0)
        self.assertEqual(result.plan.weighted_total, 1)  # app1: [0,0] 1 miss


class GreedyCounterexampleTests(unittest.TestCase):
    def setUp(self):
        self.apps, self.sets, self.budget = counterexample_workload()
        self.curves = build_miss_curves(self.apps, self.sets, 4)

    def test_hand_checked_curves(self):
        for c in self.curves:
            self.assertEqual(tuple(c.misses), COUNTEREXAMPLE_CURVES[c.app_id])

    def test_greedy_is_suboptimal(self):
        g = greedy_marginal_allocation(self.curves, self.budget)
        self.assertEqual(dict(g.allocation), GREEDY_ALLOCATION)
        self.assertEqual(g.weighted_total, GREEDY_TOTAL)
        plan = optimize_allocation(self.curves, self.budget)
        self.assertEqual(plan.as_dict(), OPTIMAL_ALLOCATION)
        self.assertEqual(plan.weighted_total, OPTIMAL_TOTAL)
        # 反例成立：贪心严格劣于 DP。
        self.assertGreater(g.weighted_total, plan.weighted_total)

    def test_demo_workload_runs_and_is_consistent(self):
        apps, sets, budget = demo_workload()
        result = plan_isolation(apps, sets, budget, max_ways=8)
        self.assertLessEqual(result.plan.ways_used, budget)
        # 隔离重放与规划目标一致。
        self.assertEqual(
            result.replay.weighted_miss_total, result.plan.weighted_total
        )


class FailureSemanticsTests(unittest.TestCase):
    def test_infeasible_through_pipeline(self):
        apps = [AppTrace(0, [0, 1]), AppTrace(1, [2, 3])]
        with self.assertRaises(InfeasibleAllocationError):
            plan_isolation(apps, 1, 1)  # 2 应用、预算 1

    def test_budget_exhausted_type(self):
        apps, _, _ = counterexample_workload()
        curves = build_miss_curves(apps, 1, 4)
        with self.assertRaises(SearchBudgetExhausted):
            optimize_allocation(curves, 4, state_budget=1)
        # 放宽预算后正常得到最优。
        plan = optimize_allocation(curves, 4)
        self.assertEqual(plan.weighted_total, OPTIMAL_TOTAL)


if __name__ == "__main__":
    unittest.main()
