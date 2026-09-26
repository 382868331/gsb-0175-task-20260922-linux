"""Tests for plan_fair_allocation (lexicographic fair way allocation).

The brute-force reference enumerates every integer allocation with
1..total_ways ways per app and total <= budget, scores each with the
three-level objective (exact Fraction max-ratio, weighted miss sum,
allocation vector) and takes the minimum.  It never calls the planner
under test; only miss_curve (independently tested) is shared.
"""

import itertools
import random
import unittest
from fractions import Fraction

from wayplan import (
    FairPlanResult,
    InfeasibleBudgetError,
    ValidationError,
    miss_curve,
    plan_allocation,
    plan_fair_allocation,
)


def brute_force_fair(curves, weights, references, total_ways, budget):
    """Exhaustive reference: best (max_ratio, weighted_sum, allocation)."""
    n = len(curves)
    best = None
    for alloc in itertools.product(range(1, total_ways + 1), repeat=n):
        if sum(alloc) > budget:
            continue
        ratios = [Fraction(curves[i][alloc[i] - 1], references[i])
                  for i in range(n)]
        weighted = sum(weights[i] * curves[i][alloc[i] - 1] for i in range(n))
        cand = (max(ratios), weighted, alloc)
        if best is None or cand < best:
            best = cand
    return best


class FairPlanStructureTest(unittest.TestCase):
    def setUp(self):
        self.traces = [
            ("beta", ([0, 1, 2] * 30)[:80]),
            ("alpha", ([0] * 7 + [1]) * 10),
        ]
        self.weights = [2, 3]
        self.references = [80, 40]

    def test_result_fields_and_exact_ratio(self):
        res = plan_fair_allocation(self.traces, self.weights, self.references,
                                   num_sets=1, total_ways=6, budget=4)
        self.assertIsInstance(res, FairPlanResult)
        self.assertEqual(res.app_ids, ("alpha", "beta"))  # sorted by app id
        self.assertEqual(res.weights, (3, 2))             # follow the apps
        self.assertEqual(res.references, (40, 80))
        self.assertEqual(res.used_ways, sum(res.allocation))
        self.assertLessEqual(res.used_ways, res.budget)
        self.assertTrue(all(a >= 1 for a in res.allocation))
        # max_ratio fields form a reduced fraction equal to the property
        frac = Fraction(res.max_ratio_num, res.max_ratio_den)
        self.assertEqual(frac, res.max_ratio)
        self.assertEqual(
            frac,
            max(Fraction(m, r) for m, r in zip(res.per_app_misses,
                                               res.references)),
        )
        # max_ratio_apps are exactly the apps attaining the ratio
        expected = tuple(
            app for app, m, r in zip(res.app_ids, res.per_app_misses,
                                     res.references)
            if Fraction(m, r) == frac
        )
        self.assertEqual(res.max_ratio_apps, expected)
        self.assertEqual(res.allocation_map(),
                         dict(zip(res.app_ids, res.allocation)))

    def test_every_app_gets_at_least_one_way(self):
        # Fair mode never disables an application, even with skewed weights.
        res = plan_fair_allocation(self.traces, [1000, 1], self.references,
                                   num_sets=1, total_ways=6, budget=3)
        self.assertTrue(all(a >= 1 for a in res.allocation))


class FairPlanBruteForceTest(unittest.TestCase):
    def check(self, traces, weights, references, num_sets, total_ways, budget):
        res = plan_fair_allocation(traces, weights, references,
                                   num_sets=num_sets, total_ways=total_ways,
                                   budget=budget)
        order = sorted(range(len(traces)), key=lambda i: traces[i][0])
        curves = tuple(
            miss_curve(traces[i][1], num_sets, total_ways) for i in order
        )
        ws = tuple(weights[i] for i in order)
        refs = tuple(references[i] for i in order)
        want_ratio, want_weighted, want_alloc = brute_force_fair(
            curves, ws, refs, total_ways, budget)
        self.assertEqual(res.allocation, want_alloc)
        self.assertEqual(res.max_ratio, want_ratio)
        self.assertEqual(res.weighted_misses, want_weighted)

    def test_hand_built_cases(self):
        traces = [
            ("c", [i % 5 for i in range(60)]),
            ("a", ([0] * 5 + [1, 2]) * 8),
            ("b", [7] * 40 + [8, 9] * 10),
        ]
        for budget in (3, 4, 5):
            with self.subTest(budget=budget):
                self.check(traces, [1, 3, 2], [60, 30, 45],
                           num_sets=2, total_ways=5, budget=budget)

    def test_randomised_small_cases(self):
        rng = random.Random(20260926)
        for trial in range(30):
            n = rng.randint(1, 3)
            total_ways = rng.randint(3, 5)
            budget = rng.randint(n, total_ways)
            traces, weights, refs = [], [], []
            for k in range(n):
                size = rng.randint(2, 6)
                blocks = [rng.randrange(size) for _ in range(40)]
                traces.append((f"app{k}", blocks))
                weights.append(rng.randint(1, 4))
                refs.append(rng.randint(1, 40))
            with self.subTest(trial=trial):
                self.check(traces, weights, refs,
                           num_sets=rng.randint(1, 3),
                           total_ways=total_ways, budget=budget)

    def test_no_float_tolerance_ties(self):
        # Two ratios that agree to ~1e-16 as floats but differ exactly:
        # 1/3 vs 3333333333333333/10**16 must not be conflated.
        self.assertNotEqual(Fraction(1, 3),
                            Fraction(3333333333333333, 10**16))
        # Planner-level check: app "a" misses 1/3, app "b" misses the
        # nearby-but-distinct ratio; the max must be reported exactly.
        res = plan_fair_allocation(
            [("a", [0]), ("b", [1])], [1, 1], [3, 10**16],
            num_sets=2, total_ways=2, budget=2)
        self.assertEqual(res.allocation, (1, 1))
        self.assertEqual(res.per_app_misses, (1, 1))
        self.assertEqual(res.max_ratio, Fraction(1, 3))
        self.assertEqual(res.max_ratio_apps, ("a",))


class FairVsCostMinTest(unittest.TestCase):
    """A case where the fair plan costs more total misses but helps the
    worst-off application — the documented trade-off of the two entries."""

    def test_fair_mode_accepts_higher_total_for_lower_max_ratio(self):
        # appA: 1 way -> 40 misses, 2 ways -> 2 (huge early gain, weight 6).
        # appB: cyclic 3-block loop; <3 ways -> 200 misses, 3 ways -> 3.
        blocks_a = ([0] * 9 + [1]) * 20
        blocks_b = ([10, 11, 12] * 67)[:200]
        traces = [("appA", blocks_a), ("appB", blocks_b)]
        weights = [6, 1]
        references = [80, 200]
        kw = dict(num_sets=1, total_ways=4, budget=4)

        cost = plan_allocation(traces, weights, **kw)
        fair = plan_fair_allocation(traces, weights, references, **kw)

        # Cost-min entry: appA 2 ways + appB 1 way (weighted total 212;
        # giving appB a 2nd way buys nothing under LRU thrashing) but
        # appB sits at ratio 200/200 = 1.
        self.assertEqual(cost.allocation, (2, 1))
        self.assertEqual(cost.weighted_misses, 212)
        # Fair entry: appB gets its 3rd way; weighted total rises to 243
        # but the worst ratio drops from 1 to 40/80 = 1/2.
        self.assertEqual(fair.allocation, (1, 3))
        self.assertEqual(fair.weighted_misses, 243)
        self.assertGreater(fair.weighted_misses, cost.weighted_misses)
        self.assertEqual(fair.max_ratio, Fraction(1, 2))
        self.assertEqual(fair.max_ratio_apps, ("appA",))
        fair_worst = max(Fraction(m, r) for m, r in zip(
            fair.per_app_misses, fair.references))
        cost_worst = max(Fraction(m, r) for m, r in zip(
            cost.per_app_misses, references))
        self.assertLess(fair_worst, cost_worst)


class FairValidationTest(unittest.TestCase):
    traces = [("a", [0, 1]), ("b", [2])]

    def plan(self, **over):
        kw = dict(traces=self.traces, weights=[1, 1], references=[2, 1],
                  num_sets=2, total_ways=4, budget=3)
        kw.update(over)
        return plan_fair_allocation(**kw)

    def test_reference_errors(self):
        for bad, frag in [
            ([0, 1], "references[0]"),          # zero
            ([-1, 1], "references[0]"),         # negative
            ([1.5, 1], "references[0]"),        # float
            ([True, 1], "references[0]"),       # bool
            ([2], "references"),                # wrong length
            ("xx", "references"),               # not a sequence of ints
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError) as cm:
                    self.plan(references=bad)
                self.assertIn(frag, str(cm.exception))

    def test_infeasible_budget(self):
        with self.assertRaises(InfeasibleBudgetError):
            self.plan(budget=1)  # 2 apps need >= 2 ways

    def test_old_entry_still_works(self):
        res = plan_allocation(self.traces, [1, 1],
                              num_sets=2, total_ways=4, budget=3)
        self.assertEqual(res.used_ways, sum(res.allocation))


if __name__ == "__main__":
    unittest.main()
