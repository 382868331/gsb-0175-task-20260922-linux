"""Fair planner tests: hand-checked cases, exact rational comparisons,
brute-force cross-check, and edge cases.

The brute-force reference in this file is fully independent of the wayplan
core: it uses its own list-based LRU simulator and enumerates every legal
allocation, comparing the three-tier objective with fractions.Fraction.
It never calls wayplan to obtain expected values.
"""

import itertools
import random
import unittest
from fractions import Fraction

from wayplan import (
    InfeasibleBudgetError,
    ValidationError,
    plan_allocation,
    plan_fair_allocation,
)


# ---------------------------------------------------------------------------
# Independent reference implementation (NOT the code under test)
# ---------------------------------------------------------------------------

def ref_lru_misses(blocks, num_sets, num_ways):
    """Deliberately naive list-based LRU; cold cache; set = block % num_sets."""
    sets = [[] for _ in range(num_sets)]  # index 0 = LRU
    misses = 0
    for b in blocks:
        s = sets[b % num_sets]
        if b in s:
            s.remove(b)
            s.append(b)
        else:
            misses += 1
            s.append(b)
            if len(s) > num_ways:
                s.pop(0)
    return misses


def ref_fair_brute_force(traces_blocks, weights, references, num_sets,
                         total_ways, budget):
    """Enumerate every legal allocation; return the best (key, allocation).

    Key = (max ratio as Fraction, weighted miss sum, allocation tuple),
    compared lexicographically — the documented three-tier objective.
    """
    n = len(traces_blocks)
    curves = [
        [ref_lru_misses(bl, num_sets, w) for w in range(1, total_ways + 1)]
        for bl in traces_blocks
    ]
    best = None
    for alloc in itertools.product(range(1, total_ways + 1), repeat=n):
        if sum(alloc) > budget:
            continue
        misses = [curves[i][alloc[i] - 1] for i in range(n)]
        key = (
            max(Fraction(misses[i], references[i]) for i in range(n)),
            sum(weights[i] * misses[i] for i in range(n)),
            alloc,
        )
        if best is None or key < best[0]:
            best = (key, alloc)
    return best


# Hand-checked divergence instance (curves verified by hand simulation):
#   app a, num_sets=1: 1 way -> 6 misses, 2 ways -> 4 misses
#   app b, num_sets=1: 1 way -> 5 misses, 2 ways -> 4 misses
# references a=7, b=6; weights a=1, b=3; budget 3.
#   (2, 1): ratios 4/7, 5/6 -> max 5/6; weighted 4 + 15 = 19
#   (1, 2): ratios 6/7, 2/3 -> max 6/7; weighted 6 + 12 = 18
#   (1, 1): ratios 6/7, 5/6 -> max 6/7
# 5/6 < 6/7 (they differ by 1/42), so the fair optimum is (2, 1) even
# though its weighted miss sum is worse by 1.
HAND_TRACES = [("a", [2, 0, 4, 2, 4, 4, 2, 2]), ("b", [2, 3, 3, 2, 0, 0, 0, 1])]
HAND_WEIGHTS = [1, 3]
HAND_REFS = [7, 6]


class HandCheckedFairTests(unittest.TestCase):
    def test_hand_case_picks_fair_allocation(self):
        res = plan_fair_allocation(
            HAND_TRACES, HAND_WEIGHTS, HAND_REFS,
            num_sets=1, total_ways=4, budget=3,
        )
        self.assertEqual(res.allocation, (2, 1))
        self.assertEqual(res.per_app_misses, (4, 5))
        self.assertEqual(res.weighted_misses, 19)
        self.assertEqual(res.used_ways, 3)

    def test_hand_case_exact_max_ratio(self):
        res = plan_fair_allocation(
            HAND_TRACES, HAND_WEIGHTS, HAND_REFS,
            num_sets=1, total_ways=4, budget=3,
        )
        # Exact numerator/denominator, not a float approximation.
        self.assertEqual((res.max_ratio_num, res.max_ratio_den), (5, 6))
        self.assertEqual(res.max_ratio, Fraction(5, 6))
        self.assertEqual(res.max_ratio_apps, ("b",))

    def test_close_ratios_are_not_merged_by_tolerance(self):
        # 5/6 and 6/7 differ by only 1/42; any float tolerance >= 1/42
        # would call the two candidate plans a tie and (by the weighted
        # tier) pick (1, 2).  Exact rational comparison must pick (2, 1).
        self.assertAlmostEqual(5 / 6, 6 / 7, delta=0.05)  # would "tie"
        res = plan_fair_allocation(
            HAND_TRACES, HAND_WEIGHTS, HAND_REFS,
            num_sets=1, total_ways=4, budget=3,
        )
        self.assertEqual(res.allocation, (2, 1))

    def test_fair_can_cost_more_weighted_misses_than_weighted_optimum(self):
        fair = plan_fair_allocation(
            HAND_TRACES, HAND_WEIGHTS, HAND_REFS,
            num_sets=1, total_ways=4, budget=3,
        )
        weighted = plan_allocation(
            HAND_TRACES, HAND_WEIGHTS, num_sets=1, total_ways=4, budget=3,
        )
        self.assertEqual(weighted.allocation, (1, 2))
        self.assertLess(weighted.weighted_misses, fair.weighted_misses)
        worst_weighted = max(
            Fraction(m, r)
            for m, r in zip(weighted.per_app_misses, HAND_REFS)
        )
        self.assertLess(fair.max_ratio, worst_weighted)

    def test_every_app_keeps_at_least_one_way(self):
        res = plan_fair_allocation(
            HAND_TRACES, HAND_WEIGHTS, HAND_REFS,
            num_sets=1, total_ways=4, budget=4,
        )
        self.assertTrue(all(w >= 1 for w in res.allocation))

    def test_max_ratio_apps_lists_all_tied_apps(self):
        # Both apps forced to 1 way; misses 2 and 4 with references 4 and 8
        # give the identical ratio 1/2, so both must be reported.
        traces = [("a", [0, 0, 1, 1]), ("b", [0, 1, 2, 3])]
        res = plan_fair_allocation(
            traces, [1, 1], [4, 8], num_sets=1, total_ways=2, budget=2,
        )
        self.assertEqual(res.allocation, (1, 1))
        self.assertEqual(res.max_ratio, Fraction(1, 2))
        self.assertEqual(res.max_ratio_apps, ("a", "b"))


class ExhaustiveCrossCheckTests(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration(self):
        rng = random.Random(20260926)
        cases = 0
        for _ in range(60):
            n = rng.randint(1, 3)
            num_sets = rng.randint(1, 3)
            total_ways = rng.randint(max(n, 2), 5)
            budget = rng.randint(n, total_ways)
            blocks = [
                [rng.randrange(6) for _ in range(rng.randint(1, 12))]
                for _ in range(n)
            ]
            weights = [rng.randint(1, 5) for _ in range(n)]
            references = [rng.randint(1, 12) for _ in range(n)]
            traces = [(f"app{k}", blk) for k, blk in enumerate(blocks)]

            res = plan_fair_allocation(
                traces, weights, references,
                num_sets=num_sets, total_ways=total_ways, budget=budget,
            )
            (exp_ratio, exp_cost, exp_alloc), _ = ref_fair_brute_force(
                blocks, weights, references, num_sets, total_ways, budget,
            )
            self.assertEqual(res.allocation, exp_alloc)
            self.assertEqual(res.max_ratio, exp_ratio)
            self.assertEqual(res.weighted_misses, exp_cost)
            cases += 1
        self.assertEqual(cases, 60)

    def test_worst_apps_match_reference(self):
        rng = random.Random(7)
        for _ in range(30):
            n = rng.randint(1, 3)
            num_sets = rng.randint(1, 3)
            total_ways = rng.randint(max(n, 2), 5)
            budget = rng.randint(n, total_ways)
            blocks = [
                [rng.randrange(6) for _ in range(rng.randint(1, 12))]
                for _ in range(n)
            ]
            weights = [rng.randint(1, 5) for _ in range(n)]
            references = [rng.randint(1, 12) for _ in range(n)]
            ids = [f"app{k}" for k in range(n)]
            traces = list(zip(ids, blocks))
            res = plan_fair_allocation(
                traces, weights, references,
                num_sets=num_sets, total_ways=total_ways, budget=budget,
            )
            # apps are sorted by id inside the result; align references.
            ref_by_id = dict(zip(ids, references))
            expect = tuple(
                app_id
                for app_id, misses in zip(res.app_ids, res.per_app_misses)
                if Fraction(misses, ref_by_id[app_id]) == res.max_ratio
            )
            self.assertEqual(res.max_ratio_apps, expect)


class FairValidationTests(unittest.TestCase):
    def test_infeasible_budget_is_proved_not_timed_out(self):
        with self.assertRaises(InfeasibleBudgetError):
            plan_fair_allocation(
                HAND_TRACES, HAND_WEIGHTS, HAND_REFS,
                num_sets=1, total_ways=4, budget=1,
            )

    def test_references_must_be_positive_ints(self):
        for bad in ([0, 6], [-1, 6], [7], [7, 6, 6], [7, 2.5], [7, True]):
            with self.assertRaises(ValidationError, msg=f"refs={bad}"):
                plan_fair_allocation(
                    HAND_TRACES, HAND_WEIGHTS, bad,
                    num_sets=1, total_ways=4, budget=3,
                )

    def test_references_error_message_is_locatable(self):
        with self.assertRaises(ValidationError) as ctx:
            plan_fair_allocation(
                HAND_TRACES, HAND_WEIGHTS, [7, 0],
                num_sets=1, total_ways=4, budget=3,
            )
        self.assertIn("references[1]", str(ctx.exception))

    def test_references_follow_input_order_not_sorted_order(self):
        # Supplied as (b, a) with references aligned to that input order;
        # the library sorts internally and references must follow the app.
        traces = [("b", [2, 3, 3, 2, 0, 0, 0, 1]), ("a", [2, 0, 4, 2, 4, 4, 2, 2])]
        res = plan_fair_allocation(
            traces, [3, 1], [6, 7], num_sets=1, total_ways=4, budget=3,
        )
        self.assertEqual(res.app_ids, ("a", "b"))
        self.assertEqual(res.references, (7, 6))
        self.assertEqual(res.allocation, (2, 1))


if __name__ == "__main__":
    unittest.main()
