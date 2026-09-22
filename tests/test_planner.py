"""Planner tests: hand-checked cases, brute-force cross-check, edge cases.

The brute-force reference in this file is fully independent of the wayplan
core: it uses its own list-based LRU simulator and enumerates every legal
allocation.  It never calls wayplan to obtain expected values.
"""

import itertools
import random
import unittest

from wayplan import (
    InfeasibleBudgetError,
    ValidationError,
    plan_allocation,
    replay_isolated,
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


def ref_brute_force(traces_blocks, weights, num_sets, total_ways, budget):
    """Enumerate every legal allocation; return (cost, allocation_tuple).

    Tie-break: lexicographically smallest allocation vector, matching the
    documented contract, implemented here by plain tuple comparison.
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
        cost = sum(weights[i] * curves[i][alloc[i] - 1] for i in range(n))
        cand = (cost, alloc)
        if best is None or cand < best:
            best = cand
    return best


# ---------------------------------------------------------------------------
# Hand-computed planning cases
# ---------------------------------------------------------------------------

class HandComputedPlanning(unittest.TestCase):
    def test_weighted_choice_by_hand(self):
        # num_sets=1.
        #   appA: [0,1]*4 -> curve (8, 2) for ways 1..2
        #   appB: [7]*5   -> curve (1, 1)
        # weights (3, 1):
        #   budget=2 forces (1,1): cost 3*8 + 1*1 = 25
        #   budget=3: (1,2)=25 vs (2,1)=3*2+1=7 -> (2,1) wins
        traces = [("appA", [0, 1] * 4), ("appB", [7] * 5)]
        r2 = plan_allocation(traces, [3, 1], num_sets=1, total_ways=3, budget=2)
        self.assertEqual(r2.allocation, (1, 1))
        self.assertEqual(r2.weighted_misses, 25)
        r3 = plan_allocation(traces, [3, 1], num_sets=1, total_ways=3, budget=3)
        self.assertEqual(r3.allocation, (2, 1))
        self.assertEqual(r3.weighted_misses, 7)
        self.assertEqual(r3.per_app_misses, (2, 1))
        self.assertEqual(r3.used_ways, 3)

    def test_extra_ways_with_no_benefit_are_not_allocated(self):
        # Both apps touch blocks in distinct sets: cold misses only, curves
        # perfectly flat.  Budget 5 must NOT be consumed; the lexicographically
        # smallest optimal vector (1, 1) is returned.
        traces = [("appA", [0, 1, 2, 3]), ("appB", [0, 1])]
        r = plan_allocation(traces, [1, 1], num_sets=4, total_ways=6, budget=5)
        self.assertEqual(r.curves[0], (4, 4, 4, 4, 4, 4))
        self.assertEqual(r.curves[1], (2, 2, 2, 2, 2, 2))
        self.assertEqual(r.allocation, (1, 1))
        self.assertEqual(r.used_ways, 2)
        self.assertEqual(r.weighted_misses, 6)

    def test_tie_break_is_lexicographic_by_app_id(self):
        # Identical flat curves, weights 1: every allocation costs the same.
        # Apps are ordered by id, so ("a", "b") -> smallest vector (1, 1).
        # Passing traces in reverse input order must not change the result.
        traces = [("b", [0, 1]), ("a", [0, 1])]
        r = plan_allocation(traces, [1, 1], num_sets=4, total_ways=4, budget=3)
        self.assertEqual(r.app_ids, ("a", "b"))
        self.assertEqual(r.allocation, (1, 1))

    def test_weights_follow_input_order_not_sorted_order(self):
        # Traces given unsorted; weight 5 belongs to "z" (first in input).
        # z: [0,1]*4 -> (8, 2); a: [9]*5 -> (1, 1).  Budget 3.
        # (z=1,a=2): 5*8+1=41 ; (z=2,a=1): 5*2+1=11 -> z gets the extra way.
        traces = [("z", [0, 1] * 4), ("a", [9] * 5)]
        r = plan_allocation(traces, [5, 1], num_sets=1, total_ways=3, budget=3)
        alloc = r.allocation_map()
        self.assertEqual(alloc, {"a": 1, "z": 2})
        self.assertEqual(r.weighted_misses, 11)

    def test_greedy_counterexample(self):
        # Non-convex curves defeat marginal-gain greedy (hand-computed):
        #   appA: 20 x (0 x9, 1) -> curve (40, 2, 2, ...) [gain 38 at way 2]
        #   appB: [10,11,12] cycled -> curve (200, 200, 3, ...) [gain only at 3]
        # budget=4, weights (1,1).
        # Greedy: (1,1) -> +A (gain 38) -> (2,1) -> tie 0/0, +A -> (3,1),
        #         cost 2 + 200 = 202.
        # Optimal: (1,3), cost 40 + 3 = 43.
        blocks_a = ([0] * 9 + [1]) * 20
        blocks_b = ([10, 11, 12] * 67)[:200]
        traces = [("appA", blocks_a), ("appB", blocks_b)]
        r = plan_allocation(traces, [1, 1], num_sets=1, total_ways=4, budget=4)
        self.assertEqual(r.curves[0], (40, 2, 2, 2))
        self.assertEqual(r.curves[1], (200, 200, 3, 3))
        self.assertEqual(r.allocation, (1, 3))
        self.assertEqual(r.weighted_misses, 43)
        # The greedy marginal-benefit answer would be (3, 1) at cost 202.
        self.assertNotEqual(r.allocation, (3, 1))
        self.assertLess(r.weighted_misses, 202)

    def test_infeasible_budget_is_proved_not_timed_out(self):
        traces = [("a", [1]), ("b", [2]), ("c", [3])]
        with self.assertRaises(InfeasibleBudgetError) as ctx:
            plan_allocation(traces, [1, 1, 1], num_sets=1, total_ways=4, budget=2)
        self.assertIn("budget", str(ctx.exception))
        # Infeasibility is a distinct outcome from invalid input.
        self.assertNotIsInstance(ctx.exception, ValidationError)


# ---------------------------------------------------------------------------
# Brute-force cross-check on deterministic random instances
# ---------------------------------------------------------------------------

class BruteForceCrossCheck(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration(self):
        rng = random.Random(20260922)
        cases = 0
        for _ in range(60):
            n = rng.randint(1, 6)
            num_sets = rng.randint(1, 6)
            total_ways = rng.randint(n, 8)  # always room for 1 way per app
            budget = rng.randint(n, total_ways)
            traces, blocks_only, weights = [], [], []
            for i in range(n):
                length = rng.randint(1, 40)
                pool = rng.randint(1, num_sets * 4)
                blocks = [rng.randrange(pool) for _ in range(length)]
                traces.append((f"app{i}", blocks))
                blocks_only.append(blocks)
                weights.append(rng.randint(1, 5))
            result = plan_allocation(
                traces, weights, num_sets, total_ways, budget
            )
            exp_cost, exp_alloc = ref_brute_force(
                blocks_only, weights, num_sets, total_ways, budget
            )
            self.assertEqual(result.weighted_misses, exp_cost)
            self.assertEqual(result.allocation, exp_alloc)
            # Curves must match the independent simulator exactly.
            for i, blocks in enumerate(blocks_only):
                for w in range(1, total_ways + 1):
                    self.assertEqual(
                        result.curves[i][w - 1],
                        ref_lru_misses(blocks, num_sets, w),
                    )
            cases += 1
        self.assertGreaterEqual(cases, 50)  # make sure the loop really ran


# ---------------------------------------------------------------------------
# Replay consistency (supplementary round-trip check)
# ---------------------------------------------------------------------------

class ReplayConsistency(unittest.TestCase):
    def test_replay_misses_match_planned_curves(self):
        rng = random.Random(7)
        traces = [
            (f"app{i}", [rng.randrange(12) for _ in range(50)]) for i in range(3)
        ]
        r = plan_allocation(traces, [2, 1, 3], num_sets=3, total_ways=6, budget=5)
        stats = replay_isolated(traces, r.allocation_map(), num_sets=3)
        self.assertEqual([s.app_id for s in stats], ["app0", "app1", "app2"])
        for i, s in enumerate(stats):
            self.assertEqual(s.ways, r.allocation[i])
            self.assertEqual(s.misses, r.per_app_misses[i])
            self.assertEqual(s.accesses, 50)
            self.assertEqual(s.hits + s.misses, s.accesses)
            self.assertEqual(s.hit_rate, s.hits / s.accesses)
            self.assertTrue(0.0 <= s.hit_rate <= 1.0)

    def test_replay_validates_allocation(self):
        traces = [("a", [1]), ("b", [2])]
        with self.assertRaises(ValidationError):  # missing app
            replay_isolated(traces, {"a": 1}, num_sets=1)
        with self.assertRaises(ValidationError):  # unknown app
            replay_isolated(traces, {"a": 1, "b": 1, "c": 1}, num_sets=1)
        with self.assertRaises(ValidationError):  # zero ways
            replay_isolated(traces, {"a": 1, "b": 0}, num_sets=1)
        with self.assertRaises(ValidationError):  # bool ways
            replay_isolated(traces, {"a": 1, "b": True}, num_sets=1)


if __name__ == "__main__":
    unittest.main()
