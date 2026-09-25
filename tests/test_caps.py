"""Tests for hard miss caps, disableable apps, diagnostics and verification.

The brute-force reference in this file is fully independent of the wayplan
core: it uses its own naive list-based LRU simulator and enumerates every
disable subset x every way allocation.  It never calls wayplan to obtain
expected values.
"""

import itertools
import math
import random
import unittest
from dataclasses import replace

from wayplan import (
    InfeasibleBudgetError,
    ValidationError,
    plan_allocation,
    verify_plan,
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


def ref_enumerate(blocks, weights, caps, pens, num_sets, total_ways, budget):
    """Enumerate every disable subset x every way allocation.

    Returns (cost, disabled_count, allocation_tuple) of the best candidate
    under the objective order (cost, disabled count, lexicographic
    allocation with 0 = disabled), or None when nothing is feasible.
    """
    n = len(blocks)
    curves = [
        [ref_lru_misses(bl, num_sets, w) for w in range(1, total_ways + 1)]
        for bl in blocks
    ]
    best = None
    for mask in itertools.product((False, True), repeat=n):
        if any(mask[i] and pens[i] is None for i in range(n)):
            continue  # app not declared disableable
        active = [i for i in range(n) if not mask[i]]
        for ways in itertools.product(range(1, total_ways + 1),
                                      repeat=len(active)):
            if sum(ways) > budget:
                continue
            alloc = [0] * n
            for i, a in zip(active, ways):
                alloc[i] = a
            cost = sum(pens[i] for i in range(n) if mask[i])
            ok = True
            for i in active:
                m = curves[i][alloc[i] - 1]
                if caps[i] is not None and m > caps[i]:
                    ok = False
                    break
                cost += weights[i] * m
            if not ok:
                continue
            cand = (cost, sum(mask), tuple(alloc))
            if best is None or cand < best:
                best = cand
    return best


# ---------------------------------------------------------------------------
# Hand-computed traces shared by several cases (num_sets=1, 4-way curves)
# ---------------------------------------------------------------------------

# appA: alternate 2 blocks, 40 accesses -> curve (40, 2, 2, 2)
BLOCKS_A = [0, 1] * 20
# appB: cycle 3 blocks, 200 accesses -> curve (200, 200, 3, 3)
BLOCKS_B = ([10, 11, 12] * 67)[:200]
# appC: 25 x (20,21) then 17 x (20,21,22) -> curve (101, 51, 3, 3)
BLOCKS_C = ([20, 21] * 25) + ([20, 21, 22] * 17)


class HandComputedCaps(unittest.TestCase):
    def test_curve_shapes_used_below(self):
        r = plan_allocation(
            [("appA", BLOCKS_A), ("appB", BLOCKS_B), ("appC", BLOCKS_C)],
            [1, 1, 1], num_sets=1, total_ways=5, budget=5,
        )
        self.assertEqual(r.curves[0], (40, 2, 2, 2, 2))
        self.assertEqual(r.curves[1], (200, 200, 3, 3, 3))
        self.assertEqual(r.curves[2], (101, 51, 3, 3, 3))

    def test_cap_not_solvable_by_patching_unconstrained_optimum(self):
        # Unconstrained optimum (hand-computed over the curves above):
        #   (1,3,1) = 40+3+101 = 144.
        # Hard cap appA <= 2 forces appA to >= 2 ways.  A local repair of
        # (1,3,1) can only free one way from appB (3->2, +197) giving
        # (2,2,1) = 303.  The true constrained optimum moves the way
        # further: (2,1,2) = 2+200+51 = 253.  The planner must find 253,
        # proving it does not compute-then-patch.
        traces = [("appA", BLOCKS_A), ("appB", BLOCKS_B), ("appC", BLOCKS_C)]
        unc = plan_allocation(traces, [1, 1, 1],
                              num_sets=1, total_ways=5, budget=5)
        self.assertEqual(unc.allocation, (1, 3, 1))
        self.assertEqual(unc.objective, 144)

        r = plan_allocation(traces, [1, 1, 1],
                            num_sets=1, total_ways=5, budget=5,
                            miss_caps=[2, None, None])
        self.assertEqual(r.allocation, (2, 1, 2))
        self.assertEqual(r.objective, 253)
        self.assertEqual(r.weighted_misses, 253)
        # The cap is met with equality (exactly at the boundary).
        self.assertEqual(r.per_app_misses[0], 2)
        # The patched-unconstrained answer would be strictly worse.
        self.assertLess(r.objective, 303)

    def test_cap_exactly_met_boundary_is_feasible(self):
        # appA capped at exactly its 2-way miss count: boundary is inclusive.
        r = plan_allocation(
            [("appA", BLOCKS_A), ("appB", BLOCKS_B)], [1, 1],
            num_sets=1, total_ways=4, budget=4, miss_caps=[2, None],
        )
        self.assertEqual(r.allocation, (2, 1))
        self.assertEqual(r.per_app_misses, (2, 200))
        self.assertEqual(r.objective, 202)

    def test_unachievable_cap_forces_disable_when_declared(self):
        # appB's best achievable is 3 misses; cap 2 is unreachable, so the
        # only feasible plans disable appB (penalty 100).
        r = plan_allocation(
            [("appA", BLOCKS_A), ("appB", BLOCKS_B)], [1, 1],
            num_sets=1, total_ways=4, budget=4,
            miss_caps=[None, 2], disable_penalties=[None, 100],
        )
        self.assertEqual(r.allocation, (2, 0))
        self.assertEqual(r.disabled, (False, True))
        self.assertEqual(r.disabled_app_ids, ("appB",))
        self.assertEqual(r.per_app_misses, (2, None))
        self.assertEqual(r.weighted_misses, 2)
        self.assertEqual(r.disable_penalty_total, 100)
        self.assertEqual(r.objective, 102)
        self.assertEqual(r.used_ways, 2)
        self.assertEqual(r.allocation_map(), {"appA": 2, "appB": 0})

    def test_unachievable_cap_without_disable_is_infeasible(self):
        with self.assertRaises(InfeasibleBudgetError) as ctx:
            plan_allocation(
                [("appA", BLOCKS_A), ("appB", BLOCKS_B)], [1, 1],
                num_sets=1, total_ways=4, budget=4, miss_caps=[None, 2],
            )
        e = ctx.exception
        self.assertEqual(e.app_ids, ("appA", "appB"))
        self.assertEqual(e.min_ways, (1, None))  # appB cap unreachable
        self.assertIsNone(e.required_total)
        self.assertEqual(e.budget, 4)
        self.assertIn("'appB'", str(e))
        self.assertIn("unachievable", str(e))

    def test_disable_penalty_tie_prefers_keeping_app_active(self):
        # Flat curve: active costs 4 at any way count; disabling also costs
        # 4.  Equal objective -> fewer disabled apps wins -> stays active.
        r = plan_allocation(
            [("solo", [0, 1, 2, 3])], [1],
            num_sets=4, total_ways=4, budget=2, disable_penalties=[4],
        )
        self.assertEqual(r.allocation, (1,))
        self.assertEqual(r.disabled, (False,))
        self.assertEqual(r.objective, 4)

    def test_disable_set_tie_broken_by_allocation_lexicographic(self):
        # Two identical flat-curve apps, active cost 4 = penalty 4, budget 1
        # forces exactly one disabled.  (0,1) and (1,0) tie on cost and
        # disabled count; the lexicographically smaller (0,1) wins.
        traces = [("appA", [0, 1, 2, 3]), ("appB", [4, 5, 6, 7])]
        r = plan_allocation(
            traces, [1, 1], num_sets=4, total_ways=2, budget=1,
            disable_penalties=[4, 4],
        )
        self.assertEqual(r.allocation, (0, 1))
        self.assertEqual(r.disabled_app_ids, ("appA",))
        self.assertEqual(r.objective, 8)

    def test_zero_budget_disables_everything(self):
        traces = [("a", [0, 1]), ("b", [2, 3])]
        r = plan_allocation(
            traces, [1, 1], num_sets=2, total_ways=4, budget=0,
            disable_penalties=[3, 7],
        )
        self.assertEqual(r.allocation, (0, 0))
        self.assertEqual(r.used_ways, 0)
        self.assertEqual(r.weighted_misses, 0)
        self.assertEqual(r.disable_penalty_total, 10)
        self.assertEqual(r.objective, 10)
        self.assertEqual(r.disabled_app_ids, ("a", "b"))
        self.assertEqual(r.per_app_misses, (None, None))

    def test_zero_budget_with_mandatory_app_is_infeasible(self):
        traces = [("a", [0, 1]), ("b", [2, 3])]
        with self.assertRaises(InfeasibleBudgetError) as ctx:
            plan_allocation(
                traces, [1, 1], num_sets=2, total_ways=4, budget=0,
                disable_penalties=[3, None],  # b may not be disabled
            )
        e = ctx.exception
        self.assertEqual(e.min_ways, (0, 1))
        self.assertEqual(e.required_total, 1)
        self.assertEqual(e.budget, 0)

    def test_caps_follow_input_order_not_sorted_order(self):
        # Traces given unsorted; the cap 2 belongs to "z" (first in input).
        # z: [0,1]*4 -> curve (8, 2): needs 2 ways to meet cap 2.
        # a: [9]*5   -> curve (1, 1): always fine.
        # Budget 2 < 1+2 -> infeasible iff the cap really tracks z.
        traces = [("z", [0, 1] * 4), ("a", [9] * 5)]
        with self.assertRaises(InfeasibleBudgetError) as ctx:
            plan_allocation(traces, [1, 1], num_sets=1, total_ways=3,
                            budget=2, miss_caps=[2, None])
        self.assertEqual(ctx.exception.min_ways, (1, 2))  # sorted: a, z
        # Same call without the cap is feasible.
        r = plan_allocation(traces, [1, 1], num_sets=1, total_ways=3,
                            budget=2)
        self.assertEqual(r.allocation, (1, 1))

    def test_diagnostics_for_plain_budget_shortfall(self):
        traces = [("c", [3]), ("a", [1]), ("b", [2])]
        with self.assertRaises(InfeasibleBudgetError) as ctx:
            plan_allocation(traces, [1, 1, 1],
                            num_sets=1, total_ways=4, budget=2)
        e = ctx.exception
        self.assertEqual(e.app_ids, ("a", "b", "c"))
        self.assertEqual(e.min_ways, (1, 1, 1))
        self.assertEqual(e.required_total, 3)
        self.assertEqual(e.budget, 2)
        self.assertIn("necessary", str(e))  # caveat is stated, not hidden

    def test_defaults_preserve_old_semantics(self):
        r = plan_allocation([("a", [0, 1]), ("b", [2])], [2, 1],
                            num_sets=2, total_ways=3, budget=2)
        self.assertEqual(r.miss_caps, (None, None))
        self.assertEqual(r.disable_penalties, (None, None))
        self.assertEqual(r.disabled, (False, False))
        self.assertEqual(r.disabled_app_ids, ())
        self.assertEqual(r.disable_penalty_total, 0)
        self.assertEqual(r.objective, r.weighted_misses)


# ---------------------------------------------------------------------------
# Brute-force cross-check: enumeration of disable sets x all allocations
# ---------------------------------------------------------------------------

class BruteForceCapsCrossCheck(unittest.TestCase):
    def test_dp_matches_disable_set_enumeration(self):
        rng = random.Random(20260925)
        feasible = infeasible = zero_budget = 0
        for _ in range(50):
            n = rng.randint(1, 4)
            num_sets = rng.randint(1, 4)
            total_ways = rng.randint(1, 5)
            budget = rng.randint(0, total_ways)
            traces, blocks_only, weights, caps, pens = [], [], [], [], []
            for i in range(n):
                length = rng.randint(1, 30)
                pool = rng.randint(1, num_sets * 3)
                blocks = [rng.randrange(pool) for _ in range(length)]
                traces.append((f"app{i}", blocks))
                blocks_only.append(blocks)
                weights.append(rng.randint(1, 4))
                caps.append(None if rng.random() < 0.5
                            else rng.randint(0, length))
                pens.append(None if rng.random() < 0.5
                            else rng.randint(0, 40))
            exp = ref_enumerate(blocks_only, weights, caps, pens,
                                num_sets, total_ways, budget)
            if exp is None:
                with self.assertRaises(InfeasibleBudgetError):
                    plan_allocation(
                        traces, weights, num_sets, total_ways, budget,
                        miss_caps=caps, disable_penalties=pens,
                    )
                infeasible += 1
                continue
            exp_cost, exp_ndis, exp_alloc = exp
            r = plan_allocation(
                traces, weights, num_sets, total_ways, budget,
                miss_caps=caps, disable_penalties=pens,
            )
            self.assertEqual(r.objective, exp_cost)
            self.assertEqual(r.allocation, exp_alloc)
            self.assertEqual(sum(r.disabled), exp_ndis)
            self.assertEqual(r.used_ways, sum(exp_alloc))
            self.assertLessEqual(r.used_ways, budget)
            # Every hard cap holds at the planned allocation.
            for i in range(n):
                if exp_alloc[i] > 0 and caps[i] is not None:
                    self.assertLessEqual(r.per_app_misses[i], caps[i])
            # Independent replay confirms every hard constraint.
            self.assertTrue(verify_plan(traces, r).ok)
            feasible += 1
            zero_budget += budget == 0
        self.assertGreaterEqual(feasible, 20)   # loop really exercised both
        self.assertGreaterEqual(infeasible, 5)  # branches, not just one


# ---------------------------------------------------------------------------
# verify_plan: replay confirms every hard constraint
# ---------------------------------------------------------------------------

class VerifyPlan(unittest.TestCase):
    TRACES = [("appA", BLOCKS_A), ("appB", BLOCKS_B), ("appC", BLOCKS_C)]

    def plan(self, **kw):
        args = dict(num_sets=1, total_ways=4, budget=4,
                    miss_caps=[2, None, None],
                    disable_penalties=[None, 50, None])
        args.update(kw)
        return plan_allocation(self.TRACES, [1, 1, 1], **args)

    def test_valid_plan_passes_every_check(self):
        r = self.plan()
        self.assertEqual(r.disabled_app_ids, ("appB",))
        report = verify_plan(self.TRACES, r)
        self.assertTrue(report.ok)
        self.assertEqual([c.name for c in report.checks],
                         ["allocation_shape", "budget_respected",
                          "disable_rules", "caps_respected",
                          "replay_matches_plan", "objective_arithmetic"])
        self.assertTrue(all(c.ok for c in report.checks))
        # Disabled app is reported separately, not replayed.
        self.assertEqual(report.disabled_app_ids, ("appB",))
        self.assertEqual([s.app_id for s in report.replay], ["appA", "appC"])
        for s in report.replay:
            self.assertEqual(s.hits + s.misses, s.accesses)
            self.assertTrue(0.0 <= s.hit_rate <= 1.0)

    def test_tampered_cap_is_caught_by_replay(self):
        r = self.plan()
        # Lie about the cap: claim appA may miss 0 times.
        tampered = replace(r, miss_caps=(0, None, None))
        report = verify_plan(self.TRACES, tampered)
        self.assertFalse(report.ok)
        by_name = {c.name: c for c in report.checks}
        self.assertFalse(by_name["caps_respected"].ok)
        self.assertTrue(by_name["replay_matches_plan"].ok)

    def test_tampered_objective_is_caught(self):
        r = self.plan()
        report = verify_plan(self.TRACES, replace(r, objective=r.objective + 1))
        self.assertFalse(report.ok)
        by_name = {c.name: c for c in report.checks}
        self.assertFalse(by_name["objective_arithmetic"].ok)

    def test_tampered_allocation_breaks_budget_check(self):
        r = self.plan()  # budget 4
        tampered = replace(r, allocation=(4, 0, 1), used_ways=5)
        report = verify_plan(self.TRACES, tampered)
        self.assertFalse(report.ok)
        by_name = {c.name: c for c in report.checks}
        self.assertFalse(by_name["budget_respected"].ok)

    def test_verify_rejects_structural_contract_violations(self):
        r = self.plan()
        with self.assertRaises(ValidationError):
            verify_plan(self.TRACES, "not-a-plan")
        with self.assertRaises(ValidationError):  # app ids do not match
            verify_plan([("x", [1]), ("y", [2]), ("z", [3])], r)


# ---------------------------------------------------------------------------
# Validation of the new parameters
# ---------------------------------------------------------------------------

class CapsDisableValidation(unittest.TestCase):
    TRACES = [("a", [0, 1, 0]), ("b", [2, 3])]

    def plan(self, **kw):
        args = dict(num_sets=2, total_ways=4, budget=3)
        args.update(kw)
        return plan_allocation(self.TRACES, [1, 2], **args)

    def test_miss_caps_contract(self):
        for bad, needle in [
            ([1], "miss_caps"),                # wrong length
            ([-1, None], "miss_caps[0]"),      # negative
            ([True, None], "miss_caps[0]"),    # bool
            ([math.nan, None], "miss_caps[0]"),
            ([1.5, None], "miss_caps[0]"),
            ("12", "miss_caps"),               # wrong type
            (5, "miss_caps"),
        ]:
            with self.assertRaises(ValidationError) as ctx:
                self.plan(miss_caps=bad)
            self.assertIn(needle, str(ctx.exception))

    def test_disable_penalties_contract(self):
        for bad, needle in [
            ([1], "disable_penalties"),
            ([None, -5], "disable_penalties[1]"),
            ([None, False], "disable_penalties[1]"),
            ([None, math.inf], "disable_penalties[1]"),
            ([None, 2.5], "disable_penalties[1]"),
            ("3", "disable_penalties"),
        ]:
            with self.assertRaises(ValidationError) as ctx:
                self.plan(disable_penalties=bad)
            self.assertIn(needle, str(ctx.exception))

    def test_zero_cap_and_zero_penalty_are_valid(self):
        # cap 0: app can never be active (a cold trace always misses), so it
        # must be disabled; penalty 0 makes that free.
        r = self.plan(miss_caps=[0, None], disable_penalties=[0, None])
        self.assertEqual(r.disabled, (True, False))
        self.assertEqual(r.disable_penalty_total, 0)
        self.assertEqual(r.allocation[1], 1)  # b keeps its mandatory way

    def test_failed_call_leaves_no_partial_state(self):
        with self.assertRaises(ValidationError):
            self.plan(miss_caps=[math.nan, None])
        r = self.plan()  # library holds no mutable module-level state
        self.assertEqual(r.objective, r.weighted_misses)


if __name__ == "__main__":
    unittest.main()
