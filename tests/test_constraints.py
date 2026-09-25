"""Tests for hard miss caps, paid deactivation, and plan verification (v1.1).

The brute-force reference in this file is fully independent of the wayplan
core: it uses its own list-based LRU simulator and enumerates every
deactivation set x every way allocation.  It never calls wayplan to obtain
expected values.  All hand-computed curves below are for num_sets=1 (every
block in the same set) unless stated otherwise.
"""

import itertools
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


def ref_brute_force(blocks, weights, num_sets, total_ways, budget, caps, pens):
    """Enumerate every deactivation set x every way allocation.

    Way count 0 means "deactivated" and is legal only where a penalty was
    declared.  Returns (objective, deactivation_count, allocation_tuple)
    or None when no candidate satisfies every hard cap within budget.
    Tie-break matches the documented contract via plain tuple comparison.
    """
    n = len(blocks)
    curves = [
        [ref_lru_misses(bl, num_sets, w) for w in range(1, total_ways + 1)]
        for bl in blocks
    ]
    best = None
    for alloc in itertools.product(range(0, total_ways + 1), repeat=n):
        if sum(alloc) > budget:
            continue
        cost = 0
        deact = 0
        feasible = True
        for i in range(n):
            a = alloc[i]
            if a == 0:
                if pens[i] is None:
                    feasible = False
                    break
                cost += pens[i]
                deact += 1
            else:
                if caps[i] is not None and curves[i][a - 1] > caps[i]:
                    feasible = False
                    break
                cost += weights[i] * curves[i][a - 1]
        if not feasible:
            continue
        cand = (cost, deact, alloc)
        if best is None or cand < best:
            best = cand
    return best


# Hand-built traces (num_sets=1 throughout this file's hand cases):
#   THRASH3: [0,1,2] cycled   -> curve (12, 12, 3, 3) for ways 1..4
#   THRASH4: [5,6,7,8] cycled -> curve (12, 12, 12, 4)
THRASH3 = [0, 1, 2] * 4
THRASH4 = [5, 6, 7, 8] * 3


class HandComputedConstraints(unittest.TestCase):
    def test_hard_cap_binds_and_changes_allocation(self):
        # aCrit: (12,12,3,3); bBulk: (12,12,12,4); weights (1,1), budget 5.
        # Unconstrained optimum: (3,1) cost 15.  Cap bBulk <= 4 forces
        # bBulk to 4 ways, leaving 1 for aCrit: (1,4) cost 16.
        traces = [("aCrit", THRASH3), ("bBulk", THRASH4)]
        plain = plan_allocation(traces, [1, 1], num_sets=1, total_ways=5, budget=5)
        self.assertEqual(plain.curves[0], (12, 12, 3, 3, 3))
        self.assertEqual(plain.curves[1], (12, 12, 12, 4, 4))
        self.assertEqual(plain.allocation, (3, 1))
        self.assertEqual(plain.objective, 15)

        capped = plan_allocation(
            traces, [1, 1], num_sets=1, total_ways=5, budget=5,
            miss_caps=[None, 4],
        )
        self.assertEqual(capped.allocation, (1, 4))
        self.assertEqual(capped.per_app_misses, (12, 4))
        self.assertEqual(capped.objective, 16)
        self.assertEqual(capped.weighted_misses, 16)
        self.assertEqual(capped.deactivation_count, 0)
        # The cap is a hard constraint: the unconstrained optimum (3,1)
        # would give bBulk 12 misses > cap 4 and is therefore infeasible.
        self.assertNotEqual(capped.allocation, plain.allocation)

    def test_deactivation_beats_keeping(self):
        # Adds cBatch ([9,10]*4 -> (8,2,2,2), weight 1, penalty 5) to the
        # capped scenario above, budget 6.
        #   keep cBatch:    (1,4,1) cost 12+4+8 = 24
        #   deactivate it:  (1,4,0) cost 12+4+5 = 21  <- wins
        traces = [("aCrit", THRASH3), ("bBulk", THRASH4), ("cBatch", [9, 10] * 4)]
        r = plan_allocation(
            traces, [1, 1, 1], num_sets=1, total_ways=6, budget=6,
            miss_caps=[None, 4, None],
            deactivation_penalties=[None, None, 5],
        )
        self.assertEqual(r.allocation, (1, 4, 0))
        self.assertEqual(r.deactivated, (False, False, True))
        self.assertEqual(r.deactivated_app_ids(), ("cBatch",))
        self.assertEqual(r.deactivation_count, 1)
        self.assertEqual(r.deactivation_cost, 5)
        self.assertEqual(r.weighted_misses, 16)
        self.assertEqual(r.objective, 21)
        self.assertEqual(r.used_ways, 5)
        self.assertIsNone(r.per_app_misses[2])
        self.assertEqual(r.allocation_map(), {"aCrit": 1, "bBulk": 4, "cBatch": 0})

    def test_keep_vs_deactivate_tie_prefers_keeping(self):
        # solo: [0,1,2]*2 -> (6,6,3); budget 1 forces 1 way (6 misses).
        # Penalty exactly 6 ties the objective; fewer deactivations wins.
        traces = [("solo", [0, 1, 2] * 2)]
        r = plan_allocation(
            traces, [1], num_sets=1, total_ways=3, budget=1,
            deactivation_penalties=[6],
        )
        self.assertEqual(r.allocation, (1,))
        self.assertEqual(r.deactivation_count, 0)
        self.assertEqual(r.objective, 6)

    def test_deactivate_either_tie_breaks_lexicographically(self):
        # p, q: flat curves (4,4), penalty 4 each; r mandatory, curve (1,1).
        # Budget 2 fits r plus exactly one of p/q; the other is deactivated.
        #   (p=1,q=0,r=1): cost 4+4+1 = 9, 1 deactivation
        #   (p=0,q=1,r=1): cost 4+4+1 = 9, 1 deactivation
        # Tied on objective and count; 0 sorts before 1, so (0,1,1) wins.
        traces = [("p", [0, 1, 2, 3]), ("q", [4, 5, 6, 7]), ("r", [8])]
        r = plan_allocation(
            traces, [1, 1, 1], num_sets=1, total_ways=2, budget=2,
            deactivation_penalties=[4, 4, None],
        )
        self.assertEqual(r.allocation, (0, 1, 1))
        self.assertEqual(r.deactivated_app_ids(), ("p",))
        self.assertEqual(r.objective, 9)

    def test_zero_budget_deactivates_everything_declared(self):
        traces = [("x", [0, 1]), ("y", [2, 3])]
        r = plan_allocation(
            traces, [1, 1], num_sets=1, total_ways=2, budget=0,
            deactivation_penalties=[3, 7],
        )
        self.assertEqual(r.allocation, (0, 0))
        self.assertEqual(r.used_ways, 0)
        self.assertEqual(r.weighted_misses, 0)
        self.assertEqual(r.deactivation_cost, 10)
        self.assertEqual(r.objective, 10)
        self.assertEqual(r.deactivation_count, 2)

    def test_zero_budget_with_mandatory_app_is_infeasible(self):
        traces = [("x", [0, 1]), ("y", [2, 3])]
        with self.assertRaises(InfeasibleBudgetError):
            plan_allocation(
                traces, [1, 1], num_sets=1, total_ways=2, budget=0,
                deactivation_penalties=[3, None],  # y may not be deactivated
            )

    def test_exactly_at_cap_is_feasible(self):
        # aCrit curve (12,12,3): cap 3 is met exactly at 3 ways.
        traces = [("aCrit", THRASH3)]
        r = plan_allocation(
            traces, [1], num_sets=1, total_ways=3, budget=3,
            miss_caps=[3],
        )
        self.assertEqual(r.allocation, (3,))
        self.assertEqual(r.per_app_misses, (3,))
        self.assertEqual(r.objective, 3)
        # One miss above the cap is already unachievable -> infeasible.
        with self.assertRaises(InfeasibleBudgetError):
            plan_allocation(
                traces, [1], num_sets=1, total_ways=3, budget=3,
                miss_caps=[2],
            )

    def test_unachievable_cap_forces_deactivation_or_infeasibility(self):
        traces = [("aCrit", THRASH3)]
        # Cap 2 can never be met (curve bottoms out at 3).  Declared
        # deactivatable with penalty 100 -> deactivated, objective 100.
        r = plan_allocation(
            traces, [1], num_sets=1, total_ways=3, budget=3,
            miss_caps=[2], deactivation_penalties=[100],
        )
        self.assertEqual(r.allocation, (0,))
        self.assertEqual(r.objective, 100)
        # Not deactivatable -> proved infeasible, with diagnostics.
        with self.assertRaises(InfeasibleBudgetError) as ctx:
            plan_allocation(
                traces, [1], num_sets=1, total_ways=3, budget=3,
                miss_caps=[2],
            )
        err = ctx.exception
        self.assertIn("budget", str(err))
        self.assertIn("unachievable", str(err))
        diag = err.diagnostics
        self.assertEqual(diag["budget"], 3)
        self.assertEqual(diag["total_ways"], 3)
        self.assertEqual(len(diag["per_app"]), 1)
        entry = diag["per_app"][0]
        self.assertEqual(entry["app_id"], "aCrit")
        self.assertIsNone(entry["min_ways_if_active"])
        self.assertFalse(entry["deactivatable"])

    def test_diagnostics_report_per_app_minima_and_budget(self):
        # aCrit needs >= 3 ways (cap 3), bBulk needs 4 (cap 4): sum 7 > 6.
        traces = [("aCrit", THRASH3), ("bBulk", THRASH4)]
        with self.assertRaises(InfeasibleBudgetError) as ctx:
            plan_allocation(
                traces, [1, 1], num_sets=1, total_ways=6, budget=6,
                miss_caps=[3, 4],
            )
        diag = ctx.exception.diagnostics
        self.assertEqual(diag["budget"], 6)
        self.assertEqual(
            [e["min_ways_if_active"] for e in diag["per_app"]], [3, 4]
        )
        self.assertIn("aCrit: >= 3 way(s)", str(ctx.exception))
        self.assertIn("bBulk: >= 4 way(s)", str(ctx.exception))


class VerifyPlan(unittest.TestCase):
    def _constrained_plan(self):
        traces = [("aCrit", THRASH3), ("bBulk", THRASH4), ("cBatch", [9, 10] * 4)]
        result = plan_allocation(
            traces, [1, 1, 1], num_sets=1, total_ways=6, budget=6,
            miss_caps=[None, 4, None],
            deactivation_penalties=[None, None, 5],
        )
        return traces, result

    def test_verify_confirms_every_hard_constraint(self):
        traces, result = self._constrained_plan()
        ver = verify_plan(result, traces)
        self.assertTrue(ver.ok, ver.failures)
        self.assertEqual(ver.failures, ())
        self.assertEqual(ver.deactivated, ("cBatch",))
        self.assertEqual([s.app_id for s in ver.replay], ["aCrit", "bBulk"])
        for s in ver.replay:
            i = result.app_ids.index(s.app_id)
            self.assertEqual(s.ways, result.allocation[i])
            self.assertEqual(s.misses, result.per_app_misses[i])
            cap = result.miss_caps[i]
            if cap is not None:
                self.assertLessEqual(s.misses, cap)
            self.assertEqual(s.hit_rate, s.hits / s.accesses)

    def test_verify_detects_cap_violation_by_replay(self):
        traces, result = self._constrained_plan()
        # Tamper: move bBulk to 1 way (12 misses > cap 4) while keeping the
        # recorded state otherwise plausible.
        tampered = replace(
            result,
            allocation=(1, 1, 0),
            per_app_misses=(12, 12, None),
            used_ways=2,
            weighted_misses=24,
            objective=29,
        )
        ver = verify_plan(tampered, traces)
        self.assertFalse(ver.ok)
        self.assertTrue(any("cap" in f for f in ver.failures), ver.failures)

    def test_verify_detects_undeclared_deactivation(self):
        traces, result = self._constrained_plan()
        # Tamper: deactivate bBulk, which has no declared penalty.
        tampered = replace(
            result,
            allocation=(1, 0, 1),
            deactivated=(False, True, True),
            per_app_misses=(12, None, None),
            deactivation_count=2,
            deactivation_cost=5,
            used_ways=2,
            weighted_misses=12,
            objective=17,
        )
        ver = verify_plan(tampered, traces)
        self.assertFalse(ver.ok)
        self.assertTrue(
            any("no deactivation penalty" in f for f in ver.failures),
            ver.failures,
        )

    def test_verify_rejects_structural_mismatch(self):
        traces, result = self._constrained_plan()
        with self.assertRaises(ValidationError):
            verify_plan(result, [("other", [1])])
        with self.assertRaises(ValidationError):
            verify_plan("not a plan", traces)


# ---------------------------------------------------------------------------
# Brute-force cross-check on deterministic random instances
# ---------------------------------------------------------------------------

class BruteForceCrossCheck(unittest.TestCase):
    def test_dp_matches_enumeration_with_caps_and_deactivation(self):
        rng = random.Random(20260925)
        feasible_cases = infeasible_cases = 0
        for _ in range(40):
            n = rng.randint(1, 4)
            num_sets = rng.randint(1, 4)
            total_ways = rng.randint(1, 5)
            budget = rng.randint(0, total_ways)  # zero budget included
            blocks_only, weights = [], []
            for i in range(n):
                length = rng.randint(1, 30)
                pool = rng.randint(1, num_sets * 3)
                blocks_only.append([rng.randrange(pool) for _ in range(length)])
                weights.append(rng.randint(1, 4))
            traces = [(f"app{i}", bl) for i, bl in enumerate(blocks_only)]
            # Caps: none at all, or per-app None / small int (0 included,
            # which is unachievable for any active app).
            if rng.random() < 0.4:
                caps = None
                caps_list = [None] * n
            else:
                caps_list = [
                    None if rng.random() < 0.5 else rng.randint(0, 12)
                    for _ in range(n)
                ]
                caps = caps_list
            # Penalties: none at all, or per-app None / small int.
            if rng.random() < 0.4:
                pens = None
                pens_list = [None] * n
            else:
                pens_list = [
                    None if rng.random() < 0.5 else rng.randint(0, 15)
                    for _ in range(n)
                ]
                pens = pens_list

            expected = ref_brute_force(
                blocks_only, weights, num_sets, total_ways, budget,
                caps_list, pens_list,
            )
            if expected is None:
                with self.assertRaises(InfeasibleBudgetError):
                    plan_allocation(
                        traces, weights, num_sets, total_ways, budget,
                        miss_caps=caps, deactivation_penalties=pens,
                    )
                infeasible_cases += 1
                continue
            exp_obj, exp_deact, exp_alloc = expected
            result = plan_allocation(
                traces, weights, num_sets, total_ways, budget,
                miss_caps=caps, deactivation_penalties=pens,
            )
            self.assertEqual(result.objective, exp_obj)
            self.assertEqual(result.deactivation_count, exp_deact)
            self.assertEqual(result.allocation, exp_alloc)
            # Every feasible random plan must also pass replay verification.
            ver = verify_plan(result, traces)
            self.assertTrue(ver.ok, ver.failures)
            feasible_cases += 1
        # Make sure both branches really ran (deterministic seed).
        self.assertGreaterEqual(feasible_cases, 10)
        self.assertGreaterEqual(infeasible_cases, 5)


if __name__ == "__main__":
    unittest.main()
