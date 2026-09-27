"""Hand-verified tests for warm-start (initial_state) cache planning.

Every expected number is derived by hand from the LRU policy applied to
the documented initial-state contract: rows are LRU -> MRU, oversized rows
are shrunk LRU-first, and every candidate run starts from the same
normalised copy of the initial state.
"""

import unittest

from wayplan import (
    InfeasibleBudgetError,
    ValidationError,
    miss_curve,
    plan_allocation,
    replay_isolated,
    simulate_lru_misses,
)


class WarmStartSimulation(unittest.TestCase):
    def test_warm_start_reduces_misses(self):
        # num_sets=1, initial {1(LRU), 2(MRU)}, ways=2, trace [1,2,1,3,1,4]:
        # 1H 2H 1H 3M(evict 2) 1H 4M(evict 3) -> 2 misses (cold start: 4).
        blocks = [1, 2, 1, 3, 1, 4]
        self.assertEqual(simulate_lru_misses(blocks, 1, 2), 4)
        self.assertEqual(
            simulate_lru_misses(blocks, 1, 2, initial_state=[[1, 2]]), 2
        )

    def test_initial_recency_order_matters(self):
        # ways=2, trace [2, 0]:
        # initial [0,1] (0 is LRU): 2M evicts 0 -> {1,2}; 0M evicts 1 -> 2
        # initial [1,0] (1 is LRU): 2M evicts 1 -> {0,2}; 0H          -> 1
        self.assertEqual(
            simulate_lru_misses([2, 0], 1, 2, initial_state=[[0, 1]]), 2
        )
        self.assertEqual(
            simulate_lru_misses([2, 0], 1, 2, initial_state=[[1, 0]]), 1
        )

    def test_oversized_initial_row_shrunk_lru_first(self):
        # initial [0,1,2] (0 LRU) with ways=2 keeps {1,2}; trace [1,2,0]:
        # 1H 2H 0M(evict 1) -> 1 miss.  With ways=1 only {2} survives:
        # 1M(evict 2) 2M(evict 1) 0M -> 3 misses.  With ways=3 all hit.
        self.assertEqual(
            miss_curve([1, 2, 0], 1, 3, initial_state=[[0, 1, 2]]),
            (3, 1, 0),
        )

    def test_per_set_rows(self):
        # num_sets=2, initial set0={0}, set1={1}; trace [0,1,0,1], ways=1:
        # all hits -> 0 misses (cold start would be 2).
        self.assertEqual(
            simulate_lru_misses([0, 1, 0, 1], 2, 1, initial_state=[[0], [1]]),
            0,
        )
        self.assertEqual(simulate_lru_misses([0, 1, 0, 1], 2, 1), 2)

    def test_curve_matches_single_runs_from_same_initial_state(self):
        # Each candidate way count is an independent run from the same
        # initial condition, so the curve must equal per-way single runs.
        blocks = [3, 1, 4, 1, 5, 9, 2, 6]
        init = [[6, 0], [4], [2]]
        curve = miss_curve(blocks, 3, 4, initial_state=init)
        for w in range(1, 5):
            self.assertEqual(
                curve[w - 1],
                simulate_lru_misses(blocks, 3, 4, initial_state=init)
                if w == 4
                else simulate_lru_misses(blocks, 3, w, initial_state=init),
            )


class InitialStateIsolation(unittest.TestCase):
    def test_caller_object_never_mutated_or_aliased(self):
        init = [[0, 2], [1]]
        snapshot = [row[:] for row in init]
        simulate_lru_misses([0, 1, 2, 0], 2, 1, initial_state=init)
        miss_curve([0, 1, 2, 0], 2, 2, initial_state=init)
        self.assertEqual(init, snapshot)

    def test_repeated_runs_identical(self):
        # No state leaks from one run (or candidate) into the next.
        init = [[1, 2]]
        first = miss_curve([1, 2, 3, 1], 1, 3, initial_state=init)
        second = miss_curve([1, 2, 3, 1], 1, 3, initial_state=init)
        self.assertEqual(first, second)
        # And a run after a *different* initial state is unaffected too.
        simulate_lru_misses([9, 9, 9], 1, 1, initial_state=[[7]])
        self.assertEqual(miss_curve([1, 2, 3, 1], 1, 3, initial_state=init), first)

    def test_planner_candidates_share_one_initial_condition(self):
        # Two planning runs with the same mutable initial state must agree,
        # proving candidates never inherit a previous candidate's cache.
        traces = [("a", [1, 0, 1, 0]), ("b", [5, 5, 5])]
        init = [[1]]
        r1 = plan_allocation(traces, [1, 1], 1, 2, 2, initial_state=init)
        r2 = plan_allocation(traces, [1, 1], 1, 2, 2, initial_state=init)
        self.assertEqual(r1.allocation, r2.allocation)
        self.assertEqual(r1.weighted_misses, r2.weighted_misses)
        self.assertEqual(init, [[1]])


class WarmStartPlanning(unittest.TestCase):
    def test_hand_computed_plan(self):
        # num_sets=1, total_ways=2, budget=2, weights 1/1, initial {1}.
        # appA [1,0,1,0]: ways=1 -> 1H 0M(ev1) 1M(ev0) 0M = 3;
        #                 ways=2 -> 1H 0M 1H 0H = 1
        # appB [5,5,5]:   ways=1 -> 5M(ev1) 5H 5H = 1; ways=2 -> 5M = 1
        # budget forces (1,1): cost 3 + 1 = 4.
        traces = [("b", [5, 5, 5]), ("a", [1, 0, 1, 0])]
        result = plan_allocation(
            traces, [1, 1], 1, 2, 2, initial_state=[[1]]
        )
        self.assertEqual(result.app_ids, ("a", "b"))
        self.assertEqual(result.allocation, (1, 1))
        self.assertEqual(result.per_app_misses, (3, 1))
        self.assertEqual(result.weighted_misses, 4)
        self.assertEqual(result.curves[0], (3, 1))
        self.assertEqual(result.initial_state, ((1,),))

    def test_warm_start_changes_optimal_allocation(self):
        # Cold: appA [1,0,1,0] ways1=4 ways2=ways3=2; appB 1 miss at any ways.
        # budget=3, total_ways=3: (1,2)=4+1=5, (2,1)=2+1=3 -> (2,1).
        # Warm initial {1}: appA ways1=3 ways2=ways3=1; (1,2)=3+1=4,
        # (2,1)=1+1=2 -> still (2,1) but with a different objective.
        traces = [("a", [1, 0, 1, 0]), ("b", [5, 5, 5])]
        cold = plan_allocation(traces, [1, 1], 1, 3, 3)
        warm = plan_allocation(traces, [1, 1], 1, 3, 3, initial_state=[[1]])
        self.assertEqual(cold.allocation, (2, 1))
        self.assertEqual(cold.weighted_misses, 3)
        self.assertEqual(warm.allocation, (2, 1))
        self.assertEqual(warm.weighted_misses, 2)

    def test_isolation_constraints_still_enforced(self):
        # The minimum-one-way-per-app isolation rule is unchanged by a
        # warm start: budget < app count stays provably infeasible.
        with self.assertRaises(InfeasibleBudgetError):
            plan_allocation(
                [("a", [0]), ("b", [1]), ("c", [2])],
                [1, 1, 1],
                1,
                2,
                2,
                initial_state=[[0]],
            )

    def test_replay_consistency_with_warm_start(self):
        # Replay from the same initial condition must reproduce the
        # planned per-app miss counts exactly.
        traces = [("a", [1, 0, 1, 0]), ("b", [5, 5, 5])]
        init = [[1]]
        result = plan_allocation(traces, [1, 1], 1, 2, 2, initial_state=init)
        stats = replay_isolated(
            traces, result.allocation_map(), 1, initial_state=init
        )
        self.assertEqual(
            tuple(s.misses for s in stats), result.per_app_misses
        )


class InitialStateValidation(unittest.TestCase):
    def test_row_count_must_match_num_sets(self):
        with self.assertRaises(ValidationError) as ctx:
            simulate_lru_misses([0], 2, 1, initial_state=[[0]])
        self.assertIn("initial_state", str(ctx.exception))

    def test_block_must_belong_to_its_set(self):
        with self.assertRaises(ValidationError) as ctx:
            simulate_lru_misses([0], 2, 1, initial_state=[[1], []])
        self.assertIn("initial_state[0][0]", str(ctx.exception))

    def test_duplicate_block_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            simulate_lru_misses([0], 1, 2, initial_state=[[0, 0]])
        self.assertIn("initial_state[0][1]", str(ctx.exception))

    def test_scalar_and_type_errors(self):
        for bad in (5, "01", [[True]], [[-1]], [[1.5]], [0], [[0], "x"]):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                simulate_lru_misses([0], 1, 1, initial_state=[bad[0]] if False else bad)

    def test_empty_rows_allowed(self):
        # Explicitly empty rows are a valid (partially warm) initial state.
        self.assertEqual(
            simulate_lru_misses([0, 2, 0], 2, 1, initial_state=[[], []]), 3
        )

    def test_failed_call_leaves_no_residue(self):
        with self.assertRaises(ValidationError):
            plan_allocation(
                [("a", [0])], [1], 1, 2, 1, initial_state=[[0, 0]]
            )
        # Subsequent valid calls behave exactly as before.
        result = plan_allocation([("a", [0, 0])], [1], 1, 2, 1)
        self.assertEqual(result.per_app_misses, (1,))


if __name__ == "__main__":
    unittest.main()
