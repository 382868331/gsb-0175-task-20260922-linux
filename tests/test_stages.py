"""Staged-planner tests: hand-checked cases, exhaustive cross-check, edges.

The brute-force reference in this file enumerates every feasible way
sequence independently; it never calls plan_stages to obtain expected
values.
"""

import itertools
import math
import random
import unittest

from wayplan import (
    MAX_DYN_WAYS,
    MAX_STAGES,
    MAX_TASKS,
    StagePlanResult,
    TaskSpec,
    ValidationError,
    WayPlanError,
    plan_stages,
)


# ---------------------------------------------------------------------------
# Independent reference implementation (NOT the code under test)
# ---------------------------------------------------------------------------

def ref_forward_dp(tasks, migration_cost, switch_cost, max_change):
    """Independent forward DP oracle for instances too large to enumerate.

    Written in a different style from the code under test (forward sweep
    keyed by prefix tuples, no backtracking).  Ties are resolved by plain
    tuple comparison of full prefixes, matching the lexicographic contract.
    """
    n = len(tasks)
    s_count = len(tasks[0]["losses"])
    w_count = len(tasks[0]["losses"][0])
    cap = MAX_DYN_WAYS if max_change is None else max_change

    def choices_at(s):
        return [
            list(range(tasks[i]["min_ways"][s], w_count + 1))
            for i in range(n)
        ]

    states = {}
    for combo in itertools.product(*choices_at(0)):
        loss = 0
        for i, w in enumerate(combo):
            loss += tasks[i]["losses"][0][w - 1]
        states[combo] = (loss, combo)

    for s in range(1, s_count):
        nxt = {}
        for prev, (pcost, ppath) in states.items():
            for combo in itertools.product(*choices_at(s)):
                move = 0
                ok = True
                for i, w in enumerate(combo):
                    d = abs(w - prev[i])
                    if d > cap:
                        ok = False
                        break
                    if d:
                        move += migration_cost * d + switch_cost
                if not ok:
                    continue
                loss = 0
                for i, w in enumerate(combo):
                    loss += tasks[i]["losses"][s][w - 1]
                cand = (pcost + move + loss, ppath + combo)
                old = nxt.get(combo)
                if old is None or cand < old:
                    nxt[combo] = cand
        states = nxt
    return min(states.values())


def ref_brute_force(tasks, migration_cost, switch_cost, max_change):
    """Enumerate every feasible (task, stage) way assignment (small sizes).

    Returns (best_cost, flat_vector) with the vector flattened as
    (stage0 task0.., stage1 task0.., ...); ties resolved by plain tuple
    comparison, i.e. lexicographically smallest.
    """
    n = len(tasks)
    s_count = len(tasks[0]["losses"])
    w_count = len(tasks[0]["losses"][0])
    cap = MAX_DYN_WAYS if max_change is None else max_change

    per_task_choices = []
    for t in tasks:
        per_task_choices.append([
            list(range(t["min_ways"][s], w_count + 1))
            for s in range(s_count)
        ])

    best = None
    for seqs in itertools.product(
        *(itertools.product(*c) for c in per_task_choices)
    ):
        feasible = True
        cost = 0
        flat = []
        for s in range(s_count):
            for i in range(n):
                w = seqs[i][s]
                cost += tasks[i]["losses"][s][w - 1]
                flat.append(w)
                if s > 0:
                    d = abs(w - seqs[i][s - 1])
                    if d > cap:
                        feasible = False
                    elif d:
                        cost += migration_cost * d + switch_cost
        if not feasible:
            continue
        cand = (cost, tuple(flat))
        if best is None or cand < best:
            best = cand
    assert best is not None, "valid instance must be feasible"
    return best


def random_tasks(rng, n=None, stages=None, ways=None):
    n = n or rng.randint(1, MAX_TASKS)
    stages = stages or rng.randint(1, MAX_STAGES)
    ways = ways or rng.randint(1, MAX_DYN_WAYS)
    specs, raw = [], []
    for i in range(n):
        losses = [
            [rng.randint(0, 30) for _ in range(ways)] for _ in range(stages)
        ]
        mins = [rng.randint(1, ways) for _ in range(stages)]
        specs.append(TaskSpec(f"t{i}", losses, min_ways=mins))
        raw.append({"losses": losses, "min_ways": mins})
    return specs, raw


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------

class HandComputedStagedPlanning(unittest.TestCase):
    def test_migration_cost_changes_the_optimum(self):
        # One task, two stages over 2 way counts:
        #   stage 1: 1 way -> 0 loss, 2 ways -> 100
        #   stage 2: 1 way -> 100,     2 ways -> 0
        # Plans: switch 1->2 costs migration * 1; stay-at-1 / stay-at-2
        # each pay 100 stage loss.
        t = TaskSpec("t", [(0, 100), (100, 0)])

        r6 = plan_stages([t], migration_cost=6)
        self.assertEqual(r6.ways, ((1, 2),))
        self.assertEqual(r6.stage_losses, (0, 0))
        self.assertEqual(r6.transition_costs, (0, 6))
        self.assertEqual(r6.total_loss, 0)
        self.assertEqual(r6.total_transition_cost, 6)
        self.assertEqual(r6.total_cost, 6)

        # At migration 101 staying put (100 loss) beats switching (101).
        # Both stay plans cost 100; lexicographic tie-break -> (1, 1).
        r101 = plan_stages([t], migration_cost=101)
        self.assertEqual(r101.ways, ((1, 1),))
        self.assertEqual(r101.stage_losses, (0, 100))
        self.assertEqual(r101.transition_costs, (0, 0))
        self.assertEqual(r101.total_cost, 100)

        # Exactly indifferent: switching and staying both cost 100,
        # tie resolved toward the smaller flattened vector.
        r100 = plan_stages([t], migration_cost=100)
        self.assertEqual(r100.ways, ((1, 1),))
        self.assertEqual(r100.total_cost, 100)

    def test_switch_cost_is_per_actual_change(self):
        # 3 stages; best per-stage ways are 1, 2, 1.  Every move is one
        # unit; with migration_cost=0 each actual change still pays
        # switch_cost once.
        t = TaskSpec("t", [(0, 50), (50, 0), (0, 50)])
        r = plan_stages([t], migration_cost=0, switch_cost=7)
        self.assertEqual(r.ways, ((1, 2, 1),))
        self.assertEqual(r.transition_costs, (0, 7, 7))
        self.assertEqual(r.total_cost, 14)
        # Zero transition costs reproduce the per-stage greedy choice,
        # which here is also globally optimal.
        r0 = plan_stages([t], migration_cost=0, switch_cost=0)
        self.assertEqual(r0.ways, ((1, 2, 1),))
        self.assertEqual(r0.total_cost, 0)

    def test_max_change_forces_detour(self):
        # Stage 1 forces 3 ways (min_ways=3, loss 0 there); stage 2 wants
        # 1 way (loss 0).  Direct 3->1 costs migration 2; with max_change=1
        # it is forbidden.
        t = TaskSpec("t", [(100, 100, 0), (0, 50, 100)], min_ways=[3, 1])
        # Only 2 stages: feasible detour is 3->2: loss 50 + migration 1.
        r = plan_stages([t], migration_cost=1, max_change=1)
        self.assertEqual(r.ways, ((3, 2),))
        self.assertEqual(r.total_cost, 51)
        # Without the cap the direct move costs 2 and loses nothing.
        u = plan_stages([t], migration_cost=1)
        self.assertEqual(u.ways, ((3, 1),))
        self.assertEqual(u.total_cost, 2)

    def test_max_change_across_three_stages(self):
        # 3 stages, minima force 3 at stage 1; losses want 1 at stages 2-3.
        # max_change=1 forces at least one intermediate way count.
        t = TaskSpec(
            "t",
            [(99, 99, 0), (0, 50, 99), (0, 50, 99)],
            min_ways=[3, 1, 1],
        )
        r = plan_stages([t], migration_cost=10, max_change=1)
        self.assertEqual(r.ways[0][0], 3)
        self.assertLessEqual(abs(r.ways[0][1] - 3), 1)
        self.assertLessEqual(abs(r.ways[0][2] - r.ways[0][1]), 1)
        # Paths 3,2,1 = loss 50 + mig 20 = 70 vs 3,2,2 = 100 + 10 ... 70 wins.
        self.assertEqual(r.ways, ((3, 2, 1),))
        self.assertEqual(r.total_cost, 70)

    def test_multi_task_costs_are_summed_per_task(self):
        # A wants to switch 1->2 (pays 6); B's flat curves keep it at 1.
        a = TaskSpec("A", [(0, 100), (100, 0)])
        b = TaskSpec("B", [(0, 100), (0, 100)])
        r = plan_stages([a, b], migration_cost=6)
        self.assertEqual(r.task_ids, ("A", "B"))
        self.assertEqual(r.ways, ((1, 2), (1, 1)))
        self.assertEqual(r.transition_costs, (0, 6))
        self.assertEqual(r.total_cost, 6)
        self.assertEqual(r.ways_map(), {"A": (1, 2), "B": (1, 1)})
        self.assertEqual(
            r.stage_ways_map(), [{"A": 1, "B": 1}, {"A": 2, "B": 1}]
        )

    def test_tie_break_lexicographic_sorted_by_task_id(self):
        # Flat zero-loss curves: every plan costs 0; the smallest flattened
        # vector puts every task at its minimum at every stage.
        a = TaskSpec("a", [(0, 0), (0, 0)])
        b = TaskSpec("b", [(0, 0), (0, 0)])
        r = plan_stages([b, a])  # input order reversed
        self.assertEqual(r.task_ids, ("a", "b"))
        self.assertEqual(r.ways, ((1, 1), (1, 1)))

    def test_zero_migration_means_independent_stage_choices(self):
        # With no transition terms, each stage is independently optimized;
        # verify against per-stage minima picked by hand.
        a = TaskSpec("a", [(5, 1, 9), (9, 8, 0), (2, 0, 3)])
        r = plan_stages([a], migration_cost=0)
        self.assertEqual(r.ways, ((2, 3, 2),))
        self.assertEqual(r.stage_losses, (1, 0, 0))
        self.assertEqual(r.total_cost, 1)

    def test_single_stage_has_no_transition_cost(self):
        t = TaskSpec("t", [(5, 0, 2)])
        r = plan_stages([t], migration_cost=99, switch_cost=99)
        self.assertEqual(r.num_stages, 1)
        self.assertEqual(r.ways, ((2,),))
        self.assertEqual(r.transition_costs, (0,))
        self.assertEqual(r.total_transition_cost, 0)
        self.assertEqual(r.total_cost, 0)

    def test_min_ways_scalar_and_per_stage(self):
        # Scalar min 2 forbids the otherwise-optimal 1-way choice.
        t_scalar = TaskSpec("t", [(0, 0, 0), (0, 0, 0)], min_ways=2)
        r = plan_stages([t_scalar])
        self.assertEqual(r.ways, ((2, 2),))
        # Per-stage minima: stage 1 must use 3, stage 2 may use 1.
        t_list = TaskSpec("t", [(9, 9, 0), (0, 9, 9)], min_ways=[3, 1])
        r2 = plan_stages([t_list], migration_cost=0)
        self.assertEqual(r2.ways, ((3, 1),))


# ---------------------------------------------------------------------------
# Exhaustive cross-check on deterministic random instances
# ---------------------------------------------------------------------------

class BruteForceCrossCheck(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration(self):
        # Full enumeration on small instances; an independently written
        # forward-DP oracle on the larger ones.
        rng = random.Random(20260924)
        brute_cases = dp_cases = 0
        for _ in range(120):
            n = rng.randint(1, MAX_TASKS)
            stages = rng.randint(1, MAX_STAGES)
            ways = rng.randint(1, MAX_DYN_WAYS)
            specs, raw = random_tasks(rng, n, stages, ways)
            migration_cost = rng.choice([0, 1, 2, 5, 20])
            switch_cost = rng.choice([0, 0, 3, 15])
            max_change = rng.choice([None, None, 1, 2])
            result = plan_stages(
                specs, migration_cost=migration_cost,
                switch_cost=switch_cost, max_change=max_change,
            )
            # Enumeration cost is ways^(stages*n) sequence combinations.
            if ways ** (stages * n) <= 4096:
                exp_cost, exp_flat = ref_brute_force(
                    raw, migration_cost, switch_cost, max_change
                )
                brute_cases += 1
            else:
                exp_cost, exp_flat = ref_forward_dp(
                    raw, migration_cost, switch_cost, max_change
                )
                dp_cases += 1
            self.assertEqual(result.total_cost, exp_cost)
            flat = tuple(
                result.ways[i][s]
                for s in range(stages) for i in range(n)
            )
            self.assertEqual(flat, exp_flat)
            # Component bookkeeping must add up exactly.
            self.assertEqual(
                result.total_cost,
                result.total_loss + result.total_transition_cost,
            )
            self.assertEqual(
                result.total_transition_cost, sum(result.transition_costs)
            )
            self.assertEqual(result.total_loss, sum(result.stage_losses))
        self.assertGreaterEqual(brute_cases, 20)
        self.assertGreaterEqual(dp_cases, 20)


# ---------------------------------------------------------------------------
# Result shape and boundary behavior
# ---------------------------------------------------------------------------

class ResultShapeAndBounds(unittest.TestCase):
    def test_result_is_immutable(self):
        r = plan_stages([TaskSpec("t", [(0, 1)])])
        self.assertIsInstance(r, StagePlanResult)
        with self.assertRaises(Exception):
            r.total_cost = 5
        self.assertEqual(len(r.stage_ways_map()), 1)

    def test_max_size_instance_runs(self):
        # 3 tasks x 4 stages x 8 ways: the largest supported instance.
        rng = random.Random(11)
        specs = [
            TaskSpec(
                f"t{i}",
                [[rng.randint(0, 5) for _ in range(8)] for _ in range(4)],
            )
            for i in range(3)
        ]
        r = plan_stages(specs, migration_cost=2, switch_cost=1, max_change=3)
        self.assertEqual(len(r.ways), 3)
        self.assertEqual(len(r.ways[0]), 4)
        for seq in r.ways:
            for s in range(1, 4):
                self.assertLessEqual(abs(seq[s] - seq[s - 1]), 3)

    def test_unlimited_max_change_encoded(self):
        r = plan_stages([TaskSpec("t", [(0, 1)])])
        self.assertEqual(r.max_change, MAX_DYN_WAYS)
        r2 = plan_stages([TaskSpec("t", [(0, 1)])], max_change=2)
        self.assertEqual(r2.max_change, 2)

    def test_non_monotone_curves_are_supported(self):
        # Loss goes down then up with more ways: no shape assumption.
        t = TaskSpec("t", [(10, 0, 10, 5), (0, 10, 0, 10)])
        r = plan_stages([t], migration_cost=0)
        self.assertEqual(r.ways[0][0], 2)   # unique minimum
        # Stage 2 ties at 1 and 3 ways; lexicographic tie-break picks 1.
        self.assertEqual(r.ways[0][1], 1)


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------

class StagedValidation(unittest.TestCase):
    def _expect(self, *args, **kwargs):
        with self.assertRaises(ValidationError) as ctx:
            plan_stages(*args, **kwargs)
        return str(ctx.exception)

    def test_bad_task_id(self):
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("", [(0, 1)])
        self.assertIn("task_id", str(ctx.exception))

    def test_tasks_must_be_taskspec(self):
        msg = self._expect([("t", [(0, 1)])])
        self.assertIn("tasks[0]", msg)

    def test_wrong_task_count(self):
        msg = self._expect([])
        self.assertIn("tasks", msg)
        specs = [TaskSpec(f"t{i}", [(0,)]) for i in range(MAX_TASKS + 1)]
        msg = self._expect(specs)
        self.assertIn("tasks", msg)

    def test_duplicate_task_ids(self):
        msg = self._expect([TaskSpec("x", [(0,)]), TaskSpec("x", [(0,)])])
        self.assertIn("duplicate", msg)

    def test_stage_count_mismatch(self):
        msg = self._expect(
            [TaskSpec("a", [(0,), (0,)]), TaskSpec("b", [(0,)])]
        )
        self.assertIn("stage", msg)

    def test_curve_length_mismatch(self):
        msg = self._expect(
            [TaskSpec("a", [(0, 0)]), TaskSpec("b", [(0,)])]
        )
        self.assertIn("way count", msg)

    def test_too_many_stages_or_ways(self):
        # Limits are enforced while constructing the spec.
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [(0,)] * (MAX_STAGES + 1))
        self.assertIn("stage", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [(0,) * (MAX_DYN_WAYS + 1)])
        self.assertIn("way count", str(ctx.exception))

    def test_empty_curve_and_empty_stage_sequence(self):
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [[]])
        self.assertIn("at least 1 way", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [])
        self.assertIn("at least one stage", str(ctx.exception))

    def test_bad_loss_values(self):
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [[-1]])
        self.assertIn("losses of task 't', stage 0", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [[True]])
        self.assertIn("bool", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [[1.5]])
        self.assertIn("float", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [[math.nan]])
        self.assertIn("non-finite", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [(7)])  # not iterable: a bare int
        self.assertIn("losses of task 't'", str(ctx.exception))

    def test_min_ways_validation(self):
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [(0, 0)], min_ways=0)
        self.assertIn("min_ways", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [(0, 0)], min_ways=[1, 2])  # wrong length
        self.assertIn("min_ways", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [(0, 0)], min_ways=3)  # exceeds curve length
        self.assertIn("exceeds", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec("t", [(0, 0)], min_ways=True)
        self.assertIn("bool", str(ctx.exception))

    def test_cost_and_max_change_validation(self):
        specs = [TaskSpec("t", [(0, 1)])]
        msg = self._expect(specs, migration_cost=-1)
        self.assertIn("migration_cost", msg)
        msg = self._expect(specs, switch_cost=-2)
        self.assertIn("switch_cost", msg)
        msg = self._expect(specs, migration_cost=1.0)
        self.assertIn("float", msg)
        msg = self._expect(specs, migration_cost=True)
        self.assertIn("bool", msg)
        msg = self._expect(specs, max_change=0)
        self.assertIn("max_change", msg)
        msg = self._expect(specs, max_change=9)
        self.assertIn("max_change", msg)

    def test_errors_share_base_class(self):
        with self.assertRaises(WayPlanError):
            plan_stages([])

    def test_failed_call_leaves_no_state(self):
        # The module is stateless: a rejected call must not affect a later
        # valid call, and repeat calls give identical results.
        with self.assertRaises(ValidationError):
            plan_stages([TaskSpec("t", [(0, 1)])], migration_cost=-1)
        r1 = plan_stages([TaskSpec("t", [(0, 100), (100, 0)])],
                         migration_cost=5)
        r2 = plan_stages([TaskSpec("t", [(0, 100), (100, 0)])],
                         migration_cost=5)
        self.assertEqual(r1, r2)
        self.assertEqual(r1.ways, ((1, 2),))


if __name__ == "__main__":
    unittest.main()
