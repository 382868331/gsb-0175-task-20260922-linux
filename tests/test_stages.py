"""Multi-stage planner tests: hand-checked DP cases, brute-force cross-check.

The brute-force reference in this file is fully independent of the
wayplan.stages code: it enumerates every way-count path and scores them
itself.  It never calls the planner to obtain expected values.
"""

import itertools
import random
import unittest

from wayplan import (
    MAX_STAGES,
    MAX_TASKS,
    MAX_TASK_WAYS,
    InfeasibleStageError,
    Stage,
    Task,
    ValidationError,
    plan_task_stages,
)


# ---------------------------------------------------------------------------
# Independent reference implementation (NOT the code under test)
# ---------------------------------------------------------------------------

def ref_brute_force(raw_stages, num_ways, migration_cost, max_change):
    """Enumerate every way-count path; return (cost, path) or None.

    raw_stages: list of (min_ways, max_loss, curve) tuples.
    Feasibility, switching charge and the lexicographic tie-break are all
    re-implemented here from scratch.
    """
    t = len(raw_stages)
    best = None
    for path in itertools.product(range(1, num_ways + 1), repeat=t):
        ok = True
        loss_sum = 0
        mig_sum = 0
        for i, (min_ways, max_loss, curve) in enumerate(raw_stages):
            w = path[i]
            if w < min_ways or curve[w - 1] > max_loss:
                ok = False
                break
            loss_sum += curve[w - 1]
            if i > 0:
                if abs(w - path[i - 1]) > max_change:
                    ok = False
                    break
                if w != path[i - 1]:
                    mig_sum += migration_cost
        if not ok:
            continue
        cand = (loss_sum + mig_sum, path)
        if best is None or cand < best:
            best = cand
    return best  # None <=> provably infeasible


def ref_split_cost(raw_stages, path, migration_cost):
    """Independent (loss_sum, migration_sum) decomposition for one path."""
    loss_sum = sum(raw_stages[i][2][path[i] - 1] for i in range(len(path)))
    mig_sum = migration_cost * sum(
        1 for i in range(1, len(path)) if path[i] != path[i - 1]
    )
    return loss_sum, mig_sum


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------

class HandComputedStages(unittest.TestCase):
    def test_switching_cost_changes_the_optimum(self):
        # num_ways=3, two stages:
        #   s0 curve (100, 10, 1): w=3 cheap, w=1 expensive
        #   s1 curve (1, 100, 100): w=1 cheap, w=3 expensive
        # Candidate paths (cap 100 everywhere, max_change=3):
        #   (3,1): 1 + 1 + switch        = 2 + mig
        #   (2,1): 10 + 1 + switch       = 11 + mig
        #   (1,1): 100 + 1, no switch    = 101
        #   (3,2): 1 + 100 + switch      = 101 + mig
        stages = [Stage(1, 100, [100, 10, 1]), Stage(1, 100, [1, 100, 100])]
        free = plan_task_stages([Task("t", stages)], 3,
                                migration_cost=0, max_change=3)
        self.assertEqual(free.plans[0].ways, (3, 1))
        self.assertEqual(free.plans[0].total_cost, 2)
        # migration=10: switching still wins (2+10=12 < 101)
        cheap = plan_task_stages([Task("t", stages)], 3,
                                 migration_cost=10, max_change=3)
        self.assertEqual(cheap.plans[0].ways, (3, 1))
        self.assertEqual(cheap.plans[0].total_loss, 2)
        self.assertEqual(cheap.plans[0].total_migration_cost, 10)
        self.assertEqual(cheap.plans[0].migration_costs, (0, 10))
        # migration=100: never move; best stay path is (1,1) = 101
        dear = plan_task_stages([Task("t", stages)], 3,
                                migration_cost=100, max_change=3)
        self.assertEqual(dear.plans[0].ways, (1, 1))
        self.assertEqual(dear.plans[0].total_migration_cost, 0)
        self.assertEqual(dear.plans[0].total_cost, 101)

    def test_max_change_blocks_otherwise_optimal_path(self):
        # Same stages; migration=10.  With max_change=3 the optimum is
        # (3,1) cost 12, but |3-1|=2, so with max_change=1 it is banned.
        # Reachable best is (2,1): 10 + 1 + 10 = 21.
        stages = [Stage(1, 100, [100, 10, 1]), Stage(1, 100, [1, 100, 100])]
        r = plan_task_stages([Task("t", stages)], 3,
                             migration_cost=10, max_change=1)
        self.assertEqual(r.plans[0].ways, (2, 1))
        self.assertEqual(r.plans[0].total_cost, 21)

    def test_min_ways_and_loss_floor_are_both_enforced(self):
        # s0 curve (5, 4, 4), cap 4, min 3: only w=3 feasible.
        # s1 curve (3, 3, 5), cap 4, min 2: w=3 loses 5 > 4, so only w=2.
        # migration=2, max_change=1: (3,2) is the sole path, cost 4+3+2 = 9.
        stages = [Stage(3, 4, [5, 4, 4]), Stage(2, 4, [3, 3, 5])]
        r = plan_task_stages([Task("t", stages)], 3,
                             migration_cost=2, max_change=1)
        self.assertEqual(r.plans[0].ways, (3, 2))
        self.assertEqual(r.plans[0].stage_losses, (4, 3))
        self.assertEqual(r.plans[0].migration_costs, (0, 2))
        self.assertEqual(r.plans[0].total_cost, 9)

    def test_loss_equal_to_floor_is_accepted(self):
        # cap is a <= constraint: loss == max_loss must be feasible.
        stages = [Stage(1, 5, [5, 1]), Stage(1, 1, [5, 1])]
        r = plan_task_stages([Task("t", stages)], 2, migration_cost=0)
        # stage0 all feasible, best w=2 (loss1); stage1 only w=2 (loss1)
        self.assertEqual(r.plans[0].ways, (2, 2))
        self.assertEqual(r.plans[0].total_cost, 2)

    def test_single_stage_has_no_migration(self):
        r = plan_task_stages([Task("t", [Stage(1, 10, [5, 1])])], 2,
                             migration_cost=7)
        p = r.plans[0]
        self.assertEqual(p.ways, (2,))
        self.assertEqual(p.stage_losses, (1,))
        self.assertEqual(p.migration_costs, (0,))
        self.assertEqual((p.total_loss, p.total_migration_cost,
                          p.total_cost), (1, 0, 1))

    def test_tie_break_is_lexicographic_path(self):
        # Every configuration has loss 3, no migration cost: all feasible
        # paths tie.  The lexicographically smallest path is all minimum
        # ways: (1, 1, 1).
        stages = [Stage(1, 3, [3, 3, 3]) for _ in range(3)]
        r = plan_task_stages([Task("t", stages)], 3, migration_cost=0)
        self.assertEqual(r.plans[0].ways, (1, 1, 1))
        self.assertEqual(r.plans[0].total_cost, 9)

    def test_staying_put_avoids_the_fee_even_when_other_ways_tie_on_loss(self):
        # Both w=2 and w=3 cost 1; migration fee must prevent pointless
        # oscillation 2 -> 3 -> 2.
        stages = [Stage(1, 1, [5, 1, 1]), Stage(1, 1, [5, 1, 1]),
                  Stage(1, 1, [5, 1, 1])]
        r = plan_task_stages([Task("t", stages)], 3, migration_cost=4)
        self.assertEqual(r.plans[0].ways, (2, 2, 2))
        self.assertEqual(r.plans[0].total_cost, 3)
        self.assertEqual(r.plans[0].total_migration_cost, 0)

    def test_per_task_overrides_and_sorting(self):
        # Identical stage data for two tasks:
        # s0 (10, 0), s1 (0, 10).  With migration=1 the optimum is (2,1)
        # cost 1; with migration=100 moving is too expensive and the best
        # stay path is (1,1) cost 10 (lexicographic over (2,2)).
        stages = [Stage(1, 10, [10, 0]), Stage(1, 10, [0, 10])]
        tasks = [
            Task("z", stages, migration_cost=1),       # override
            Task("a", stages, migration_cost=100),     # override
        ]
        r = plan_task_stages(tasks, 2, migration_cost=999)  # default unused
        self.assertEqual(r.task_ids, ("a", "z"))
        a, z = r.plans
        self.assertEqual(a.ways, (1, 1))
        self.assertEqual(a.total_cost, 10)
        self.assertEqual(z.ways, (2, 1))
        self.assertEqual(z.total_cost, 1)
        self.assertEqual(r.total_cost, 11)
        self.assertEqual(r.plan_map()["z"].ways, (2, 1))

    def test_max_change_override_per_task(self):
        # s0 forces w=4 (min_ways=4, flat curve), s1 can use 1..4 with a
        # flat curve.  Global max_change=1: task a can move at most to 3;
        # task b overrides to 3 and can reach 1.
        s0 = Stage(4, 9, [9, 9, 9, 9])
        s1 = Stage(1, 9, [9, 9, 9, 9])
        tasks = [Task("a", [s0, s1], max_change=1),
                 Task("b", [s0, s1], max_change=3)]
        r = plan_task_stages(tasks, 4)
        self.assertEqual(r.plan_map()["a"].ways, (4, 3))
        self.assertEqual(r.plan_map()["b"].ways, (4, 1))

    def test_tuple_short_forms_are_accepted(self):
        tasks = [("t", [(1, 4, [5, 4, 4]), (1, 4, [3, 3, 4])])]
        r = plan_task_stages(tasks, 3, migration_cost=2, max_change=2)
        self.assertEqual(r.plans[0].ways, (2, 2))
        self.assertEqual(r.plans[0].total_cost, 7)

    def test_boundary_sizes_accepted(self):
        # 3 tasks x 4 stages x 8 ways, all at the interface ceiling.
        flat = Stage(1, 0, [0] * MAX_TASK_WAYS)
        tasks = [
            Task(f"t{i}", [flat] * MAX_STAGES) for i in range(MAX_TASKS)
        ]
        r = plan_task_stages(tasks, MAX_TASK_WAYS,
                             migration_cost=1, max_change=MAX_TASK_WAYS)
        self.assertEqual(len(r.plans), MAX_TASKS)
        for p in r.plans:
            self.assertEqual(p.ways, (1, 1, 1, 1))
            self.assertEqual(p.total_cost, 0)
        self.assertEqual(r.total_cost, 0)

    def test_num_ways_one(self):
        stages = [Stage(1, 5, [3]), Stage(1, 5, [4])]
        r = plan_task_stages([Task("t", stages)], 1,
                             migration_cost=9, max_change=1)
        self.assertEqual(r.plans[0].ways, (1, 1))
        self.assertEqual(r.plans[0].total_cost, 7)
        self.assertEqual(r.plans[0].total_migration_cost, 0)


# ---------------------------------------------------------------------------
# Infeasibility (valid input, provably no solution)
# ---------------------------------------------------------------------------

class InfeasibleStages(unittest.TestCase):
    def test_stage_without_acceptable_way_count(self):
        # cap 0 but every loss is positive -> no feasible way count
        with self.assertRaises(InfeasibleStageError) as ctx:
            plan_task_stages([Task("t", [Stage(1, 0, [5, 1])])], 2)
        self.assertIn("stage 0", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, ValidationError)

    def test_unreachable_across_boundary(self):
        # s0 forces w=4; s1 accepts only w=1 (loss 0 there, 5 elsewhere);
        # max_change=1 makes the gap of 3 uncrossable.
        s0 = Stage(4, 9, [9, 9, 9, 9])
        s1 = Stage(1, 0, [0, 5, 5, 5])
        with self.assertRaises(InfeasibleStageError) as ctx:
            plan_task_stages([Task("t", [s0, s1], max_change=1)], 4)
        msg = str(ctx.exception)
        self.assertIn("stage 1", msg)
        self.assertIn("max_change", msg)

    def test_infeasibility_is_per_task_and_does_not_mutate_input(self):
        good = Task("ok", [Stage(1, 9, [1, 9])])
        bad = Task("bad", [Stage(1, 0, [5, 5])])
        with self.assertRaises(InfeasibleStageError):
            plan_task_stages([good, bad], 2)
        # A later, fully valid call is unaffected (no residual state).
        r = plan_task_stages([good, Task("also", [Stage(1, 9, [9, 1])])], 2)
        self.assertEqual(r.plan_map()["ok"].ways, (1,))
        self.assertEqual(r.plan_map()["also"].ways, (2,))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def one_task_plan(**overrides):
    kwargs = dict(
        tasks=[Task("t", [Stage(1, 9, [9, 1, 1])])],
        num_ways=3, migration_cost=0, max_change=None,
    )
    kwargs.update(overrides)
    return plan_task_stages(**kwargs)


class StageValidation(unittest.TestCase):
    def test_count_bounds(self):
        flat = Stage(1, 9, [1])
        with self.assertRaises(ValidationError):  # no tasks
            plan_task_stages([], 1)
        with self.assertRaises(ValidationError):  # 4 tasks
            plan_task_stages([(f"t{i}", [flat]) for i in range(MAX_TASKS + 1)], 1)
        with self.assertRaises(ValidationError):  # no stages
            Task("t", [])
        with self.assertRaises(ValidationError):  # 5 stages
            Task("t", [Stage(1, 9, [1])] * (MAX_STAGES + 1))

    def test_duplicate_and_empty_task_ids(self):
        flat = Stage(1, 9, [1])
        with self.assertRaises(ValidationError):
            plan_task_stages([("a", [flat]), ("a", [flat])], 1)
        with self.assertRaises(ValidationError):
            plan_task_stages([("", [flat])], 1)
        with self.assertRaises(ValidationError):
            Task(7, [flat])

    def test_num_ways_bounds(self):
        with self.assertRaises(ValidationError):
            one_task_plan(num_ways=0)
        with self.assertRaises(ValidationError):
            one_task_plan(num_ways=MAX_TASK_WAYS + 1)
        with self.assertRaises(ValidationError):
            one_task_plan(num_ways=True)
        with self.assertRaises(ValidationError):
            one_task_plan(num_ways=2.0)

    def test_curve_length_must_match_num_ways(self):
        with self.assertRaises(ValidationError) as ctx:
            plan_task_stages([Task("t", [Stage(1, 9, [9, 1])])], 3)
        self.assertIn("loss_curve", str(ctx.exception))
        self.assertIn("stage 0", str(ctx.exception))

    def test_curve_values_non_negative_ints(self):
        with self.assertRaises(ValidationError):
            Stage(1, 9, [1, -1])
        with self.assertRaises(ValidationError):
            Stage(1, 9, [1, 2.0])
        with self.assertRaises(ValidationError):
            Stage(1, 9, [True])
        with self.assertRaises(ValidationError):
            Stage(1, 9, [])
        with self.assertRaises(ValidationError):
            Stage(1, 9, 5)

    def test_stage_fields(self):
        with self.assertRaises(ValidationError):  # min_ways zero
            Stage(0, 9, [1])
        with self.assertRaises(ValidationError):  # min_ways bool
            Stage(True, 9, [1])
        with self.assertRaises(ValidationError):  # min_ways > num_ways
            plan_task_stages([Task("t", [Stage(4, 9, [9, 9, 9])])], 3)
        with self.assertRaises(ValidationError):  # NaN floor
            Stage(1, float("nan"), [1])
        with self.assertRaises(ValidationError):  # malformed triple
            Task("t", [(1, 9)])

    def test_migration_and_max_change(self):
        with self.assertRaises(ValidationError):
            one_task_plan(migration_cost=-1)
        with self.assertRaises(ValidationError):
            one_task_plan(migration_cost=1.5)
        with self.assertRaises(ValidationError):
            one_task_plan(max_change=0)
        with self.assertRaises(ValidationError):
            one_task_plan(max_change=4)  # > num_ways
        with self.assertRaises(ValidationError):  # task-level override
            plan_task_stages(
                [Task("t", [Stage(1, 9, [1, 1])], max_change=3)], num_ways=2
            )
        with self.assertRaises(ValidationError):
            Task("t", [Stage(1, 9, [1, 1])], migration_cost=-1)

    def test_messages_are_locatable(self):
        cases = [
            (dict(num_ways=0), "num_ways"),
            (dict(migration_cost=-2), "migration_cost"),
            (dict(max_change=9), "max_change"),
        ]
        for kwargs, needle in cases:
            with self.assertRaises(ValidationError) as ctx:
                one_task_plan(**kwargs)
            self.assertIn(needle, str(ctx.exception))

    def test_bad_task_iterable(self):
        with self.assertRaises(ValidationError):
            plan_task_stages("ab", 2)
        with self.assertRaises(ValidationError):
            plan_task_stages([42], 2)

    def test_stage_objects_are_immutable(self):
        s = Stage(1, 9, [1, 2])
        with self.assertRaises(Exception):
            s.min_ways = 2
        t = Task("t", [s])
        with self.assertRaises(Exception):
            t.task_id = "x"


# ---------------------------------------------------------------------------
# Exhaustive cross-check on deterministic random instances
# ---------------------------------------------------------------------------

class BruteForceCrossCheck(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration(self):
        rng = random.Random(20260924)
        feasible_cases = 0
        infeasible_cases = 0
        for _ in range(80):
            num_ways = rng.randint(1, MAX_TASK_WAYS)
            t = rng.randint(1, MAX_STAGES)
            migration_cost = rng.randint(0, 8)
            max_change = rng.randint(1, num_ways)
            raw = []
            for _ in range(t):
                curve = tuple(rng.randint(0, 20) for _ in range(num_ways))
                min_ways = rng.randint(1, num_ways)
                # Pick the floor from an actual curve value at/after
                # min_ways so feasible instances are common; occasionally
                # force an impossibly tight floor to cover infeasibility.
                if rng.random() < 0.15:
                    max_loss = min(curve[min_ways - 1:]) - 1
                else:
                    max_loss = rng.choice(curve[min_ways - 1:])
                raw.append((min_ways, max_loss, curve))
            stages = [Stage(m, cap, c) for m, cap, c in raw]
            expected = ref_brute_force(
                raw, num_ways, migration_cost, max_change
            )
            if expected is None:
                infeasible_cases += 1
                with self.assertRaises(InfeasibleStageError):
                    plan_task_stages(
                        [Task("t", stages)], num_ways,
                        migration_cost=migration_cost, max_change=max_change,
                    )
                continue
            feasible_cases += 1
            r = plan_task_stages(
                [Task("t", stages)], num_ways,
                migration_cost=migration_cost, max_change=max_change,
            )
            p = r.plans[0]
            exp_cost, exp_path = expected
            self.assertEqual(p.ways, exp_path)
            self.assertEqual(p.total_cost, exp_cost)
            exp_loss, exp_mig = ref_split_cost(raw, exp_path, migration_cost)
            self.assertEqual(p.total_loss, exp_loss)
            self.assertEqual(p.total_migration_cost, exp_mig)
            self.assertEqual(p.total_cost, exp_loss + exp_mig)
            self.assertEqual(p.migration_costs[0], 0)
        # Both branches of the semantics must actually have been exercised.
        self.assertGreaterEqual(feasible_cases, 50)
        self.assertGreaterEqual(infeasible_cases, 3)


if __name__ == "__main__":
    unittest.main()
