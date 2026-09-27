"""Warm-cache tests: explicit initial contents, order, and candidate isolation."""

import itertools
import random
import unittest

from wayplan import (
    ValidationError,
    plan_allocation,
    replay_isolated,
    simulate_lru_misses,
)


def ref_lru_with_initial(blocks, num_sets, num_ways, initial=None):
    """Independent list LRU.  Initial groups are ordered LRU -> MRU."""
    sets = []
    for set_index in range(num_sets):
        group = list(initial[set_index]) if initial is not None else []
        sets.append(group[max(0, len(group) - num_ways):])

    misses = 0
    for block in blocks:
        group = sets[block % num_sets]
        if block in group:
            group.remove(block)
            group.append(block)
        else:
            misses += 1
            group.append(block)
            if len(group) > num_ways:
                group.pop(0)
    return misses


class WarmLRUHandChecks(unittest.TestCase):
    def test_initial_order_distinguishes_lru_from_mru(self):
        # One set, two ways, warm entries ordered LRU -> MRU: (0, 1).
        # Accessing 1 is a hit and makes order (0, 1); new block 2 evicts 0.
        self.assertEqual(
            simulate_lru_misses([1, 2], 1, 2, [(0, 1)]), 1
        )
        # Reversing the initial order makes 0 MRU; block 2 evicts 1 instead.
        # Subsequent accesses to 0 and 1 expose the different state.
        self.assertEqual(
            simulate_lru_misses([2, 0, 1], 1, 2, [(1, 0)]), 2
        )

    def test_shrinking_evicts_lru_entries(self):
        # Initial group (0, 1, 2) is LRU -> MRU.  One way retains only 2,
        # so accesses to 0 and 1 miss, then 2 is gone and misses as well.
        self.assertEqual(
            simulate_lru_misses([2, 0, 1, 2], 1, 1, [(0, 1, 2)]), 3
        )
        # Two ways retain (1, 2); 0 was the LRU evicted by shrinkage.
        self.assertEqual(
            simulate_lru_misses([1, 2, 0, 2], 1, 2, [(0, 1, 2)]), 1
        )

    def test_set_group_contents_are_validated(self):
        with self.assertRaises(ValidationError) as ctx:
            simulate_lru_misses([0], 2, 2, [(1,), ()])
        self.assertIn("maps to set 1", str(ctx.exception))
        with self.assertRaises(ValidationError):
            simulate_lru_misses([0], 1, 2, [(0, 0)])
        with self.assertRaises(ValidationError):
            simulate_lru_misses([0], 2, 2, [(), (), ()])


class WarmPlanning(unittest.TestCase):
    def test_planner_uses_same_warm_state_for_every_candidate(self):
        traces = [
            ("a", [3]),
            ("b", [0, 1]),
        ]
        initial = {
            # In one set: a with two ways has 3 as MRU and avoids its access;
            # b with two ways has both later blocks present.
            "a": [(0, 3)],
            "b": [(0, 1)],
        }
        result = plan_allocation(
            traces, [1, 1], num_sets=1, total_ways=3, budget=3,
            initial_caches=initial,
        )
        # Warm curves:
        # a: 1 way keeps 3 -> (0, 0, 0)
        # b: 1 way keeps 1; 0 misses and evicts it, then 1 misses -> 2.
        self.assertEqual(result.curves, ((0, 0, 0), (2, 0, 0)))
        self.assertEqual(result.allocation, (1, 2))
        self.assertEqual(result.weighted_misses, 0)
        self.assertEqual(
            result.initial_caches_map(),
            {"a": ((0, 3),), "b": ((0, 1),)},
        )

    def test_each_candidate_gets_independent_fresh_copy(self):
        # Every way count replays from the same (0, 1, 2) LRU -> MRU state.
        # Reusing state left by the 1-way run would change the 2-way result.
        blocks = [1, 0, 2]
        initial = [(0, 1, 2)]
        result = plan_allocation(
            [("app", blocks)], [1], num_sets=1, total_ways=3, budget=3,
            initial_caches={"app": initial},
        )
        self.assertEqual(result.curves[0], (3, 2, 0))
        stats = replay_isolated(
            [("app", blocks)], {"app": 3}, num_sets=1,
            initial_caches={"app": initial},
        )
        self.assertEqual(stats[0].misses, 0)

    def test_brute_force_matches_warm_planning(self):
        rng = random.Random(92701)
        for _ in range(30):
            num_sets = rng.randint(1, 4)
            n = rng.randint(1, 3)
            total_ways = rng.randint(n, 6)
            budget = rng.randint(n, total_ways)
            traces = []
            blocks_by_app = []
            initial = {}
            for app_index in range(n):
                app_id = f"app{app_index}"
                blocks = [
                    set_index + num_sets * rng.randrange(4)
                    for set_index in (rng.randrange(num_sets) for _ in range(24))
                ]
                groups = []
                for set_index in range(num_sets):
                    choices = [
                        block for block in range(set_index, num_sets * 4, num_sets)
                    ]
                    rng.shuffle(choices)
                    groups.append(tuple(choices[:rng.randrange(total_ways + 1)]))
                traces.append((app_id, blocks))
                blocks_by_app.append(blocks)
                initial[app_id] = groups

            weights = [rng.randint(1, 4) for _ in range(n)]
            result = plan_allocation(
                traces, weights, num_sets, total_ways, budget,
                initial_caches=initial,
            )

            expected = None
            for allocation in itertools.product(
                range(1, total_ways + 1), repeat=n
            ):
                if sum(allocation) > budget:
                    continue
                cost = 0
                for i, ways in enumerate(allocation):
                    cost += weights[i] * ref_lru_with_initial(
                        blocks_by_app[i], num_sets, ways,
                        initial[f"app{i}"],
                    )
                candidate = (cost, allocation)
                if expected is None or candidate < expected:
                    expected = candidate
            self.assertEqual((result.weighted_misses, result.allocation), expected)

    def test_initial_cache_inputs_are_not_mutated(self):
        traces = [("a", [0, 2])]
        initial = {"a": [(0, 1)]}
        snapshot = {"a": [(0, 1)]}
        plan_allocation(
            traces, [1], 1, 2, 2, initial_caches=initial
        )
        replay_isolated(traces, {"a": 1}, 1, initial_caches=initial)
        self.assertEqual(initial, snapshot)

    def test_planner_rejects_unknown_and_non_mapping_initial_caches(self):
        traces = [("a", [0])]
        with self.assertRaises(ValidationError) as ctx:
            plan_allocation(
                traces, [1], 1, 1, 1,
                initial_caches={"a": [()], "b": [()]},
            )
        self.assertIn("unknown app_id", str(ctx.exception))
        with self.assertRaises(ValidationError):
            plan_allocation(traces, [1], 1, 1, 1, initial_caches=[[()]])


if __name__ == "__main__":
    unittest.main()
