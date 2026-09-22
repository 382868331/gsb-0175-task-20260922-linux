"""Hand-verified LRU simulation tests.

Every expected number in this file is computed by hand from the LRU
policy, not by running the implementation under test.
"""

import unittest

from wayplan import (
    ValidationError,
    miss_curve,
    simulate_lru_misses,
)


class HandComputedLRU(unittest.TestCase):
    def test_hand_curve_single_set(self):
        # num_sets=1, trace [1,2,1,3,1,4], worked out by hand:
        # ways=1: every access misses (each new block evicts the old) -> 6
        # ways=2: 1M 2M 1H 3M(evict 2) 1H 4M(evict 3)              -> 4
        # ways=3: 1M 2M 1H 3M 1H 4M(evict 2, the LRU)              -> 4
        # ways=4: only the four cold misses                        -> 4
        blocks = [1, 2, 1, 3, 1, 4]
        self.assertEqual(miss_curve(blocks, num_sets=1, max_ways=4), (6, 4, 4, 4))
        self.assertEqual(simulate_lru_misses(blocks, 1, 2), 4)

    def test_cyclic_conflict_loop(self):
        # A loop over k+1 blocks in one set defeats LRU with <= k ways:
        # every access evicts the block that will be needed next.
        blocks = [0, 1, 2] * 10  # 30 accesses, one set
        self.assertEqual(simulate_lru_misses(blocks, 1, 1), 30)
        self.assertEqual(simulate_lru_misses(blocks, 1, 2), 30)
        # With 3 ways the whole loop fits: only 3 cold misses.
        self.assertEqual(simulate_lru_misses(blocks, 1, 3), 3)

    def test_set_mapping_isolates_blocks(self):
        # num_sets=2: block b -> set b % 2.
        # [0,2,0,2] both land in set 0 and thrash with 1 way -> 4 misses.
        self.assertEqual(simulate_lru_misses([0, 2, 0, 2], 2, 1), 4)
        # [0,1,0,1] land in different sets, no conflict -> 2 cold misses.
        self.assertEqual(simulate_lru_misses([0, 1, 0, 1], 2, 1), 2)

    def test_lru_order_not_fifo(self):
        # Distinguishes LRU from FIFO: [0,1,2,0,3] with 2 ways, one set.
        # LRU: 0M 1M 2M(evict 0) 0M(evict 1) 3M(evict 2) -> 5
        # (FIFO would give the same here, so use a second trace:)
        # [0,1,0,2,0] ways=2: LRU 0M 1M 0H 2M(evict 1) 0H -> 3,
        # FIFO would evict 0 at the 4th access and miss the 5th -> 4.
        self.assertEqual(simulate_lru_misses([0, 1, 2, 0, 3], 1, 2), 5)
        self.assertEqual(simulate_lru_misses([0, 1, 0, 2, 0], 1, 2), 3)


class SimulationInputValidation(unittest.TestCase):
    def test_bool_rejected_as_integer_parameter(self):
        with self.assertRaises(ValidationError):
            simulate_lru_misses([1, 2], True, 2)
        with self.assertRaises(ValidationError):
            simulate_lru_misses([1, 2], 1, False)
        with self.assertRaises(ValidationError):
            miss_curve([1, 2], 1, True)

    def test_out_of_range_geometry(self):
        for bad_sets in (0, 17, -1):
            with self.assertRaises(ValidationError):
                simulate_lru_misses([1], bad_sets, 1)
        for bad_ways in (0, 13, -2):
            with self.assertRaises(ValidationError):
                simulate_lru_misses([1], 1, bad_ways)

    def test_bad_blocks(self):
        with self.assertRaises(ValidationError):  # empty trace
            simulate_lru_misses([], 1, 1)
        with self.assertRaises(ValidationError):  # negative block
            simulate_lru_misses([1, -1], 1, 1)
        with self.assertRaises(ValidationError):  # bool block
            simulate_lru_misses([True], 1, 1)
        with self.assertRaises(ValidationError):  # non-iterable
            simulate_lru_misses(5, 1, 1)

    def test_error_message_names_parameter(self):
        try:
            simulate_lru_misses([1], 0, 1)
        except ValidationError as e:
            self.assertIn("num_sets", str(e))
        else:
            self.fail("expected ValidationError")


if __name__ == "__main__":
    unittest.main()
