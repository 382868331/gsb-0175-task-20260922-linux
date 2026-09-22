"""LRU 仿真测试：手工逐步例 + 循环冲突 + 多余路无收益 + 独立参考对照。"""

from __future__ import annotations

import unittest

from cacheisolation import AppTrace, simulate_lru
from cacheisolation.samples import cycling_trace

from reference import reference_lru_misses, reference_lru_trace


class HandLRUTests(unittest.TestCase):
    def test_empty_trace(self):
        r = simulate_lru([], 1, 1)
        self.assertEqual((r.accesses, r.misses, r.hits), (0, 0, 0))
        self.assertEqual(r.miss_rate, 0.0)

    def test_single_block_repeated(self):
        r = simulate_lru([0, 0, 0, 0], 1, 1)
        self.assertEqual((r.misses, r.hits), (1, 3))
        self.assertEqual(r.miss_rate, 0.25)  # 1/4 可精确表示

    def test_two_way_hand_walkthrough(self):
        # sets=1, ways=2，逐步手工核对 LRU。
        trace = [0, 1, 0, 2, 0]
        events, ref_misses, ref_hits = reference_lru_trace(trace, 1, 2)
        # M M H M H
        self.assertEqual(events, [True, True, False, True, False])
        r = simulate_lru(trace, 1, 2)
        self.assertEqual((r.misses, r.hits), (ref_misses, ref_hits))
        self.assertEqual((r.misses, r.hits), (3, 2))
        # 再访问 1：1 已在访问 2 时被淘汰 -> miss；随后 1 命中。
        r2 = simulate_lru(trace + [1, 1], 1, 2)
        self.assertEqual((r2.misses, r2.hits), (4, 3))

    def test_direct_mapped_thrashing_vs_separate_sets(self):
        # sets=2 ways=1：0 与 2 同组（%2==0），互相淘汰 -> 4 次全缺失；
        same_set = simulate_lru([0, 2, 0, 2], 2, 1)
        self.assertEqual(same_set.misses, 4)
        # 0 与 1 不同组 -> 冷缺失 2 次后全命中。
        diff_set = simulate_lru([0, 1, 0, 1], 2, 1)
        self.assertEqual((diff_set.misses, diff_set.hits), (2, 2))

    def test_sets_way_independence(self):
        # 16 组各自独立：不同组的块互不淘汰。
        trace = list(range(16)) * 3
        r = simulate_lru(trace, 16, 1)
        self.assertEqual(r.misses, 16)  # 首轮冷缺失，之后每组命中


class CyclingConflictTests(unittest.TestCase):
    def test_loop_conflict_and_redundant_ways(self):
        # footprint=3 循环：sets=1。
        trace = cycling_trace(0, footprint=3, rounds=4).blocks
        self.assertEqual(len(trace), 12)
        by_ways = {w: simulate_lru(trace, 1, w).misses for w in range(1, 6)}
        # 路数不足：循环冲突，每次访问都缺失。
        self.assertEqual(by_ways[1], 12)
        self.assertEqual(by_ways[2], 12)
        # 达到 footprint：仅首轮冷缺失。
        self.assertEqual(by_ways[3], 3)
        # 多余的路没有任何收益（曲线平台，且非凸：第2路0收益、第3路大收益）。
        self.assertEqual(by_ways[4], 3)
        self.assertEqual(by_ways[5], 3)

    def test_two_block_alternation(self):
        trace = (0, 1) * 6
        self.assertEqual(simulate_lru(trace, 1, 1).misses, 12)
        for w in (2, 3, 4):
            self.assertEqual(simulate_lru(trace, 1, w).misses, 2)

    def test_monotone_nonincrease(self):
        # 确定性合成轨迹：混合步长，逐路 miss 单调不增。
        trace = tuple((t * 7 + (t // 3) * 5) % 11 for t in range(60))
        prev = None
        for w in range(1, 13):
            m = simulate_lru(trace, 4, w).misses
            if prev is not None:
                self.assertLessEqual(m, prev)
            prev = m


class CrossReferenceTests(unittest.TestCase):
    """与独立参考（list 版 LRU）在多种形状下逐一对照。"""

    def test_against_reference_many_shapes(self):
        traces = [
            tuple((t * 3) % 7 for t in range(40)),
            tuple((t // 2) % 5 for t in range(30)),
            tuple(range(8)) * 5,
            (0,) * 10,
            tuple((t * t) % 6 for t in range(25)),
        ]
        for trace in traces:
            for sets in (1, 2, 3, 4, 8, 16):
                for ways in range(1, 13):
                    got = simulate_lru(trace, sets, ways)
                    exp = reference_lru_misses(trace, sets, ways)
                    self.assertEqual(
                        got.misses, exp,
                        msg=f"trace={trace[:8]}... sets={sets} ways={ways}",
                    )
                    self.assertEqual(got.misses + got.hits, len(trace))


if __name__ == "__main__":
    unittest.main()
