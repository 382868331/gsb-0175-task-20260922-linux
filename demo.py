"""Deterministic offline demo for the wayplan library.

Runs real computations (no sleeps, no prerecorded output), all well under
the 90-second demo budget (measured < 1 second total):

  1. a normal 3-app planning run: per-way miss curves, the exact optimal
     allocation, and replay statistics on the isolated caches;
  2. a hand-built counterexample where marginal-gain greedy is provably
     worse than the exact planner;
  3. hard per-app miss caps + optional paid deactivation: the caps are
     enforced inside the exact DP (never repaired after the fact), the
     deactivated app is reported separately, and verify_plan re-confirms
     every hard constraint by actual replay;
  4. genuinely triggered rejections (NaN weight, infeasible budget, and an
     unachievable hard cap with its necessary-condition diagnostics).

Everything is computed from deterministic in-repo data (fixed seeds).
"""

import math
import random

from wayplan import (
    InfeasibleBudgetError,
    ValidationError,
    plan_allocation,
    replay_isolated,
    verify_plan,
)


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_curves(result):
    header = "app     weight | " + " ".join(
        f"{w:>5}w" for w in range(1, result.total_ways + 1)
    )
    print("misses per way count (curve):")
    print(header)
    for i, app_id in enumerate(result.app_ids):
        cells = " ".join(f"{m:>6}" for m in result.curves[i])
        print(f"{app_id:<8}{result.weights[i]:>6} | {cells}")


def print_replay(stats):
    print("replay on isolated caches:")
    print(f"{'app':<8}{'ways':>5}{'accesses':>9}{'hits':>6}{'misses':>7}{'hit_rate':>10}")
    for s in stats:
        print(f"{s.app_id:<8}{s.ways:>5}{s.accesses:>9}{s.hits:>6}"
              f"{s.misses:>7}{s.hit_rate:>10.4f}")


def greedy_marginal_allocation(curves, weights, total_ways, budget):
    """Marginal-gain greedy, implemented here only to be compared against.

    Starts from 1 way per app and repeatedly gives one extra way to the app
    with the largest weighted miss reduction (ties -> lowest app index).
    """
    n = len(curves)
    alloc = [1] * n
    for _ in range(budget - n):
        best_i, best_gain = None, -1
        for i in range(n):
            if alloc[i] >= total_ways:
                continue
            gain = weights[i] * (curves[i][alloc[i] - 1] - curves[i][alloc[i]])
            if gain > best_gain:
                best_i, best_gain = i, gain
        if best_i is None:
            break
        alloc[best_i] += 1
    return alloc


def demo_normal_planning():
    section("1. Normal planning run (3 apps, deterministic synthetic traces)")
    rng = random.Random(20260922)

    # alpha: tight 3-block loop in one set with occasional random noise.
    alpha = [(i % 3) if rng.random() < 0.9 else rng.randrange(16)
             for i in range(120)]
    # beta: skewed random over 12 blocks (hot few, cold many).
    beta = rng.choices(range(12),
                       weights=[40, 25, 15, 8, 4, 2, 1, 1, 1, 1, 1, 1],
                       k=120)
    # gamma: sequential scan of 24 blocks, repeated (streaming pattern).
    gamma = [i % 24 for i in range(120)]

    # Traces are deliberately passed NOT sorted by app id, and weights are
    # aligned with this input order (alpha=2, beta=3, gamma=1).
    traces = [("gamma", gamma), ("alpha", alpha), ("beta", beta)]
    weights = [1, 2, 3]

    result = plan_allocation(traces, weights, num_sets=4, total_ways=8, budget=5)
    print_curves(result)
    print()
    print(f"optimal allocation (by app id): {result.allocation_map()}")
    print(f"used ways: {result.used_ways} / budget {result.budget} "
          f"(cache has {result.total_ways} ways, {result.num_sets} sets)")
    print(f"minimised weighted miss sum: {result.weighted_misses}")
    print(f"per-app misses at allocation: "
          f"{dict(zip(result.app_ids, result.per_app_misses))}")
    print()
    print_replay(replay_isolated(traces, result.allocation_map(), num_sets=4))


def demo_greedy_counterexample():
    section("2. Greedy counterexample (non-convex miss curves)")
    # appA: 20 x (9 accesses to block 0, then block 1) in a single set.
    #   1 way -> 40 misses; 2 ways -> 2 misses; 3 ways -> 2 (no further gain).
    # appB: cyclic loop over 3 blocks in the same set.
    #   1 or 2 ways -> 200 misses (LRU thrashes); 3 ways -> 3 cold misses.
    blocks_a = ([0] * 9 + [1]) * 20
    blocks_b = ([10, 11, 12] * 67)[:200]
    traces = [("appA", blocks_a), ("appB", blocks_b)]
    weights = [1, 1]

    result = plan_allocation(traces, weights, num_sets=1, total_ways=4, budget=4)
    print_curves(result)
    greedy = greedy_marginal_allocation(result.curves, weights,
                                        total_ways=4, budget=4)
    greedy_cost = sum(
        weights[i] * result.curves[i][greedy[i] - 1] for i in range(2)
    )
    print()
    print(f"greedy marginal-gain allocation: {dict(zip(result.app_ids, greedy))}"
          f"  -> weighted misses {greedy_cost}")
    print(f"exact optimal allocation:        {result.allocation_map()}"
          f"  -> weighted misses {result.weighted_misses}")
    print(f"greedy is worse by {greedy_cost - result.weighted_misses} weighted "
          f"misses: appB's payoff only appears at 3 ways, which greedy "
          f"never reaches after spending ways on appA's early gain.")


def demo_caps_and_deactivation():
    section("3. Hard miss caps + optional paid deactivation (num_sets=1)")
    # critical: [0,1,2] cycled -> curve (24,24,3,3,3,3); hard cap 6 misses
    #   forces >= 3 ways.  weight 2.
    # batch:    [3,4,5,6] cycled -> curve (24,24,24,4,4,4); declared
    #   deactivatable at penalty 20.  weight 1.
    # stream:   single hot block -> curve (1,1,1,1,1,1).  weight 1.
    critical = [0, 1, 2] * 8
    batch = [3, 4, 5, 6] * 6
    stream = [7] * 30
    traces = [("critical", critical), ("batch", batch), ("stream", stream)]
    weights = [2, 1, 1]

    plain = plan_allocation(traces, weights, num_sets=1, total_ways=6, budget=6)
    print("unconstrained reference (no caps, no deactivation):")
    print(f"  allocation {plain.allocation_map()}  "
          f"weighted misses {plain.weighted_misses}")
    print()

    result = plan_allocation(
        traces, weights, num_sets=1, total_ways=6, budget=6,
        miss_caps=[6, None, None],            # critical may miss at most 6
        deactivation_penalties=[None, 20, None],  # only batch may be dropped
    )
    print_curves(result)
    print()
    print("with hard cap critical<=6 and batch deactivatable (penalty 20):")
    print(f"  allocation (0 = deactivated): {result.allocation_map()}")
    print(f"  deactivated apps (reported separately): "
          f"{result.deactivated_app_ids()}")
    print(f"  weighted misses {result.weighted_misses} + deactivation "
          f"penalties {result.deactivation_cost} = objective "
          f"{result.objective}")
    print(f"  used ways: {result.used_ways} / budget {result.budget}")
    print("  keeping batch would cost 24 misses at any way count it can "
          "get here;")
    print("  paying the penalty of 20 is strictly cheaper, so the exact "
          "planner deactivates it.")
    print()

    ver = verify_plan(result, traces)
    print(f"verify_plan replay check: ok={ver.ok} "
          f"(failures: {list(ver.failures)})")
    print(f"  deactivated as reported by replay: {ver.deactivated}")
    print_replay(ver.replay)
    for s in ver.replay:
        i = result.app_ids.index(s.app_id)
        cap = result.miss_caps[i]
        if cap is not None:
            print(f"  hard cap confirmed by replay: {s.app_id} misses "
                  f"{s.misses} <= cap {cap}")


def demo_rejections():
    section("4. Genuinely triggered rejections")
    try:
        plan_allocation([("a", [0, 1]), ("b", [2])], [1, math.nan],
                        num_sets=2, total_ways=4, budget=3)
    except ValidationError as e:
        print(f"NaN weight rejected   -> ValidationError: {e}")
    try:
        plan_allocation([("a", [0]), ("b", [1]), ("c", [2])], [1, 1, 1],
                        num_sets=2, total_ways=4, budget=2)
    except InfeasibleBudgetError as e:
        print(f"infeasible budget     -> InfeasibleBudgetError: {e}")
    try:
        # critical's curve bottoms out at 3 misses; cap 2 is unachievable
        # and the app may not be deactivated -> proved infeasible.
        plan_allocation([("critical", [0, 1, 2] * 8)], [1],
                        num_sets=1, total_ways=6, budget=6, miss_caps=[2])
    except InfeasibleBudgetError as e:
        print(f"unachievable hard cap -> InfeasibleBudgetError: {e}")
        print(f"  diagnostics attribute: {e.diagnostics}")


def main():
    demo_normal_planning()
    demo_greedy_counterexample()
    demo_caps_and_deactivation()
    demo_rejections()
    print()
    print("demo finished: all numbers above were computed in this run.")


if __name__ == "__main__":
    main()
