"""Deterministic offline demo for the wayplan library.

Runs three real computations (no sleeps, no prerecorded output):

  1. a normal 3-app planning run: per-way miss curves, the exact optimal
     allocation, and replay statistics on the isolated caches;
  2. a hand-built counterexample where marginal-gain greedy is provably
     worse than the exact planner;
  3. two genuinely triggered rejections (NaN weight, infeasible budget).

Everything is computed from deterministic in-repo data (fixed seeds).
"""

import math
import random

from wayplan import (
    InfeasibleBudgetError,
    InfeasibleStageError,
    Stage,
    Task,
    ValidationError,
    plan_allocation,
    plan_task_stages,
    replay_isolated,
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


def demo_rejections():
    section("3. Genuinely triggered rejections")
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


def print_task_plan(p):
    print(f"task {p.task_id!r}:")
    print(f"{'stage':>6}{'ways':>6}{'loss':>7}{'migration':>11}")
    for i, (w, lo, mi) in enumerate(p.stages()):
        print(f"{i:>6}{w:>6}{lo:>7}{mi:>11}")
    print(f"  stage loss {p.total_loss} + migration {p.total_migration_cost}"
          f" = total {p.total_cost}")


def demo_task_stages():
    section("4. Multi-stage planning (known stage sequences, 8 ways max)")
    # One service with a hand-built 4-stage sequence, num_ways=4.
    # Off-peak stages 0/3 reward many ways: curve (80, 40, 10, 5),
    #   performance floor loss <= 60  -> feasible ways {2, 3, 4}.
    # Peak stages 1/2 reward few ways: curve (90, 8, 30, 70),
    #   floor loss <= 30              -> feasible ways {2, 3}.
    stages = [
        Stage(1, 60, [80, 40, 10, 5]),
        Stage(1, 30, [90, 8, 30, 70]),
        Stage(1, 30, [90, 8, 30, 70]),
        Stage(1, 60, [80, 40, 10, 5]),
    ]
    cheap = plan_task_stages([Task("svc", stages)], num_ways=4,
                             migration_cost=6, max_change=1)
    print("migration_cost=6, max_change=1:")
    print_task_plan(cheap.plans[0])
    print("  (moving 3->2->2->3 pays the fee twice: 36 + 2*6 = 48)")
    dear = plan_task_stages([Task("svc", stages)], num_ways=4,
                            migration_cost=30, max_change=1)
    print("migration_cost=30, max_change=1:")
    print_task_plan(dear.plans[0])
    print("  (staying at 3 ways avoids all fees: 80 < 36 + 2*30 = 96)")

    # A second, independent task planned in the same call, with a per-task
    # migration override: 2 stages whose cheapest counts are far apart,
    # but max_change=1 forbids the direct jump in one go... here each
    # boundary must move at most 1 way.
    print()
    rigid = Task(
        "rigid",
        [Stage(4, 9, [9, 9, 9, 9]), Stage(1, 0, [0, 5, 5, 5])],
        max_change=1,
    )
    try:
        plan_task_stages([rigid], num_ways=4)
    except InfeasibleStageError as e:
        print(f"provably unreachable -> InfeasibleStageError: {e}")

    # Malformed input: loss curve shorter than num_ways.
    try:
        plan_task_stages([Task("bad", [Stage(1, 9, [9, 1])])], num_ways=4)
    except ValidationError as e:
        print(f"malformed curve      -> ValidationError: {e}")


def main():
    demo_normal_planning()
    demo_greedy_counterexample()
    demo_rejections()
    demo_task_stages()
    print()
    print("demo finished: all numbers above were computed in this run.")


if __name__ == "__main__":
    main()
