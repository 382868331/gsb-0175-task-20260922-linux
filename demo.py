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
    TaskSpec,
    ValidationError,
    plan_allocation,
    plan_stages,
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


# ---------------------------------------------------------------------------
# Staged planning: known stage sequence + per-unit migration cost
# ---------------------------------------------------------------------------

def staged_demo_tasks():
    # Two tasks, four known stages, three discrete way counts per stage.
    # Loss curves are hand-set so the whole demo is checkable on paper:
    #   X's per-stage optimum walks 1 -> 2 -> 3 -> 1 ways (last move = 2);
    #   Y is happy at 1 way until stage 4, whose optimum is 2 ways.
    x = TaskSpec("X", [
        (0, 10, 20),   # stage 1 optimum: 1 way
        (20, 0, 10),   # stage 2 optimum: 2 ways
        (20, 10, 0),   # stage 3 optimum: 3 ways
        (0, 10, 20),   # stage 4 optimum: 1 way (3 -> 1 is a 2-unit move)
    ])
    y = TaskSpec("Y", [
        (0, 5, 10),
        (0, 5, 10),
        (0, 5, 10),
        (10, 0, 5),    # stage 4 optimum: 2 ways
    ])
    return [x, y]


def print_staged_plan(label, result):
    print(f"{label}:")
    for task_id, seq in result.ways_map().items():
        print(f"  task {task_id}: ways per stage = {seq}")
    print(f"  stage losses         : {result.stage_losses}")
    print(f"  transition into stage: {result.transition_costs}")
    print(f"  total = loss {result.total_loss} + migration "
          f"{result.total_transition_cost} = {result.total_cost}")


def demo_staged_planning():
    section("4. Staged planning over a known 4-stage sequence (2 tasks)")
    tasks = staged_demo_tasks()
    print("loss curves per stage (columns = way counts 1, 2, 3):")
    for t in tasks:
        for s, curve in enumerate(t.losses):
            print(f"  task {t.task_id} stage {s + 1}: {curve}")
    print()
    # Cheap re-routing: the planner follows every per-stage optimum.
    # X moves 1->2->3->1 (4 moved units), Y moves 1->2 (1 unit): total 5.
    print_staged_plan("migration_cost=1 (cheap to re-route)",
                      plan_stages(tasks, migration_cost=1))
    print()
    # Expensive re-routing: following X's per-stage optima (1,2,3,1) moves
    # 4 units at cost 6 = 24 just for X.  The exact DP skips the climb to
    # 3 ways at stage 3: X stays at 2 then drops to 1 — 2 moved units
    # (cost 12) against 10 extra stage-3 loss, total 22 for X instead of 24.
    print_staged_plan("migration_cost=6 (re-routing is expensive)",
                      plan_stages(tasks, migration_cost=6))


def demo_staged_boundary_and_errors():
    section("5. Staged planning: max_change boundary and real rejections")
    tasks = staged_demo_tasks()
    # Boundary: max_change=1 forbids X's 2-unit move into stage 4 (3->1).
    # The exact DP skips the climb to 3 at stage 3 and stays at 2, so every
    # move is at most one way: 10 extra loss, total 13 vs 5 with no cap.
    capped = plan_stages(tasks, migration_cost=1, max_change=1)
    print_staged_plan("max_change=1 (|delta ways| <= 1 per stage)", capped)
    for seq in capped.ways:
        for s in range(1, capped.num_stages):
            assert abs(seq[s] - seq[s - 1]) <= 1
    print("  (verified: no stage-to-stage move exceeds 1 way)")
    uncapped = plan_stages(tasks, migration_cost=1)
    print(f"  without the cap the optimum total is {uncapped.total_cost}; "
          f"the cap costs {capped.total_cost - uncapped.total_cost}")
    print()
    try:
        plan_stages(tasks, migration_cost=-1)
    except ValidationError as e:
        print(f"negative migration cost -> ValidationError: {e}")
    try:
        TaskSpec("Z", [(0,) * 9])  # 9 discrete way counts, limit is 8
    except ValidationError as e:
        print(f"9-way curve rejected     -> ValidationError: {e}")


def main():
    demo_normal_planning()
    demo_greedy_counterexample()
    demo_rejections()
    demo_staged_planning()
    demo_staged_boundary_and_errors()
    print()
    print("demo finished: all numbers above were computed in this run.")


if __name__ == "__main__":
    main()
