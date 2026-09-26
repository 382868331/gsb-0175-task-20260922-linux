"""Deterministic offline demo of the fair-allocation iteration.

Computes everything live (no sleeps, no prerecorded output):

  1. fair vs cost-minimising planning on the same traces: the fair plan
     accepts a slightly higher weighted miss total to cut the worst
     app's relative loss from 1 to 1/2, reported as an exact fraction;
  2. a three-app fair run cross-checked against brute-force enumeration
     of every way allocation (all three objective levels compared);
  3. a genuinely triggered boundary: budget too small to give every app
     its mandatory single way -> InfeasibleBudgetError (proved
     infeasible, not "search exhausted").
"""

import itertools
from fractions import Fraction

from wayplan import (
    InfeasibleBudgetError,
    miss_curve,
    plan_allocation,
    plan_fair_allocation,
    replay_isolated,
)


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_fair(result):
    print(f"allocation (by app id): {result.allocation_map()}  "
          f"({result.used_ways}/{result.budget} ways used)")
    print(f"max misses/reference = {result.max_ratio_num}/{result.max_ratio_den}"
          f" (exact), attained by {result.max_ratio_apps}")
    print(f"weighted miss sum (secondary objective): {result.weighted_misses}")
    print("per-app misses:",
          dict(zip(result.app_ids, result.per_app_misses)))


def demo_fair_vs_cost_min():
    section("1. Fair plan vs cost-minimising plan (same traces, budget 4)")
    # appA: 1 way -> 40 misses, 2 ways -> 2 (huge early gain, weight 6).
    # appB: cyclic 3-block loop; <3 ways -> 200 misses, 3 ways -> 3.
    blocks_a = ([0] * 9 + [1]) * 20
    blocks_b = ([10, 11, 12] * 67)[:200]
    traces = [("appA", blocks_a), ("appB", blocks_b)]
    weights = [6, 1]
    references = [80, 200]
    kw = dict(num_sets=1, total_ways=4, budget=4)

    cost = plan_allocation(traces, weights, **kw)
    fair = plan_fair_allocation(traces, weights, references, **kw)

    print("cost-minimising entry (plan_allocation):")
    print(f"  allocation {cost.allocation_map()}, weighted misses "
          f"{cost.weighted_misses}")
    cost_worst = max(Fraction(m, r) for m, r in zip(cost.per_app_misses,
                                                    references))
    print(f"  worst relative loss: {cost_worst} "
          f"(appB keeps missing on every access)")
    print()
    print("fair entry (plan_fair_allocation):")
    print("  ", end="")
    show_fair(fair)
    print(f"  -> weighted misses {fair.weighted_misses} > "
          f"{cost.weighted_misses}: the fair plan deliberately pays "
          f"{fair.weighted_misses - cost.weighted_misses} extra weighted "
          f"misses to drop the worst ratio from {cost_worst} to "
          f"{fair.max_ratio}.")
    print()
    print("replay of the fair allocation on isolated caches:")
    for s in replay_isolated(traces, fair.allocation_map(), num_sets=1):
        print(f"  {s.app_id}: {s.ways} ways, {s.misses} misses, "
              f"hit_rate {s.hit_rate:.4f}")


def demo_brute_force_crosscheck():
    section("2. Three-app fair plan cross-checked by full enumeration")
    traces = [
        ("gamma", [i % 5 for i in range(60)]),
        ("alpha", ([0] * 5 + [1, 2]) * 8),
        ("beta", [7] * 40 + [8, 9] * 10),
    ]
    weights = [2, 3, 1]
    references = [60, 30, 45]
    num_sets, total_ways, budget = 2, 5, 5

    res = plan_fair_allocation(traces, weights, references,
                               num_sets=num_sets, total_ways=total_ways,
                               budget=budget)
    show_fair(res)

    # Independent check: enumerate every allocation, apply the three-level
    # objective directly, and confirm the planner picked the same vector.
    curves = [miss_curve(blocks, num_sets, total_ways)
              for _, blocks in traces]
    best = None
    n_feasible = 0
    for alloc in itertools.product(range(1, total_ways + 1), repeat=3):
        if sum(alloc) > budget:
            continue
        n_feasible += 1
        key = (
            max(Fraction(curves[i][alloc[i] - 1], references[i])
                for i in range(3)),
            sum(weights[i] * curves[i][alloc[i] - 1] for i in range(3)),
            alloc,
        )
        if best is None or key < best:
            best = key
    enum_ratio, enum_weighted, enum_alloc = best
    # Planner reports apps sorted by id: (alpha, beta, gamma).
    planner_alloc_by_input_order = tuple(
        res.allocation_map()[app] for app, _ in traces)
    assert planner_alloc_by_input_order == enum_alloc, \
        (planner_alloc_by_input_order, enum_alloc)
    assert res.max_ratio == enum_ratio
    assert res.weighted_misses == enum_weighted
    print(f"enumeration over all {n_feasible} feasible allocations agrees: "
          f"{enum_alloc}, max ratio {enum_ratio}, weighted sum {enum_weighted}")


def demo_boundary():
    section("3. Boundary: budget below one way per app")
    traces = [("a", [0, 1]), ("b", [2]), ("c", [3])]
    try:
        plan_fair_allocation(traces, [1, 1, 1], [2, 1, 1],
                             num_sets=2, total_ways=4, budget=2)
    except InfeasibleBudgetError as e:
        print(f"InfeasibleBudgetError: {e}")
        print("3 apps each need >= 1 way but budget is 2: proved infeasible, "
              "so no optimal fair allocation exists to report.")


def main():
    demo_fair_vs_cost_min()
    demo_brute_force_crosscheck()
    demo_boundary()
    print()
    print("demo_iteration finished: all numbers computed in this run.")


if __name__ == "__main__":
    main()

