"""Deterministic demo of the fair-allocation entry point (plan_fair_allocation).

Computes everything live (no prerecorded output, no sleeps).  Runs in well
under a second; the task budget is 90 seconds.

Section 1: a case where the fair plan costs slightly more weighted misses
           but improves the worst-off application's relative loss.
Section 2: exact rational reporting of the max ratio (no float tolerance).
Section 3: an actually triggered edge case — provably infeasible budget.
"""

from fractions import Fraction

from wayplan import (
    InfeasibleBudgetError,
    plan_allocation,
    plan_fair_allocation,
    replay_isolated,
)


def section(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def show_result(label, res, references=None):
    print(f"[{label}] allocation (ways per app): {res.allocation_map()}")
    print(f"  per-app misses        : {dict(zip(res.app_ids, res.per_app_misses))}")
    print(f"  weighted miss sum     : {res.weighted_misses}")
    if references is not None:
        ratios = {
            app: Fraction(m, r)
            for app, m, r in zip(res.app_ids, res.per_app_misses, references)
        }
        worst = max(ratios.values())
        print(f"  relative losses       : "
              + ", ".join(f"{a}={r} (~{float(r):.4f})" for a, r in ratios.items()))
        print(f"  worst relative loss   : {worst}")


def main():
    section("1. Fair plan vs. weighted-sum plan: worst app improves")
    # Two apps, one cache set, 6-way cache, budget 5 ways.
    # App 'b' has weight 5, so the weighted-sum planner sacrifices app 'a'
    # (12 misses against a reference of 4 -> relative loss 3).  The fair
    # planner accepts 8 extra weighted misses to cut the worst relative
    # loss from 3 to 5/4.
    traces = [
        ("a", [3, 4, 0, 4, 3, 0, 5, 3, 0, 1, 3, 1]),
        ("b", [2, 4, 0, 1, 3, 3, 4, 1, 5, 1, 1, 3]),
    ]
    weights = [1, 5]
    references = [4, 10]
    kw = dict(num_sets=1, total_ways=6, budget=5)

    weighted = plan_allocation(traces, weights, **kw)
    show_result("weighted-sum optimum (old entry)", weighted, references)

    fair = plan_fair_allocation(traces, weights, references, **kw)
    show_result("fair optimum (new entry)", fair, references)
    print()
    print(f"  -> worst relative loss: 3  ->  {fair.max_ratio} "
          f"(attained by {fair.max_ratio_apps})")
    print(f"  -> weighted miss sum: {weighted.weighted_misses} -> "
          f"{fair.weighted_misses} (+{fair.weighted_misses - weighted.weighted_misses}, "
          f"the price of fairness)")

    section("2. Exact rational max ratio (no float tolerance)")
    # 5/6 and 6/7 differ by only 1/42; a float tolerance of 0.05 would
    # merge them into a "tie".  Exact Fraction comparison keeps them
    # distinct and picks the truly smaller one.
    traces2 = [
        ("a", [2, 0, 4, 2, 4, 4, 2, 2]),
        ("b", [2, 3, 3, 2, 0, 0, 0, 1]),
    ]
    res2 = plan_fair_allocation(
        traces2, [1, 3], [7, 6], num_sets=1, total_ways=4, budget=3,
    )
    print(f"  allocation: {res2.allocation_map()}")
    print(f"  max ratio   : {res2.max_ratio_num}/{res2.max_ratio_den} "
          f"(exact), attained by {res2.max_ratio_apps}")
    print(f"  rejected alternative (1, 2) has worst ratio 6/7; "
          f"5/6 < 6/7 by exactly {Fraction(6, 7) - Fraction(5, 6)}")
    stats = replay_isolated(traces2, res2.allocation_map(), num_sets=1)
    for s in stats:
        print(f"  replay {s.app_id}: ways={s.ways} misses={s.misses} "
              f"hit_rate={s.hit_rate:.4f}")

    section("3. Edge case: provably infeasible budget")
    # 2 applications, each needs >= 1 way, but budget is 1: proved
    # infeasible (not a search timeout — the DP is exact and complete).
    try:
        plan_fair_allocation(traces2, [1, 1], [7, 6],
                             num_sets=1, total_ways=4, budget=1)
    except InfeasibleBudgetError as exc:
        print(f"  InfeasibleBudgetError raised as specified: {exc}")

    print()
    print("Demo finished: all values computed live, no sleeps, no recordings.")


if __name__ == "__main__":
    main()

