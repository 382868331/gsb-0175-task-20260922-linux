"""Deterministic offline demo for the wayplan library.

Runs five real computations (no sleeps, no prerecorded output):

  1. a normal 3-app planning run: per-way miss curves, the exact optimal
     allocation, and replay statistics on the isolated caches;
  2. a hand-built counterexample where marginal-gain greedy is provably
     worse than the exact planner;
  3. hard per-app miss caps + a disableable app: the constrained exact
     plan (which no patch of the unconstrained optimum can reach) and an
     independent replay verifying every hard constraint;
  4. a zero-budget plan (every app disableable) and a genuinely triggered
     infeasibility with its necessary-condition diagnostics;
  5. two genuinely triggered rejections (NaN weight, invalid cap).

Everything is computed from deterministic in-repo data (fixed seeds).
Total runtime is a fraction of a second, far below the 90 s budget.
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


def demo_caps_and_disable():
    section("3. Hard miss caps + disableable app (exact constrained plan)")
    # num_sets=1, hand-known curves:
    #   crit: [0,1] alternating          -> (40, 2, 2, 2, 2)
    #   bulk: 3-block cycle, 200 accesses-> (200, 200, 3, 3, 3)
    #   mix:  25x(20,21) + 17x(20,21,22) -> (101, 51, 3, 3, 3)
    crit = [0, 1] * 20
    bulk = ([10, 11, 12] * 67)[:200]
    mix = ([20, 21] * 25) + ([20, 21, 22] * 17)
    traces = [("crit", crit), ("bulk", bulk), ("mix", mix)]
    weights = [1, 1, 1]

    unc = plan_allocation(traces, weights, num_sets=1, total_ways=5, budget=5)
    print(f"unconstrained optimum: {unc.allocation_map()} "
          f"-> objective {unc.objective}")

    # crit gets a hard miss cap of 2; bulk may be disabled for penalty 150.
    result = plan_allocation(
        traces, weights, num_sets=1, total_ways=5, budget=5,
        miss_caps=[2, None, None],
        disable_penalties=[None, 150, None],
    )
    print_curves(result)
    print()
    print(f"constrained plan: {result.allocation_map()}")
    print(f"  objective {result.objective} = weighted misses "
          f"{result.weighted_misses} + disable penalties "
          f"{result.disable_penalty_total}")
    print(f"  disabled apps (reported separately): "
          f"{result.disabled_app_ids or 'none'}")
    print(f"  per-app misses: "
          f"{dict(zip(result.app_ids, result.per_app_misses))} "
          f"(cap on crit: 2)")
    print("  note: patching the unconstrained (bulk=3,crit=1,mix=1) to respect\n"
          "        the cap can only reach (bulk=2,crit=2,mix=1) at cost 303 —\n"
          "        the exact constrained DP instead disables bulk outright\n"
          "        (penalty 150) and reinvests the ways, reaching 155.  This\n"
          "        is why no compute-then-patch step is used.")

    report = verify_plan(traces, result)
    print()
    print("independent replay verification of every hard constraint:")
    for c in report.checks:
        print(f"  [{'ok' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    print(f"  => plan verified: {report.ok}")


def demo_zero_budget_and_infeasible():
    section("4. Zero budget and proved-infeasible diagnostics")
    traces = [("a", [0, 1]), ("b", [2, 3])]
    r = plan_allocation(traces, [1, 1], num_sets=2, total_ways=4, budget=0,
                        disable_penalties=[3, 7])
    print(f"zero budget, all apps disableable: {r.allocation_map()}, "
          f"objective {r.objective} (penalties only)")

    # bulk's best achievable is 3 misses; a cap of 2 is unreachable and the
    # app may not be disabled -> proved infeasible, with diagnostics.
    try:
        plan_allocation([("crit", [0, 1] * 20),
                         ("bulk", ([10, 11, 12] * 67)[:200])],
                        [1, 1], num_sets=1, total_ways=4, budget=4,
                        miss_caps=[None, 2])
    except InfeasibleBudgetError as e:
        print()
        print("infeasible cap configuration -> InfeasibleBudgetError:")
        print(f"  {e}")
        print(f"  diagnostic attributes: app_ids={e.app_ids}, "
              f"min_ways={e.min_ways}, required_total={e.required_total}, "
              f"budget={e.budget}")


def demo_rejections():
    section("5. Genuinely triggered rejections")
    try:
        plan_allocation([("a", [0, 1]), ("b", [2])], [1, math.nan],
                        num_sets=2, total_ways=4, budget=3)
    except ValidationError as e:
        print(f"NaN weight rejected   -> ValidationError: {e}")
    try:
        plan_allocation([("a", [0, 1]), ("b", [2])], [1, 1],
                        num_sets=2, total_ways=4, budget=3,
                        miss_caps=[-1, None])
    except ValidationError as e:
        print(f"negative cap rejected -> ValidationError: {e}")


def main():
    demo_normal_planning()
    demo_greedy_counterexample()
    demo_caps_and_disable()
    demo_zero_budget_and_infeasible()
    demo_rejections()
    print()
    print("demo finished: all numbers above were computed in this run.")


if __name__ == "__main__":
    main()
