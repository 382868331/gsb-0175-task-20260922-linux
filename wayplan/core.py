"""Cache way-partition planning with performance isolation.

Pipeline (all exact, no convexity assumption, no greedy heuristic):

1. ``simulate_lru_misses`` / ``miss_curve`` — for each application, simulate
   a set-associative LRU cache independently at every way count
   1..total_ways and record the miss count (the "per-way curve").
2. ``plan_allocation`` — exact dynamic program over (app, remaining budget)
   choosing an integer way count >= 1 per application, total <= budget,
   minimising the weighted miss sum with positive integer weights.
   Ties are broken by the lexicographically smallest allocation vector
   with applications ordered by app id.
3. ``replay_isolated`` — re-run each application on its own isolated cache
   sized to its allocated ways and report hit/miss statistics.

Address model: the input is a sequence of *block addresses* (non-negative
ints, block granularity — the block offset is already removed).  A block
``b`` maps to set ``b % num_sets`` and uses the full block id as its tag.
Each simulated cache is cold (empty) at the start of a run.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

from .errors import InfeasibleBudgetError, ValidationError

# Interface bounds fixed by the task contract.
MAX_APPS = 6
MAX_SETS = 16
MAX_WAYS = 12


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_int(value, name, lo=None, hi=None):
    """Return ``value`` if it is a genuine int in [lo, hi], else raise.

    bool is rejected even though it is an int subclass.  Floats are
    rejected; NaN and infinities get an explicit non-finite message.
    """
    if isinstance(value, bool):
        raise ValidationError(
            f"{name}: bool is not accepted as an integer parameter "
            f"(got {value!r})"
        )
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValidationError(
                f"{name}: non-finite value {value!r} is not allowed"
            )
        raise ValidationError(
            f"{name}: expected int, got float {value!r}; "
            f"non-integral numeric input is not silently coerced"
        )
    if not isinstance(value, int):
        raise ValidationError(
            f"{name}: expected int, got {type(value).__name__} ({value!r})"
        )
    if lo is not None and value < lo:
        raise ValidationError(f"{name}: must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise ValidationError(f"{name}: must be <= {hi}, got {value}")
    return value


def _require_num_sets(num_sets):
    return _require_int(num_sets, "num_sets", 1, MAX_SETS)


def _require_blocks(blocks, name="blocks"):
    if isinstance(blocks, (str, bytes)) or not hasattr(blocks, "__iter__"):
        raise ValidationError(
            f"{name}: expected an iterable of block addresses, "
            f"got {type(blocks).__name__}"
        )
    out = []
    for i, b in enumerate(blocks):
        out.append(_require_int(b, f"{name}[{i}]", 0))
    if not out:
        raise ValidationError(f"{name}: trace must contain at least one access")
    return tuple(out)


@dataclass(frozen=True)
class AppTrace:
    """One application's memory-access trace (validated, immutable)."""

    app_id: str
    blocks: tuple[int, ...]

    def __init__(self, app_id, blocks):
        if not isinstance(app_id, str) or not app_id:
            raise ValidationError(
                f"app_id: expected a non-empty str, got {app_id!r}"
            )
        object.__setattr__(self, "app_id", app_id)
        object.__setattr__(
            self, "blocks", _require_blocks(blocks, f"blocks of app {app_id!r}")
        )


def _normalize_traces(traces):
    """Validate the trace collection; return apps sorted by app_id."""
    if isinstance(traces, (str, bytes)) or not hasattr(traces, "__iter__"):
        raise ValidationError(
            f"traces: expected an iterable of AppTrace, "
            f"got {type(traces).__name__}"
        )
    apps = []
    for i, t in enumerate(traces):
        if not isinstance(t, AppTrace):
            # Convenience: accept (app_id, blocks) pairs, validated the same way.
            try:
                app_id, blocks = t
            except (TypeError, ValueError):
                raise ValidationError(
                    f"traces[{i}]: expected AppTrace or (app_id, blocks) pair, "
                    f"got {t!r}"
                ) from None
            t = AppTrace(app_id, blocks)
        apps.append(t)
    if not 1 <= len(apps) <= MAX_APPS:
        raise ValidationError(
            f"traces: expected 1..{MAX_APPS} applications, got {len(apps)}"
        )
    ids = [t.app_id for t in apps]
    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise ValidationError(f"traces: duplicate app_id(s): {dupes}")
    return tuple(apps)  # input order preserved; callers sort when needed


def _normalize_weights(weights, app_ids):
    if isinstance(weights, (str, bytes)) or not hasattr(weights, "__iter__"):
        raise ValidationError(
            f"weights: expected an iterable of positive ints, "
            f"got {type(weights).__name__}"
        )
    ws = list(weights)
    if len(ws) != len(app_ids):
        raise ValidationError(
            f"weights: expected {len(app_ids)} weights (one per application, "
            f"aligned with traces order), got {len(ws)}"
        )
    out = []
    for i, w in enumerate(ws):
        out.append(_require_int(w, f"weights[{i}]", 1))
    return tuple(out)


def _normalize_references(references, app_ids):
    if isinstance(references, (str, bytes)) or not hasattr(references, "__iter__"):
        raise ValidationError(
            f"references: expected an iterable of positive ints, "
            f"got {type(references).__name__}"
        )
    refs = list(references)
    if len(refs) != len(app_ids):
        raise ValidationError(
            f"references: expected {len(app_ids)} reference miss counts "
            f"(one per application, aligned with traces order), got {len(refs)}"
        )
    out = []
    for i, r in enumerate(refs):
        out.append(_require_int(r, f"references[{i}]", 1))
    return tuple(out)


# ---------------------------------------------------------------------------
# LRU simulation
# ---------------------------------------------------------------------------

def simulate_lru_misses(blocks, num_sets, num_ways):
    """Exact miss count of a cold set-associative LRU cache.

    ``blocks``  — iterable of non-negative int block addresses.
    ``num_sets``— 1..MAX_SETS; block b maps to set ``b % num_sets``.
    ``num_ways``— 1..MAX_WAYS; associativity of every set.
    """
    num_sets = _require_num_sets(num_sets)
    num_ways = _require_int(num_ways, "num_ways", 1, MAX_WAYS)
    blocks = _require_blocks(blocks)

    sets = [OrderedDict() for _ in range(num_sets)]  # key=block, LRU->MRU order
    misses = 0
    for b in blocks:
        cache = sets[b % num_sets]
        if b in cache:
            cache.move_to_end(b)  # hit: become most recently used
        else:
            misses += 1
            cache[b] = None
            if len(cache) > num_ways:
                cache.popitem(last=False)  # evict LRU
    return misses


def miss_curve(blocks, num_sets, max_ways):
    """Miss counts for every way count 1..max_ways.

    Returns a tuple of length ``max_ways``; element ``[w - 1]`` is the miss
    count with ``w`` ways.  No monotonicity or convexity is assumed.
    """
    max_ways = _require_int(max_ways, "max_ways", 1, MAX_WAYS)
    blocks = _require_blocks(blocks)
    num_sets = _require_num_sets(num_sets)
    return tuple(
        simulate_lru_misses(blocks, num_sets, w) for w in range(1, max_ways + 1)
    )


# ---------------------------------------------------------------------------
# Exact allocation planner (dynamic programming)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanResult:
    """Result of :func:`plan_allocation`.  All fields are immutable."""

    num_sets: int
    total_ways: int
    budget: int
    app_ids: tuple[str, ...]              # sorted by app id
    weights: tuple[int, ...]              # aligned with app_ids
    curves: tuple[tuple[int, ...], ...]   # curves[i][w-1] = misses of app i with w ways
    allocation: tuple[int, ...]           # ways per app, aligned with app_ids
    per_app_misses: tuple[int, ...]       # misses at the allocated way count
    weighted_misses: int                  # minimised objective value
    used_ways: int                        # sum(allocation), always <= budget

    def allocation_map(self):
        """Allocation as an insertion-ordered dict keyed by app id."""
        return dict(zip(self.app_ids, self.allocation))


def plan_allocation(traces, weights, num_sets, total_ways, budget):
    """Choose integer way counts minimising the weighted miss sum.

    Parameters
    ----------
    traces     : iterable of AppTrace (or (app_id, blocks) pairs), 1..6 apps,
                 unique app ids.  Apps are processed sorted by app_id, which
                 defines the allocation-vector order used for tie-breaking.
    weights    : iterable of positive ints aligned with ``traces`` order.
    num_sets   : 1..16, sets of the (shared) cache being partitioned.
    total_ways : 1..12, associativity of the cache; miss curves are
                 computed for 1..total_ways ways per app.
    budget     : 1..total_ways, maximum total ways handed out.

    Returns a :class:`PlanResult`.  Among all allocations with the minimal
    weighted miss sum, the lexicographically smallest allocation vector
    (apps ordered by app id) is returned.

    Raises
    ------
    ValidationError        — any contract violation in the inputs.
    InfeasibleBudgetError  — budget < number of applications (proved
                             infeasible; the DP itself is exact and complete,
                             so no "search exhausted" outcome exists).
    """
    apps_in = _normalize_traces(traces)
    ws_in = _normalize_weights(weights, [t.app_id for t in apps_in])
    num_sets = _require_num_sets(num_sets)
    total_ways = _require_int(total_ways, "total_ways", 1, MAX_WAYS)
    budget = _require_int(budget, "budget", 1, total_ways)

    # Pair each trace with its weight *before* sorting, so weights stay
    # aligned with the application they were supplied for.
    paired = sorted(zip(apps_in, ws_in), key=lambda p: p[0].app_id)
    apps = tuple(p[0] for p in paired)
    ws = tuple(p[1] for p in paired)
    n = len(apps)
    if budget < n:
        raise InfeasibleBudgetError(
            f"budget: {budget} way(s) cannot give each of the {n} "
            f"application(s) its mandatory minimum of 1 way"
        )

    curves = tuple(miss_curve(t.blocks, num_sets, total_ways) for t in apps)
    # cost[i][a] = weighted misses of app i when given a ways (a in 1..total_ways)
    cost = [
        [ws[i] * curves[i][a - 1] for a in range(total_ways + 1)]
        for i in range(n)
    ]

    # best[i][b] = (min_cost, allocation_tuple) for apps i..n-1 with a total
    # of at most b ways, or None when infeasible.  Processed back to front;
    # tuple comparison gives exact cost ordering with lexicographic
    # tie-breaking on the allocation vector in one pass.
    best = [[None] * (budget + 1) for _ in range(n + 1)]
    for b in range(budget + 1):
        best[n][b] = (0, ())  # no apps left: leftover budget stays unused
    for i in range(n - 1, -1, -1):
        remaining_after = n - 1 - i  # apps after i, each needs >= 1 way
        for b in range(budget + 1):
            top = min(total_ways, b - remaining_after)
            if top < 1:
                continue  # infeasible: not enough ways left for app i
            chosen = None
            for a in range(1, top + 1):
                tail = best[i + 1][b - a]
                if tail is None:
                    continue
                cand = (cost[i][a] + tail[0], (a,) + tail[1])
                if chosen is None or cand < chosen:
                    chosen = cand
            best[i][b] = chosen

    final = best[0][budget]
    if final is None:  # unreachable given the checks above; defensive
        raise InfeasibleBudgetError(
            f"budget: {budget} way(s) infeasible for {n} application(s)"
        )
    total_cost, alloc = final
    per_app = tuple(curves[i][alloc[i] - 1] for i in range(n))
    return PlanResult(
        num_sets=num_sets,
        total_ways=total_ways,
        budget=budget,
        app_ids=tuple(t.app_id for t in apps),
        weights=ws,
        curves=curves,
        allocation=alloc,
        per_app_misses=per_app,
        weighted_misses=total_cost,
        used_ways=sum(alloc),
    )


# ---------------------------------------------------------------------------
# Fair allocation planner (lexicographic max-ratio / weighted-sum / vector)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FairPlanResult:
    """Result of plan_fair_allocation.  All fields are immutable."""

    num_sets: int
    total_ways: int
    budget: int
    app_ids: tuple[str, ...]              # sorted by app id
    weights: tuple[int, ...]              # aligned with app_ids
    references: tuple[int, ...]           # reference miss counts, aligned with app_ids
    curves: tuple[tuple[int, ...], ...]   # curves[i][w-1] = misses of app i with w ways
    allocation: tuple[int, ...]           # ways per app, aligned with app_ids
    per_app_misses: tuple[int, ...]       # misses at the allocated way count
    max_ratio_num: int                    # exact numerator of max misses/reference
    max_ratio_den: int                    # exact denominator (reduced fraction)
    max_ratio_apps: tuple[str, ...]       # app ids attaining the max ratio
    weighted_misses: int                  # secondary objective value
    used_ways: int                        # sum(allocation), always <= budget

    def allocation_map(self):
        """Allocation as an insertion-ordered dict keyed by app id."""
        return dict(zip(self.app_ids, self.allocation))

    @property
    def max_ratio(self):
        """The minimised max misses/reference as an exact Fraction."""
        return Fraction(self.max_ratio_num, self.max_ratio_den)


def plan_fair_allocation(traces, weights, references, num_sets, total_ways,
                         budget):
    """Choose integer way counts minimising relative loss, lexicographically.

    Objectives, in strict lexicographic order:

    1. minimise max_i (misses_i / references_i), compared with exact
       rational arithmetic (fractions.Fraction); two distinct allocations
       are never treated as tied by a floating-point epsilon;
    2. minimise the weighted miss sum sum_i weights_i * misses_i;
    3. take the lexicographically smallest allocation vector (apps ordered
       by app id).

    Parameters
    ----------
    traces     : iterable of AppTrace (or (app_id, blocks) pairs), 1..6 apps,
                 unique app ids.
    weights    : iterable of positive ints aligned with traces order
                 (used only for the secondary objective).
    references : iterable of positive ints aligned with traces order;
                 the reference miss count each application's misses are
                 normalised against.
    num_sets, total_ways, budget : same contract as plan_allocation.

    Every application still receives at least one way (the fair mode does
    not disable applications); budget < n_apps is proved infeasible and
    raises InfeasibleBudgetError, exactly as in the cost-minimising entry
    point.  Miss curves are not assumed convex.

    Returns a FairPlanResult.
    """
    apps_in = _normalize_traces(traces)
    ws_in = _normalize_weights(weights, [t.app_id for t in apps_in])
    refs_in = _normalize_references(references, [t.app_id for t in apps_in])
    num_sets = _require_num_sets(num_sets)
    total_ways = _require_int(total_ways, "total_ways", 1, MAX_WAYS)
    budget = _require_int(budget, "budget", 1, total_ways)

    paired = sorted(zip(apps_in, ws_in, refs_in), key=lambda p: p[0].app_id)
    apps = tuple(p[0] for p in paired)
    ws = tuple(p[1] for p in paired)
    refs = tuple(p[2] for p in paired)
    n = len(apps)
    if budget < n:
        raise InfeasibleBudgetError(
            f"budget: {budget} way(s) cannot give each of the {n} "
            f"application(s) its mandatory minimum of 1 way"
        )

    curves = tuple(miss_curve(t.blocks, num_sets, total_ways) for t in apps)
    # ratio[i][a] = exact relative loss of app i with a ways;
    # wcost[i][a] = weighted misses of app i with a ways (a in 1..total_ways).
    ratio = [
        [Fraction(0) if a == 0 else Fraction(curves[i][a - 1], refs[i])
         for a in range(total_ways + 1)]
        for i in range(n)
    ]
    wcost = [
        [ws[i] * curves[i][a - 1] for a in range(total_ways + 1)]
        for i in range(n)
    ]

    # best[i][b] = (max_ratio, weighted_sum, allocation_tuple) for apps
    # i..n-1 with at most b ways, or None when infeasible.  Tuple
    # comparison implements the three-level lexicographic objective
    # exactly (Fractions compare without any tolerance).
    best = [[None] * (budget + 1) for _ in range(n + 1)]
    for b in range(budget + 1):
        best[n][b] = (Fraction(0), 0, ())
    for i in range(n - 1, -1, -1):
        remaining_after = n - 1 - i
        for b in range(budget + 1):
            top = min(total_ways, b - remaining_after)
            if top < 1:
                continue
            chosen = None
            for a in range(1, top + 1):
                tail = best[i + 1][b - a]
                if tail is None:
                    continue
                cand = (
                    max(ratio[i][a], tail[0]),
                    wcost[i][a] + tail[1],
                    (a,) + tail[2],
                )
                if chosen is None or cand < chosen:
                    chosen = cand
            best[i][b] = chosen

    final = best[0][budget]
    if final is None:  # unreachable given the checks above; defensive
        raise InfeasibleBudgetError(
            f"budget: {budget} way(s) infeasible for {n} application(s)"
        )
    max_ratio, weighted_sum, alloc = final
    per_app = tuple(curves[i][alloc[i] - 1] for i in range(n))
    attaining = tuple(
        apps[i].app_id
        for i in range(n)
        if Fraction(per_app[i], refs[i]) == max_ratio
    )
    return FairPlanResult(
        num_sets=num_sets,
        total_ways=total_ways,
        budget=budget,
        app_ids=tuple(t.app_id for t in apps),
        weights=ws,
        references=refs,
        curves=curves,
        allocation=alloc,
        per_app_misses=per_app,
        max_ratio_num=max_ratio.numerator,
        max_ratio_den=max_ratio.denominator,
        max_ratio_apps=attaining,
        weighted_misses=weighted_sum,
        used_ways=sum(alloc),
    )


# ---------------------------------------------------------------------------
# Replay on isolated caches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayStats:
    """Statistics of one application replayed on its isolated cache."""

    app_id: str
    ways: int
    accesses: int
    hits: int
    misses: int
    hit_rate: float  # hits / accesses, exact IEEE-754 double in [0.0, 1.0]


def replay_isolated(traces, allocation, num_sets):
    """Replay every app on a private LRU cache of its allocated ways.

    ``allocation`` maps app_id -> ways (int >= 1) and must cover exactly the
    app ids in ``traces``.  Returns a tuple of :class:`ReplayStats` ordered
    by app id.  The replay is independent of the planning run (cold caches),
    so its miss counts must match the planned curves — a consistency check
    the test-suite performs rather than assumes.
    """
    apps = _normalize_traces(traces)
    num_sets = _require_num_sets(num_sets)
    if not hasattr(allocation, "items"):
        raise ValidationError(
            f"allocation: expected a mapping app_id -> ways, "
            f"got {type(allocation).__name__}"
        )
    alloc = dict(allocation)
    want = {t.app_id for t in apps}
    got = set(alloc)
    if got != want:
        raise ValidationError(
            f"allocation: app ids {sorted(got)} do not match trace app ids "
            f"{sorted(want)}"
        )
    for app_id, ways in alloc.items():
        _require_int(ways, f"allocation[{app_id!r}]", 1, MAX_WAYS)

    out = []
    for t in sorted(apps, key=lambda t: t.app_id):
        ways = alloc[t.app_id]
        misses = simulate_lru_misses(t.blocks, num_sets, ways)
        accesses = len(t.blocks)
        hits = accesses - misses
        out.append(
            ReplayStats(
                app_id=t.app_id,
                ways=ways,
                accesses=accesses,
                hits=hits,
                misses=misses,
                hit_rate=hits / accesses,
            )
        )
    return tuple(out)
