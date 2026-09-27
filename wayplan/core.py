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
By default each simulated cache is cold (empty).  An explicit warm cache may
be supplied as one ordered block sequence per set, ordered LRU -> MRU.  Every
candidate allocation starts from an independent copy; reducing a candidate's
way count first evicts surplus LRU entries without counting those evictions as
subsequent-access misses.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

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


def _normalize_initial_cache(initial_cache, num_sets, name="initial_cache"):
    """Validate one warm cache: ``num_sets`` groups, each LRU -> MRU."""
    if initial_cache is None:
        return tuple(() for _ in range(num_sets))
    if isinstance(initial_cache, (str, bytes)) or not hasattr(
        initial_cache, "__iter__"
    ):
        raise ValidationError(
            f"{name}: expected {num_sets} iterable set groups, "
            f"got {type(initial_cache).__name__}"
        )
    groups = list(initial_cache)
    if len(groups) != num_sets:
        raise ValidationError(
            f"{name}: expected {num_sets} set group(s) to match num_sets, "
            f"got {len(groups)}"
        )

    normalized = []
    for set_index, group in enumerate(groups):
        group_name = f"{name}[{set_index}]"
        if isinstance(group, (str, bytes)) or not hasattr(group, "__iter__"):
            raise ValidationError(
                f"{group_name}: expected an iterable sequence of blocks "
                f"ordered LRU -> MRU, got {type(group).__name__}"
            )
        entries = []
        seen = set()
        for j, block in enumerate(group):
            block = _require_int(block, f"{group_name}[{j}]", 0)
            expected_set = block % num_sets
            if expected_set != set_index:
                raise ValidationError(
                    f"{group_name}[{j}]: block {block} maps to set "
                    f"{expected_set}, not set {set_index}"
                )
            if block in seen:
                raise ValidationError(
                    f"{group_name}[{j}]: duplicate initial block {block} "
                    f"is not a valid cache state"
                )
            seen.add(block)
            entries.append(block)
        if len(entries) > MAX_WAYS:
            raise ValidationError(
                f"{group_name}: initial group has {len(entries)} blocks, "
                f"above MAX_WAYS={MAX_WAYS}"
            )
        normalized.append(tuple(entries))
    return tuple(normalized)


def _normalize_initial_caches(initial_caches, apps, num_sets):
    """Validate an app_id -> warm-cache mapping; absent apps start cold."""
    if initial_caches is None:
        return {app.app_id: tuple(() for _ in range(num_sets)) for app in apps}
    if not hasattr(initial_caches, "items"):
        raise ValidationError(
            f"initial_caches: expected a mapping app_id -> cache groups, "
            f"got {type(initial_caches).__name__}"
        )

    by_app = dict(initial_caches)
    known = {app.app_id for app in apps}
    unknown = sorted(set(by_app) - known)
    if unknown:
        raise ValidationError(
            f"initial_caches: unknown app_id(s) {unknown}; valid ids are "
            f"{sorted(known)}"
        )

    return {
        app.app_id: _normalize_initial_cache(
            by_app.get(app.app_id),
            num_sets,
            f"initial_caches[{app.app_id!r}]",
        )
        for app in apps
    }


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


# ---------------------------------------------------------------------------
# LRU simulation
# ---------------------------------------------------------------------------

def simulate_lru_misses(blocks, num_sets, num_ways, initial_cache=None):
    """Exact miss count of a set-associative LRU cache.

    ``blocks``  — iterable of non-negative int block addresses.
    ``num_sets``— 1..MAX_SETS; block b maps to set ``b % num_sets``.
    ``num_ways``— 1..MAX_WAYS; associativity of every set.
    ``initial_cache``— optional ``num_sets`` sequences, each listing initial
                   blocks from LRU to MRU.  If a sequence is longer than
                   ``num_ways`` its LRU surplus is evicted before replay.
    """
    num_sets = _require_num_sets(num_sets)
    num_ways = _require_int(num_ways, "num_ways", 1, MAX_WAYS)
    blocks = _require_blocks(blocks)
    initial_cache = _normalize_initial_cache(initial_cache, num_sets)

    # Build independent LRU->MRU OrderedDicts, so candidate simulations never
    # observe mutations made by another candidate.
    sets = []
    for group in initial_cache:
        retained = group[-num_ways:] if len(group) > num_ways else group
        sets.append(OrderedDict((block, None) for block in retained))
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


def miss_curve(blocks, num_sets, max_ways, initial_cache=None):
    """Miss counts for every way count 1..max_ways.

    Returns a tuple of length ``max_ways``; element ``[w - 1]`` is the miss
    count with ``w`` ways.  No monotonicity or convexity is assumed.  Every
    point on the curve starts from an independent copy of ``initial_cache``.
    """
    max_ways = _require_int(max_ways, "max_ways", 1, MAX_WAYS)
    blocks = _require_blocks(blocks)
    num_sets = _require_num_sets(num_sets)
    initial_cache = _normalize_initial_cache(initial_cache, num_sets)
    return tuple(
        simulate_lru_misses(blocks, num_sets, w, initial_cache)
        for w in range(1, max_ways + 1)
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
    initial_caches: tuple[tuple[tuple[int, ...], ...], ...] = ()  # per app, LRU -> MRU

    def allocation_map(self):
        """Allocation as an insertion-ordered dict keyed by app id."""
        return dict(zip(self.app_ids, self.allocation))

    def initial_caches_map(self):
        """Initial cache groups as an insertion-ordered app-id mapping."""
        return dict(zip(self.app_ids, self.initial_caches))


def plan_allocation(
    traces, weights, num_sets, total_ways, budget, initial_caches=None
):
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
    initial_caches : optional mapping app_id -> ``num_sets`` ordered groups.
                 Each group lists blocks from LRU to MRU.  Missing apps start
                 cold.  Every candidate way count receives an independent
                 copy; shrinking to that way count evicts LRU entries first.

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

    warm = _normalize_initial_caches(initial_caches, apps, num_sets)
    for app in apps:
        for set_index, group in enumerate(warm[app.app_id]):
            if len(group) > total_ways:
                raise ValidationError(
                    f"initial_caches[{app.app_id!r}][{set_index}]: initial "
                    f"group has {len(group)} blocks but total_ways is only "
                    f"{total_ways}"
                )

    curves = tuple(
        miss_curve(t.blocks, num_sets, total_ways, warm[t.app_id])
        for t in apps
    )
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
        initial_caches=tuple(warm[t.app_id] for t in apps),
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


def replay_isolated(traces, allocation, num_sets, initial_caches=None):
    """Replay every app on a private LRU cache of its allocated ways.

    ``allocation`` maps app_id -> ways (int >= 1) and must cover exactly the
    app ids in ``traces``.  ``initial_caches`` optionally supplies each warm
    cache as LRU -> MRU groups.  Returns a tuple of :class:`ReplayStats`
    ordered by app id.  Replay uses an independent copy of the initial state
    and leaves the supplied input unchanged.
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
    warm = _normalize_initial_caches(initial_caches, apps, num_sets)

    out = []
    for t in sorted(apps, key=lambda t: t.app_id):
        ways = alloc[t.app_id]
        initial_cache = warm[t.app_id]
        misses = simulate_lru_misses(
            t.blocks, num_sets, ways, initial_cache
        )
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
