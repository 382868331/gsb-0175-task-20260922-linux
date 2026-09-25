"""Cache way-partition planning with performance isolation.

Pipeline (all exact, no convexity assumption, no greedy heuristic):

1. ``simulate_lru_misses`` / ``miss_curve`` — for each application, simulate
   a set-associative LRU cache independently at every way count
   1..total_ways and record the miss count (the "per-way curve").
2. ``plan_allocation`` — exact dynamic program over (app, remaining budget)
   choosing, per application, either an integer way count >= 1 or — only
   when the application was explicitly declared deactivatable — deactivation
   at a declared integer penalty.  Optional per-application hard miss caps
   are enforced *inside* the DP (an allocation that violates any cap is
   simply not a candidate); they are never handled by computing an
   unconstrained optimum first and repairing it locally, which would be
   unsound because miss curves are not guaranteed to be convex.
   Optimisation order (lexicographic): (1) objective = weighted miss sum
   of active applications + deactivation penalties, (2) number of
   deactivated applications, (3) allocation vector with applications
   ordered by app id (0 = deactivated, which sorts before any way count).
3. ``replay_isolated`` / ``verify_plan`` — re-run applications on their own
   isolated caches.  ``verify_plan`` takes a :class:`PlanResult` (which
   retains the full final state of the planner) and confirms every hard
   constraint — way budget, per-app miss caps, deactivation declarations,
   objective arithmetic — by actual replay on cold isolated caches.

Address model: the input is a sequence of *block addresses* (non-negative
ints, block granularity — the block offset is already removed).  A block
``b`` maps to set ``b % num_sets`` and uses the full block id as its tag.
Each simulated cache is cold (empty) at the start of a run.
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


def _normalize_optional_per_app(values, n, name):
    """Normalise an optional per-application parameter.

    ``None`` (the whole parameter) means "not declared for any application"
    and yields an all-``None`` tuple.  Otherwise ``values`` must be an
    iterable of exactly ``n`` entries aligned with the traces input order;
    each entry is either ``None`` (not declared for that application) or a
    non-negative int.
    """
    if values is None:
        return (None,) * n
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValidationError(
            f"{name}: expected None or an iterable of {n} entries "
            f"(None or non-negative int), got {type(values).__name__}"
        )
    vs = list(values)
    if len(vs) != n:
        raise ValidationError(
            f"{name}: expected {n} entries (one per application, aligned "
            f"with traces order), got {len(vs)}"
        )
    out = []
    for i, v in enumerate(vs):
        if v is None:
            out.append(None)
        else:
            out.append(_require_int(v, f"{name}[{i}]", 0))
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
    """Result of :func:`plan_allocation`.  All fields are immutable.

    The result retains the complete final state of the planner (curves,
    caps, penalties, allocation, objective components) so that
    :func:`verify_plan` — or any external checker — can re-validate every
    hard constraint by replay, without trusting the planner.
    """

    num_sets: int
    total_ways: int
    budget: int
    app_ids: tuple[str, ...]              # sorted by app id
    weights: tuple[int, ...]              # aligned with app_ids
    curves: tuple[tuple[int, ...], ...]   # curves[i][w-1] = misses of app i with w ways
    allocation: tuple[int, ...]           # ways per app (0 = deactivated), aligned with app_ids
    per_app_misses: tuple[int, ...]       # misses at the allocated way count (None if deactivated)
    weighted_misses: int                  # weighted miss sum over active apps
    used_ways: int                        # sum(allocation), always <= budget
    miss_caps: tuple                      # per-app hard miss cap (None = uncapped), aligned with app_ids
    deactivation_penalties: tuple         # per-app deactivation penalty (None = may not be deactivated)
    deactivated: tuple[bool, ...]         # True where the app is deactivated (allocation 0)
    deactivation_count: int               # number of deactivated apps
    deactivation_cost: int                # sum of penalties of deactivated apps
    objective: int                        # weighted_misses + deactivation_cost (primary minimised key)

    def allocation_map(self):
        """Allocation as an insertion-ordered dict keyed by app id.

        Deactivated applications appear with value 0.
        """
        return dict(zip(self.app_ids, self.allocation))

    def deactivated_app_ids(self):
        """Ids of deactivated applications, ordered by app id."""
        return tuple(
            app_id
            for app_id, off in zip(self.app_ids, self.deactivated)
            if off
        )


def _infeasible(app_ids, caps, pens, curves, total_ways, budget):
    """Build the proved-infeasible error with necessary-condition diagnostics."""
    per_app = []
    parts = []
    for i, app_id in enumerate(app_ids):
        cap = caps[i]
        if cap is None:
            min_ways = 1
        else:
            min_ways = next(
                (a for a in range(1, total_ways + 1) if curves[i][a - 1] <= cap),
                None,
            )
        deactivatable = pens[i] is not None
        per_app.append(
            {
                "app_id": app_id,
                "min_ways_if_active": min_ways,
                "deactivatable": deactivatable,
                "penalty": pens[i],
            }
        )
        if min_ways is None:
            part = f"{app_id}: miss cap {cap} unachievable with 1..{total_ways} way(s)"
        else:
            part = f"{app_id}: >= {min_ways} way(s)"
        if deactivatable:
            part += f" (or deactivate, penalty {pens[i]})"
        parts.append(part)
    diagnostics = {
        "budget": budget,
        "total_ways": total_ways,
        "per_app": tuple(per_app),
    }
    return InfeasibleBudgetError(
        f"budget: {budget} way(s) cannot satisfy {len(app_ids)} application(s) "
        f"under the declared hard miss caps and deactivation options; "
        f"per-app independent minimum requirements: {'; '.join(parts)}. "
        f"These minima are a necessary condition only, not a general "
        f"infeasibility proof; feasibility itself is decided by the exact, "
        f"complete planner (no search budget exists to exhaust).",
        diagnostics=diagnostics,
    )


def plan_allocation(traces, weights, num_sets, total_ways, budget,
                    miss_caps=None, deactivation_penalties=None):
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
    budget     : 0..total_ways, maximum total ways handed out.  0 is legal
                 since v1.1: it is feasible exactly when every application
                 is deactivatable.
    miss_caps  : optional; None, or an iterable aligned with ``traces``
                 order whose entries are None (no cap) or a non-negative
                 int hard cap on that application's miss count.  Caps are
                 enforced inside the exact DP — any allocation exceeding a
                 cap is infeasible, never repaired after the fact.
    deactivation_penalties : optional; None, or an iterable aligned with
                 ``traces`` order whose entries are None (application may
                 not be deactivated — it keeps its mandatory minimum of one
                 way) or a non-negative int penalty paid when the
                 application is deactivated.  A deactivated application
                 occupies no ways, contributes its penalty to the objective
                 instead of misses, and is reported separately.

    Returns a :class:`PlanResult`.  Optimisation order is lexicographic:
    (1) objective = weighted miss sum of active apps + deactivation
    penalties, (2) number of deactivated apps, (3) allocation vector with
    apps ordered by app id (0 = deactivated sorts before any way count).

    Raises
    ------
    ValidationError        — any contract violation in the inputs.
    InfeasibleBudgetError  — no allocation satisfies every hard constraint
                             within the budget (proved infeasible by the
                             exact, complete DP; the exception carries
                             per-app necessary-condition diagnostics).
    """
    apps_in = _normalize_traces(traces)
    n = len(apps_in)
    ws_in = _normalize_weights(weights, [t.app_id for t in apps_in])
    caps_in = _normalize_optional_per_app(miss_caps, n, "miss_caps")
    pens_in = _normalize_optional_per_app(
        deactivation_penalties, n, "deactivation_penalties"
    )
    num_sets = _require_num_sets(num_sets)
    total_ways = _require_int(total_ways, "total_ways", 1, MAX_WAYS)
    budget = _require_int(budget, "budget", 0, total_ways)

    # Pair each trace with its weight, cap and penalty *before* sorting, so
    # all per-app parameters stay aligned with their application.
    paired = sorted(
        zip(apps_in, ws_in, caps_in, pens_in), key=lambda p: p[0].app_id
    )
    apps = tuple(p[0] for p in paired)
    ws = tuple(p[1] for p in paired)
    caps = tuple(p[2] for p in paired)
    pens = tuple(p[3] for p in paired)

    curves = tuple(miss_curve(t.blocks, num_sets, total_ways) for t in apps)
    # cost[i][a] = weighted misses of app i when given a ways (a in 1..total_ways)
    cost = [
        [ws[i] * curves[i][a - 1] for a in range(total_ways + 1)]
        for i in range(n)
    ]

    # mandatory_after[i] = number of apps after i that may NOT be deactivated
    # and therefore each need >= 1 way.  Deactivatable apps need 0.
    mandatory_after = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        mandatory_after[i] = mandatory_after[i + 1] + (1 if pens[i] is None else 0)

    # best[i][b] = (objective, deactivation_count, allocation_tuple) for
    # apps i..n-1 with a total of at most b ways, or None when infeasible.
    # Processed back to front; tuple comparison gives the exact
    # lexicographic optimisation order (objective, deactivation count,
    # allocation vector) in one pass.
    best = [[None] * (budget + 1) for _ in range(n + 1)]
    for b in range(budget + 1):
        best[n][b] = (0, 0, ())  # no apps left: leftover budget stays unused
    for i in range(n - 1, -1, -1):
        for b in range(budget + 1):
            chosen = None
            # Option 1: deactivate app i (only if explicitly declared).
            if pens[i] is not None:
                tail = best[i + 1][b]
                if tail is not None:
                    chosen = (pens[i] + tail[0], 1 + tail[1], (0,) + tail[2])
            # Option 2: give app i a ways; its hard cap (if any) must hold.
            top = min(total_ways, b - mandatory_after[i + 1])
            cap = caps[i]
            for a in range(1, top + 1):
                if cap is not None and curves[i][a - 1] > cap:
                    continue  # hard cap: this way count is not a candidate
                tail = best[i + 1][b - a]
                if tail is None:
                    continue
                cand = (cost[i][a] + tail[0], tail[1], (a,) + tail[2])
                if chosen is None or cand < chosen:
                    chosen = cand
            best[i][b] = chosen

    final = best[0][budget]
    if final is None:
        raise _infeasible(
            tuple(t.app_id for t in apps), caps, pens, curves, total_ways, budget
        )
    objective, deact_count, alloc = final
    deactivated = tuple(a == 0 for a in alloc)
    per_app = tuple(
        None if deactivated[i] else curves[i][alloc[i] - 1] for i in range(n)
    )
    deact_cost = sum(pens[i] for i in range(n) if deactivated[i])
    return PlanResult(
        num_sets=num_sets,
        total_ways=total_ways,
        budget=budget,
        app_ids=tuple(t.app_id for t in apps),
        weights=ws,
        curves=curves,
        allocation=alloc,
        per_app_misses=per_app,
        weighted_misses=objective - deact_cost,
        used_ways=sum(alloc),
        miss_caps=caps,
        deactivation_penalties=pens,
        deactivated=deactivated,
        deactivation_count=deact_count,
        deactivation_cost=deact_cost,
        objective=objective,
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

    Deactivated applications (0 ways) are not accepted here; use
    :func:`verify_plan` to replay a plan that contains deactivations.
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


# ---------------------------------------------------------------------------
# Replay-based verification of a finished plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanVerification:
    """Outcome of :func:`verify_plan`.

    ``ok`` is True only when every hard constraint of the plan was
    confirmed by actual replay.  ``failures`` lists each violated or
    inconsistent constraint in locatable form.  ``replay`` holds the
    replay statistics of the active applications (ordered by app id);
    ``deactivated`` lists the deactivated application ids separately.
    """

    ok: bool
    failures: tuple[str, ...]
    replay: tuple[ReplayStats, ...]
    deactivated: tuple[str, ...]


def verify_plan(result, traces):
    """Re-check every hard constraint of a :class:`PlanResult` by replay.

    The planner's final state (curves, caps, penalties, allocation,
    objective components) is fully retained in ``result``; this function
    re-simulates each active application on a cold isolated cache of its
    allocated ways and confirms, independently of the planning run:

    - every active application's replayed misses equal the recorded
      ``per_app_misses`` and the recorded curve value at its way count;
    - every hard miss cap holds for the replayed miss count;
    - deactivated applications were explicitly declared deactivatable and
      occupy 0 ways; non-declared applications hold at least one way;
    - ``used_ways`` equals the allocation sum and does not exceed budget;
    - ``deactivation_count`` / ``deactivation_cost`` / ``weighted_misses``
      / ``objective`` are arithmetically consistent.

    Structural problems (``result`` is not a PlanResult, app ids do not
    match ``traces``, field lengths disagree) raise ``ValidationError``.
    Constraint violations found by replay do not raise; they are reported
    in the returned :class:`PlanVerification` with ``ok == False``.
    """
    if not isinstance(result, PlanResult):
        raise ValidationError(
            f"result: expected PlanResult, got {type(result).__name__}"
        )
    apps = _normalize_traces(traces)
    by_id = {t.app_id: t for t in apps}
    if tuple(sorted(by_id)) != result.app_ids:
        raise ValidationError(
            f"traces: app ids {sorted(by_id)} do not match "
            f"result.app_ids {list(result.app_ids)}"
        )
    n = len(result.app_ids)
    for field_name in (
        "weights", "curves", "allocation", "per_app_misses",
        "miss_caps", "deactivation_penalties", "deactivated",
    ):
        if len(getattr(result, field_name)) != n:
            raise ValidationError(
                f"result.{field_name}: length does not match result.app_ids"
            )

    failures = []
    replay = []
    deactivated_ids = []
    alloc_all_ints = True
    for i, app_id in enumerate(result.app_ids):
        t = by_id[app_id]
        a = result.allocation[i]
        if isinstance(a, bool) or not isinstance(a, int) or not 0 <= a <= result.total_ways:
            failures.append(
                f"app {app_id!r}: allocation {a!r} is not an int in "
                f"0..{result.total_ways}"
            )
            alloc_all_ints = False
            continue
        if bool(result.deactivated[i]) != (a == 0):
            failures.append(
                f"app {app_id!r}: deactivated flag {result.deactivated[i]!r} "
                f"disagrees with allocation {a}"
            )
        if a == 0:
            if result.deactivation_penalties[i] is None:
                failures.append(
                    f"app {app_id!r}: deactivated but no deactivation "
                    f"penalty was declared for it"
                )
            if result.per_app_misses[i] is not None:
                failures.append(
                    f"app {app_id!r}: deactivated but per_app_misses is "
                    f"{result.per_app_misses[i]!r}, expected None"
                )
            deactivated_ids.append(app_id)
            continue
        # Active app: confirm its constraint state by actual replay.
        misses = simulate_lru_misses(t.blocks, result.num_sets, a)
        if result.per_app_misses[i] != misses:
            failures.append(
                f"app {app_id!r}: recorded per_app_misses "
                f"{result.per_app_misses[i]!r} != replayed {misses}"
            )
        if a > len(result.curves[i]) or result.curves[i][a - 1] != misses:
            failures.append(
                f"app {app_id!r}: recorded curve disagrees with replay at "
                f"{a} way(s)"
            )
        cap = result.miss_caps[i]
        if cap is not None and misses > cap:
            failures.append(
                f"app {app_id!r}: hard miss cap violated — replayed "
                f"{misses} miss(es) > cap {cap}"
            )
        accesses = len(t.blocks)
        hits = accesses - misses
        replay.append(
            ReplayStats(
                app_id=app_id,
                ways=a,
                accesses=accesses,
                hits=hits,
                misses=misses,
                hit_rate=hits / accesses,
            )
        )

    if alloc_all_ints:
        if sum(result.allocation) != result.used_ways:
            failures.append(
                f"used_ways {result.used_ways} != sum of allocation "
                f"{sum(result.allocation)}"
            )
    if result.used_ways > result.budget:
        failures.append(
            f"used_ways {result.used_ways} exceeds budget {result.budget}"
        )
    if sum(1 for d in result.deactivated if d) != result.deactivation_count:
        failures.append(
            f"deactivation_count {result.deactivation_count} != number of "
            f"deactivated flags"
        )
    pen_total = sum(
        result.deactivation_penalties[i]
        for i in range(n)
        if result.deactivated[i]
        and isinstance(result.deactivation_penalties[i], int)
        and not isinstance(result.deactivation_penalties[i], bool)
    )
    if pen_total != result.deactivation_cost:
        failures.append(
            f"deactivation_cost {result.deactivation_cost} != sum of "
            f"penalties of deactivated apps {pen_total}"
        )
    if all(
        result.per_app_misses[i] is not None
        for i in range(n)
        if not result.deactivated[i]
    ):
        weighted = sum(
            result.weights[i] * result.per_app_misses[i]
            for i in range(n)
            if not result.deactivated[i]
        )
        if weighted != result.weighted_misses:
            failures.append(
                f"weighted_misses {result.weighted_misses} != recomputed "
                f"{weighted}"
            )
    if result.weighted_misses + result.deactivation_cost != result.objective:
        failures.append(
            f"objective {result.objective} != weighted_misses "
            f"{result.weighted_misses} + deactivation_cost "
            f"{result.deactivation_cost}"
        )

    return PlanVerification(
        ok=not failures,
        failures=tuple(failures),
        replay=tuple(replay),
        deactivated=tuple(deactivated_ids),
    )
