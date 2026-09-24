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

import itertools
import math
from collections import OrderedDict
from dataclasses import dataclass

from .errors import InfeasibleBudgetError, ValidationError

# Interface bounds fixed by the task contract.
MAX_APPS = 6
MAX_SETS = 16
MAX_WAYS = 12

# Interface bounds for the staged (known-sequence) planning model.
MAX_TASKS = 3
MAX_STAGES = 4
MAX_DYN_WAYS = 8


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


# ---------------------------------------------------------------------------
# Staged planning with known stage sequence and re-routing transition costs
# ---------------------------------------------------------------------------
#
# Model (the discrete staged problem; it does NOT include scenario trees,
# non-anticipative policies or robust lower bounds):
#
# * For each task, the ordered sequence of stages is known in advance, and
#   each stage carries a discrete loss curve loss[stage][ways] over a fixed
#   set of way counts 1..max_ways (at most MAX_DYN_WAYS).
# * At every stage each task must be assigned an integer way count satisfying
#   ``min_ways <= ways <= max_ways(stage)``; ``min_ways`` defaults to 1.
# * Changing a task's way count between two consecutive stages costs
#   ``migration_cost * |new_ways - old_ways|`` (linear per unit changed).
#   In addition every actual change (|new - old| > 0) pays a fixed
#   ``switch_cost`` (default 0); set both to 0 to disable transition costs.
# * ``max_change`` (default: unlimited) caps |new_ways - old_ways| between
#   consecutive stages.
#
# The planner is an exact dynamic program over stages with the current way
# vector as its state, and backtracks the unique lexicographically smallest
# optimum with stages outer and tasks (sorted by id) inner.  All quantities
# are exact integers.


@dataclass(frozen=True)
class TaskSpec:
    """One task's known stage sequence (validated, immutable).

    ``losses``   — tuple of ``num_stages`` curves, each a tuple of length
                   ``max_ways``; ``losses[s][w - 1]`` is the discrete
                   performance value (loss) at stage ``s`` when routed with
                   ``w`` ways.  Curves need not be monotone or convex.
    ``min_ways`` — optional per-stage minimum way requirement; either a
                   single positive int (applies to every stage) or a
                   sequence of length ``num_stages``.
    """

    task_id: str
    losses: tuple[tuple[int, ...], ...]
    min_ways: tuple[int, ...]

    def __init__(self, task_id, losses, min_ways=1):
        if not isinstance(task_id, str) or not task_id:
            raise ValidationError(
                f"task_id: expected a non-empty str, got {task_id!r}"
            )
        if isinstance(losses, (str, bytes)) or not hasattr(losses, "__iter__"):
            raise ValidationError(
                f"losses of task {task_id!r}: expected an iterable of stage "
                f"curves, got {type(losses).__name__}"
            )
        stages = []
        for s, curve in enumerate(losses):
            stages.append(
                _require_loss_curve(
                    curve, f"losses of task {task_id!r}, stage {s}"
                )
            )
        if not stages:
            raise ValidationError(
                f"losses of task {task_id!r}: at least one stage is required"
            )
        if len(stages) > MAX_STAGES:
            raise ValidationError(
                f"losses of task {task_id!r}: at most {MAX_STAGES} stages are "
                f"supported, got {len(stages)}"
            )
        if min_ways is None:
            mins = (1,) * len(stages)
        elif isinstance(min_ways, (str, bytes)) or not hasattr(
            min_ways, "__iter__"
        ):
            mins = (
                _require_int(
                    min_ways, f"min_ways of task {task_id!r}", 1, MAX_DYN_WAYS
                ),
            ) * len(stages)
        else:
            mins_raw = list(min_ways)
            if len(mins_raw) != len(stages):
                raise ValidationError(
                    f"min_ways of task {task_id!r}: expected 1 value or "
                    f"{len(stages)} values (one per stage), got {len(mins_raw)}"
                )
            mins = tuple(
                _require_int(
                    m, f"min_ways of task {task_id!r}, stage {s}",
                    1, MAX_DYN_WAYS,
                )
                for s, m in enumerate(mins_raw)
            )
        for s, (curve, lo) in enumerate(zip(stages, mins)):
            if lo > len(curve):
                raise ValidationError(
                    f"min_ways of task {task_id!r}, stage {s}: minimum {lo} "
                    f"exceeds the {len(curve)} way count(s) of that stage curve"
                )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "losses", tuple(stages))
        object.__setattr__(self, "min_ways", mins)


def _require_loss_curve(curve, name):
    """Validate one discrete per-way loss curve: non-negative ints, 1..8 ways."""
    if isinstance(curve, (str, bytes)) or not hasattr(curve, "__iter__"):
        raise ValidationError(
            f"{name}: expected an iterable of per-way loss values, "
            f"got {type(curve).__name__}"
        )
    out = []
    for w, value in enumerate(curve, start=1):
        out.append(_require_int(value, f"{name}[{w - 1}] (ways={w})", 0))
    if not out:
        raise ValidationError(f"{name}: curve must cover at least 1 way")
    if len(out) > MAX_DYN_WAYS:
        raise ValidationError(
            f"{name}: at most {MAX_DYN_WAYS} way counts are supported, "
            f"got {len(out)}"
        )
    return tuple(out)


def _normalize_tasks(tasks):
    """Validate the task collection; return tasks in input order."""
    if isinstance(tasks, (str, bytes)) or not hasattr(tasks, "__iter__"):
        raise ValidationError(
            f"tasks: expected an iterable of TaskSpec, "
            f"got {type(tasks).__name__}"
        )
    out = []
    for i, t in enumerate(tasks):
        if not isinstance(t, TaskSpec):
            raise ValidationError(
                f"tasks[{i}]: expected a TaskSpec instance, got "
                f"{type(t).__name__} ({t!r})"
            )
        out.append(t)
    if not 1 <= len(out) <= MAX_TASKS:
        raise ValidationError(
            f"tasks: expected 1..{MAX_TASKS} tasks, got {len(out)}"
        )
    ids = [t.task_id for t in out]
    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise ValidationError(f"tasks: duplicate task_id(s): {dupes}")
    num_stages = len(out[0].losses)
    max_ways = len(out[0].losses[0])
    for t in out[1:]:
        if len(t.losses) != num_stages:
            raise ValidationError(
                f"tasks: task {t.task_id!r} has {len(t.losses)} stage(s), "
                f"expected {num_stages} to match the other task(s)"
            )
        for s, curve in enumerate(t.losses):
            if len(curve) != max_ways:
                raise ValidationError(
                    f"losses of task {t.task_id!r}, stage {s}: curve has "
                    f"{len(curve)} way count(s), expected {max_ways} to match "
                    f"the other curves"
                )
    return tuple(out)


@dataclass(frozen=True)
class StagePlanResult:
    """Result of :func:`plan_stages`.  All fields are immutable."""

    task_ids: tuple[str, ...]                # tasks sorted by task id
    num_stages: int
    migration_cost: int
    switch_cost: int
    max_change: int                          # MAX_DYN_WAYS encodes "unlimited"
    ways: tuple[tuple[int, ...], ...]        # ways[task][stage]
    stage_losses: tuple[int, ...]            # summed per-way losses per stage
    transition_costs: tuple[int, ...]        # cost paid *into* stage s; [0]=0
    total_loss: int                          # sum of stage_losses
    total_transition_cost: int               # sum of transition_costs
    total_cost: int                          # minimised objective

    def ways_map(self):
        """``{task_id: (ways per stage)}`` keyed by task id."""
        return dict(zip(self.task_ids, self.ways))

    def stage_ways_map(self):
        """List with one ``{task_id: ways}`` allocation dict per stage."""
        return [dict(zip(self.task_ids, col)) for col in zip(*self.ways)]


def plan_stages(tasks, migration_cost=1, switch_cost=0, max_change=None):
    """Plan way counts over a known stage sequence, minimising total cost.

    Parameters
    ----------
    tasks          : iterable of 1..:data:`MAX_TASKS` :class:`TaskSpec`
                     instances sharing a stage count (1..:data:`MAX_STAGES`)
                     and curve length (1..:data:`MAX_DYN_WAYS`).
    migration_cost : non-negative int paid per unit of way-count change
                     between consecutive stages (linear in the difference).
    switch_cost    : non-negative int paid once per task per actual change.
    max_change     : positive int capping the way-count difference between
                     consecutive stages, or ``None`` for no cap.

    Returns a :class:`StagePlanResult`.  Among all plans attaining the
    minimum total cost, the lexicographically smallest flattened way vector
    is returned, with stages outer and tasks (sorted by id) inner, i.e. the
    plan keeps the smallest feasible way counts at the earliest differing
    stage/task.

    Raises
    ------
    ValidationError — any contract violation.  There is no separate
    infeasibility exception: every stage curve admits its own minimum way
    count and all curves share one length, so a validated instance is always
    feasible (``max_change`` cannot make stage 0 unreachable).
    """
    tasks_in = _normalize_tasks(tasks)
    migration_cost = _require_int(migration_cost, "migration_cost", 0)
    switch_cost = _require_int(switch_cost, "switch_cost", 0)
    if max_change is not None:
        max_change = _require_int(max_change, "max_change", 1, MAX_DYN_WAYS)
    else:
        max_change = MAX_DYN_WAYS  # >= any feasible difference, i.e. no cap

    ordered = tuple(sorted(tasks_in, key=lambda t: t.task_id))
    n = len(ordered)
    num_stages = len(ordered[0].losses)
    max_ways = len(ordered[0].losses[0])

    # Per-stage feasible way counts for each task: [min_ways, max_ways].
    feasible = [
        [
            tuple(range(t.min_ways[s], max_ways + 1))
            for s in range(num_stages)
        ]
        for t in ordered
    ]

    # Exact DP over stages.  State key: tuple of current way counts (1-based);
    # value: (best_cost, flattened_path).  Plain tuple comparison gives cost
    # ordering with lexicographic tie-breaking in a single comparison.
    states = {}
    for combo in itertools.product(*(feasible[i][0] for i in range(n))):
        loss = sum(ordered[i].losses[0][combo[i] - 1] for i in range(n))
        states[combo] = (loss, tuple(combo))

    for s in range(1, num_stages):
        new_states = {}
        for combo in itertools.product(*(feasible[i][s] for i in range(n))):
            best = None
            stage_loss = sum(
                ordered[i].losses[s][combo[i] - 1] for i in range(n)
            )
            for prev, (prev_cost, prev_path) in states.items():
                delta = 0
                for i in range(n):
                    d = abs(combo[i] - prev[i])
                    if d > max_change:
                        break  # infeasible transition: skip this predecessor
                    if d:
                        delta += migration_cost * d + switch_cost
                else:
                    cand = (prev_cost + stage_loss + delta, prev_path + combo)
                    if best is None or cand < best:
                        best = cand
            if best is not None:
                new_states[combo] = best
        states = new_states

    total_cost, flat = min(states.values())
    ways = tuple(
        tuple(flat[s * n + i] for s in range(num_stages))
        for i in range(n)
    )
    stage_losses = tuple(
        sum(ordered[i].losses[s][ways[i][s] - 1] for i in range(n))
        for s in range(num_stages)
    )
    transition_costs = [0]
    for s in range(1, num_stages):
        delta = 0
        for i in range(n):
            d = abs(ways[i][s] - ways[i][s - 1])
            if d:
                delta += migration_cost * d + switch_cost
        transition_costs.append(delta)
    return StagePlanResult(
        task_ids=tuple(t.task_id for t in ordered),
        num_stages=num_stages,
        migration_cost=migration_cost,
        switch_cost=switch_cost,
        max_change=max_change,
        ways=ways,
        stage_losses=tuple(stage_losses),
        transition_costs=tuple(transition_costs),
        total_loss=sum(stage_losses),
        total_transition_cost=sum(transition_costs),
        total_cost=total_cost,
    )
