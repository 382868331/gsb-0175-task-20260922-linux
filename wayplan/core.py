"""Cache way-partition planning with performance isolation.

Pipeline (all exact, no convexity assumption, no greedy heuristic):

1. ``simulate_lru_misses`` / ``miss_curve`` — for each application, simulate
   a set-associative LRU cache independently at every way count
   1..total_ways and record the miss count (the "per-way curve").
2. ``plan_allocation`` — exact dynamic program over (app, remaining budget).
   Each application either is *disabled* (only when explicitly declared
   disableable: it then pays its integer disable penalty, occupies no ways,
   and is reported separately) or receives an integer way count >= 1 whose
   miss count must respect the application's optional hard miss cap.  The
   DP minimises, in this order: total objective cost (weighted miss sum
   plus disable penalties), number of disabled applications, and the
   lexicographic allocation vector with applications ordered by app id.
   The constrained problem is solved directly in one pass — the
   unconstrained optimum is never computed first and then patched, because
   miss curves are not guaranteed convex and local repair can be wrong.
3. ``verify_plan`` — re-run each active application on its own isolated
   cache sized to its allocated ways and confirm every hard constraint
   (caps, budget, disable rules, objective arithmetic) against the plan.

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


def _normalize_optional_ints(values, name, count):
    """Validate an optional per-application int vector.

    ``None`` (the whole parameter) means "not given" and yields all-None.
    Each entry is either None (feature absent for that application) or a
    non-negative int; bool/float/NaN are rejected via ``_require_int``.
    Entries are aligned with the traces *input* order, like weights.
    """
    if values is None:
        return (None,) * count
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValidationError(
            f"{name}: expected None or an iterable of {count} entries "
            f"(None or non-negative int), got {type(values).__name__}"
        )
    vs = list(values)
    if len(vs) != count:
        raise ValidationError(
            f"{name}: expected {count} entries (one per application, "
            f"aligned with traces order), got {len(vs)}"
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
    """Result of :func:`plan_allocation`.  All fields are immutable."""

    num_sets: int
    total_ways: int
    budget: int
    app_ids: tuple[str, ...]              # sorted by app id
    weights: tuple[int, ...]              # aligned with app_ids
    curves: tuple[tuple[int, ...], ...]   # curves[i][w-1] = misses of app i with w ways
    allocation: tuple[int, ...]           # ways per app (0 = disabled), aligned with app_ids
    per_app_misses: tuple                 # misses at the allocated ways; None for disabled apps
    weighted_misses: int                  # weighted miss sum over active apps
    used_ways: int                        # sum(allocation), always <= budget
    miss_caps: tuple                      # hard miss cap per app (None = uncapped)
    disable_penalties: tuple              # disable penalty per app (None = may not disable)
    disabled: tuple[bool, ...]            # disabled[i] iff allocation[i] == 0
    disabled_app_ids: tuple[str, ...]     # disabled apps, reported separately
    disable_penalty_total: int            # sum of penalties of disabled apps
    objective: int                        # weighted_misses + disable_penalty_total (minimised)

    def allocation_map(self):
        """Allocation as an insertion-ordered dict keyed by app id."""
        return dict(zip(self.app_ids, self.allocation))


def _infeasible_error(apps, caps, pens, curves, total_ways, budget):
    """Build the proved-infeasible error with a diagnostic snapshot.

    The per-app independent minimum way requirement is: 0 for disableable
    apps (disabling needs no ways), otherwise the smallest way count whose
    miss count meets the cap (1 when uncapped), or None when the cap is
    unreachable even with ``total_ways`` ways.  Summed against the budget
    this is a *necessary* condition only — see the errors module docstring.
    """
    n = len(apps)
    app_ids = tuple(t.app_id for t in apps)
    min_ways = []
    notes = []
    for i in range(n):
        if pens[i] is not None:
            min_ways.append(0)
            continue
        cap = caps[i]
        if cap is None:
            min_ways.append(1)
            continue
        feasible = [a for a in range(1, total_ways + 1)
                    if curves[i][a - 1] <= cap]
        if feasible:
            min_ways.append(min(feasible))
        else:
            min_ways.append(None)
            notes.append(
                f"app {app_ids[i]!r} cannot meet miss cap {cap} even with "
                f"{total_ways} ways (best achievable: {min(curves[i])} misses)"
            )
    required_total = None if any(m is None for m in min_ways) else sum(min_ways)

    def fmt(i):
        if min_ways[i] is None:
            return f"{app_ids[i]}=unachievable-cap"
        if pens[i] is not None:
            return f"{app_ids[i]}=0 (disableable)"
        return f"{app_ids[i]}={min_ways[i]}"

    parts = [
        f"budget: {budget} way(s) cannot satisfy all hard constraints for "
        f"the {n} application(s) — proved infeasible by the exact, complete "
        f"planner (not a search limit)",
        "per-app independent minimum way requirements: "
        + ", ".join(fmt(i) for i in range(n)),
    ]
    if required_total is not None:
        parts.append(
            f"total minimum {required_total} way(s) vs budget {budget}"
        )
    parts.extend(notes)
    parts.append(
        "the per-app-minimum vs budget comparison is only a necessary "
        "condition, not by itself a general infeasibility proof"
    )
    return InfeasibleBudgetError(
        "; ".join(parts),
        app_ids=app_ids,
        min_ways=tuple(min_ways),
        required_total=required_total,
        budget=budget,
    )


def plan_allocation(traces, weights, num_sets, total_ways, budget,
                    miss_caps=None, disable_penalties=None):
    """Choose way counts (or disabling) minimising the total objective cost.

    Parameters
    ----------
    traces     : iterable of AppTrace (or (app_id, blocks) pairs), 1..6 apps,
                 unique app ids.  Apps are processed sorted by app_id, which
                 defines the allocation-vector order used for tie-breaking.
    weights    : iterable of positive ints aligned with ``traces`` order.
    num_sets   : 1..16, sets of the (shared) cache being partitioned.
    total_ways : 1..12, associativity of the cache; miss curves are
                 computed for 1..total_ways ways per app.
    budget     : 0..total_ways, maximum total ways handed out.  Budget 0 is
                 feasible only when every application is disableable.
    miss_caps  : optional iterable aligned with ``traces`` order; each entry
                 is None (no cap) or a non-negative int — a *hard* cap on
                 that application's miss count whenever the application is
                 active.  Allocations violating any cap are infeasible, not
                 merely penalised.
    disable_penalties : optional iterable aligned with ``traces`` order;
                 each entry is None (the application may NOT be disabled and
                 keeps its mandatory minimum of 1 way) or a non-negative int
                 penalty paid when the application is disabled.  A disabled
                 application occupies 0 ways and is reported separately.

    Objective (exact, single constrained DP — never "unconstrained optimum
    plus local repair", which non-convex miss curves can invalidate):
    minimise, in lexicographic order of the tuple

        (weighted miss sum + disable penalties, disabled count, allocation)

    where the allocation vector lists ways per app (0 = disabled) with apps
    ordered by app id.

    Returns a :class:`PlanResult`.

    Raises
    ------
    ValidationError        — any contract violation in the inputs.
    InfeasibleBudgetError  — no allocation satisfies every hard constraint
                             (proved by the exact, complete DP; carries a
                             necessary-condition diagnostic snapshot).
    """
    apps_in = _normalize_traces(traces)
    ws_in = _normalize_weights(weights, [t.app_id for t in apps_in])
    caps_in = _normalize_optional_ints(miss_caps, "miss_caps", len(apps_in))
    pens_in = _normalize_optional_ints(
        disable_penalties, "disable_penalties", len(apps_in)
    )
    num_sets = _require_num_sets(num_sets)
    total_ways = _require_int(total_ways, "total_ways", 1, MAX_WAYS)
    budget = _require_int(budget, "budget", 0, total_ways)

    # Pair each trace with its weight/cap/penalty *before* sorting, so they
    # stay aligned with the application they were supplied for.
    paired = sorted(zip(apps_in, ws_in, caps_in, pens_in),
                    key=lambda p: p[0].app_id)
    apps = tuple(p[0] for p in paired)
    ws = tuple(p[1] for p in paired)
    caps = tuple(p[2] for p in paired)
    pens = tuple(p[3] for p in paired)
    n = len(apps)

    curves = tuple(miss_curve(t.blocks, num_sets, total_ways) for t in apps)

    # options[i] = (ways, cost, is_disabled) choices for app i.  Disabling
    # is offered only when a penalty was declared; an active way count is
    # offered only when it respects the app's hard miss cap.
    options = []
    for i in range(n):
        opts = []
        if pens[i] is not None:
            opts.append((0, pens[i], 1))
        for a in range(1, total_ways + 1):
            if caps[i] is None or curves[i][a - 1] <= caps[i]:
                opts.append((a, ws[i] * curves[i][a - 1], 0))
        options.append(opts)

    # best[i][b] = (min_cost, disabled_count, allocation_tuple) for apps
    # i..n-1 with a total of at most b ways, or None when infeasible.
    # Processed back to front; tuple comparison gives the exact objective
    # ordering (cost, disabled count, lexicographic allocation) in one pass.
    best = [[None] * (budget + 1) for _ in range(n + 1)]
    for b in range(budget + 1):
        best[n][b] = (0, 0, ())  # no apps left: leftover budget stays unused
    for i in range(n - 1, -1, -1):
        for b in range(budget + 1):
            chosen = None
            for a, c, d in options[i]:
                if a > b:
                    continue
                tail = best[i + 1][b - a]
                if tail is None:
                    continue
                cand = (c + tail[0], d + tail[1], (a,) + tail[2])
                if chosen is None or cand < chosen:
                    chosen = cand
            best[i][b] = chosen

    final = best[0][budget]
    if final is None:
        raise _infeasible_error(apps, caps, pens, curves, total_ways, budget)
    total_cost, _disabled_count, alloc = final

    disabled = tuple(a == 0 for a in alloc)
    per_app = tuple(
        None if disabled[i] else curves[i][alloc[i] - 1] for i in range(n)
    )
    weighted = sum(
        ws[i] * curves[i][alloc[i] - 1] for i in range(n) if not disabled[i]
    )
    penalty_total = sum(pens[i] for i in range(n) if disabled[i])
    return PlanResult(
        num_sets=num_sets,
        total_ways=total_ways,
        budget=budget,
        app_ids=tuple(t.app_id for t in apps),
        weights=ws,
        curves=curves,
        allocation=alloc,
        per_app_misses=per_app,
        weighted_misses=weighted,
        used_ways=sum(alloc),
        miss_caps=caps,
        disable_penalties=pens,
        disabled=disabled,
        disabled_app_ids=tuple(
            apps[i].app_id for i in range(n) if disabled[i]
        ),
        disable_penalty_total=penalty_total,
        objective=total_cost,
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

    Disabled applications (0 ways) are not accepted here: they own no cache
    to replay.  Use :func:`verify_plan` for plans that may contain disabled
    applications.
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
# Plan verification by independent replay
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintCheck:
    """One hard-constraint check inside a :class:`VerificationReport`."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of :func:`verify_plan`: replay stats plus every check."""

    app_ids: tuple[str, ...]
    replay: tuple[ReplayStats, ...]       # active apps only, ordered by app id
    disabled_app_ids: tuple[str, ...]     # disabled apps, reported separately
    checks: tuple[ConstraintCheck, ...]

    @property
    def ok(self):
        """True iff every hard-constraint check passed."""
        return all(c.ok for c in self.checks)


def verify_plan(traces, plan):
    """Re-verify a :class:`PlanResult` by actual replay on isolated caches.

    Every hard constraint is re-checked from the traces and the plan alone
    (cold-cache replay, independent of the planning run):

    - ``allocation_shape``  — one entry per app, each 0..total_ways ways;
    - ``budget_respected``  — used ways do not exceed the budget;
    - ``disable_rules``     — exactly the declared-disableable apps may be
      disabled (0 ways); every other app has at least 1 way;
    - ``caps_respected``    — each active app's *replayed* miss count meets
      its hard miss cap;
    - ``replay_matches_plan`` — replayed misses equal the planned curves;
    - ``objective_arithmetic`` — weighted misses, penalties and objective
      recomputed from the replay match the plan's fields.

    Returns a :class:`VerificationReport`.  Structural contract violations
    in the arguments (bad traces, non-PlanResult plan, app id mismatch)
    raise :class:`ValidationError` instead of producing a report.
    """
    apps = _normalize_traces(traces)
    if not isinstance(plan, PlanResult):
        raise ValidationError(
            f"plan: expected a PlanResult, got {type(plan).__name__}"
        )
    ids = tuple(sorted(t.app_id for t in apps))
    if tuple(plan.app_ids) != ids:
        raise ValidationError(
            f"plan: app_ids {list(plan.app_ids)} do not match the traces' "
            f"app ids {list(ids)}"
        )
    blocks = {t.app_id: t.blocks for t in apps}
    n = len(ids)
    checks = []

    def check(name, ok, detail):
        checks.append(ConstraintCheck(name, bool(ok), detail))

    alloc = tuple(plan.allocation)
    shape_ok = (
        len(alloc) == n
        and all(type(a) is int for a in alloc)
        and all(0 <= a <= plan.total_ways for a in alloc)
        and len(plan.disabled) == n
        and all(type(d) is bool for d in plan.disabled)
        and len(plan.miss_caps) == n
        and len(plan.disable_penalties) == n
        and len(plan.per_app_misses) == n
        and len(plan.weights) == n
        and len(plan.curves) == n
    )
    check("allocation_shape", shape_ok,
          f"{n} app(s), each 0..{plan.total_ways} ways (0 = disabled)")
    if not shape_ok:
        for name in ("budget_respected", "disable_rules", "caps_respected",
                     "replay_matches_plan", "objective_arithmetic"):
            check(name, False, "skipped: malformed plan shape")
        return VerificationReport(ids, (), tuple(), tuple(checks))

    used = sum(alloc)
    check("budget_respected",
          plan.used_ways == used and used <= plan.budget,
          f"used {used} way(s) of budget {plan.budget}")

    disabled_idx = [i for i in range(n) if alloc[i] == 0]
    declared = tuple(ids[i] for i in disabled_idx)
    dr_ok = all(plan.disabled[i] == (alloc[i] == 0) for i in range(n))
    dr_ok &= all(plan.disable_penalties[i] is not None for i in disabled_idx)
    dr_ok &= all(alloc[i] >= 1 for i in range(n)
                 if plan.disable_penalties[i] is None)
    dr_ok &= tuple(plan.disabled_app_ids) == declared
    check("disable_rules", dr_ok,
          f"disabled apps: {declared or 'none'}; every non-disableable app "
          f"holds at least 1 way")

    # Actual replay of every active app on its own isolated cold cache.
    replay = []
    misses = {}
    for i in range(n):
        if alloc[i] == 0:
            continue
        m = simulate_lru_misses(blocks[ids[i]], plan.num_sets, alloc[i])
        misses[i] = m
        accesses = len(blocks[ids[i]])
        hits = accesses - m
        replay.append(ReplayStats(ids[i], alloc[i], accesses, hits, m,
                                  hits / accesses))
    replay = tuple(replay)

    violated = [ids[i] for i in misses
                if plan.miss_caps[i] is not None
                and misses[i] > plan.miss_caps[i]]
    check("caps_respected", not violated,
          "every hard miss cap holds under independent replay"
          if not violated
          else f"caps violated under replay by: {violated}")

    rm_ok = all(plan.per_app_misses[i] == misses[i] for i in misses)
    rm_ok &= all(plan.per_app_misses[i] is None for i in disabled_idx)
    rm_ok &= all(plan.curves[i][alloc[i] - 1] == misses[i] for i in misses)
    check("replay_matches_plan", rm_ok,
          "replayed miss counts equal the planned per-way curve values")

    weighted = sum(plan.weights[i] * misses[i] for i in misses)
    penalties = sum(plan.disable_penalties[i] for i in disabled_idx)
    obj_ok = (plan.weighted_misses == weighted
              and plan.disable_penalty_total == penalties
              and plan.objective == weighted + penalties)
    check("objective_arithmetic", obj_ok,
          f"objective {plan.objective} = weighted misses {weighted} "
          f"+ disable penalties {penalties}")

    return VerificationReport(ids, replay, declared, tuple(checks))
