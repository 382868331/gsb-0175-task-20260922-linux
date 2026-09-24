"""Multi-stage per-task way planning with switching (migration) cost.

Each task owns a *known stage sequence* (length 1..MAX_STAGES).  At every
stage the task receives a discrete loss curve over way counts
(``loss_curve[w - 1]`` = loss when run with ``w`` ways,
``1 <= w <= num_ways``) plus two stage constraints:

- ``min_ways``  — the stage's mandatory minimum way count;
- ``max_loss``  — the minimum-performance requirement expressed in loss
  units: a configuration is acceptable iff
  ``loss_curve[w - 1] <= max_loss`` (performance at least the required
  floor is equivalent to loss at most the corresponding ceiling).

Changing the way count between two consecutive stages is a *migration*:

- it is charged a fixed switching cost ``migration_cost`` when the way
  count actually changes (zero when it stays);
- it is allowed only when ``|w_next - w_prev| <= max_change``.

The planner minimises, independently per task,

    sum over stages of loss_t(w_t)
  + sum over stage boundaries of migration_cost * [w_t != w_{t-1}]

with an exact, complete dynamic program over (stage, way count); the
bounds are at most 3 tasks x 4 stages x 8 ways.  Among equally costly
paths the lexicographically smallest way-count vector is returned
(tasks are processed sorted by task id, matching the existing planner
convention).  Tasks never share a budget: the per-task plans are simply
reported together with their summed total cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import _require_int
from .errors import InfeasibleStageError, ValidationError

# Interface bounds fixed by this iteration's contract.
MAX_TASKS = 3
MAX_STAGES = 4
MAX_TASK_WAYS = 8


# ---------------------------------------------------------------------------
# Validated input data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    """One stage of a task: minimum ways, loss ceiling, discrete loss curve.

    ``loss_curve`` has length ``num_ways`` (validated at planning time);
    element ``[w - 1]`` is the non-negative integer loss when this stage
    runs with ``w`` ways.  ``max_loss`` is the minimum-performance
    requirement in loss units (a way count is feasible only when its loss
    does not exceed it).
    """

    min_ways: int
    max_loss: int
    loss_curve: tuple[int, ...]

    def __init__(self, min_ways, max_loss, loss_curve):
        # Bounds against num_ways are applied in the planner, which knows
        # num_ways; here we only check the types and basic ranges that are
        # meaningful on their own, so a Stage is never silently malformed.
        min_ways = _require_int(min_ways, "stage.min_ways", 1)
        max_loss = _require_int(max_loss, "stage.max_loss")
        if isinstance(loss_curve, (str, bytes)) or not hasattr(
            loss_curve, "__iter__"
        ):
            raise ValidationError(
                f"stage.loss_curve: expected an iterable of loss values, "
                f"got {type(loss_curve).__name__}"
            )
        curve = tuple(
            _require_int(v, f"stage.loss_curve[{i}]", 0)
            for i, v in enumerate(loss_curve)
        )
        if not curve:
            raise ValidationError(
                "stage.loss_curve: must contain at least one loss value"
            )
        object.__setattr__(self, "min_ways", min_ways)
        object.__setattr__(self, "max_loss", max_loss)
        object.__setattr__(self, "loss_curve", curve)


@dataclass(frozen=True)
class Task:
    """One task: an id, its stage sequence, and optional migration overrides.

    ``migration_cost`` / ``max_change`` default to ``None``, meaning the
    planner-level values apply to this task.
    """

    task_id: str
    stages: tuple[Stage, ...]
    migration_cost: int | None = None
    max_change: int | None = None

    def __init__(self, task_id, stages, migration_cost=None, max_change=None):
        if not isinstance(task_id, str) or not task_id:
            raise ValidationError(
                f"task_id: expected a non-empty str, got {task_id!r}"
            )
        if isinstance(stages, (str, bytes)) or not hasattr(stages, "__iter__"):
            raise ValidationError(
                f"stages of task {task_id!r}: expected an iterable of Stage, "
                f"got {type(stages).__name__}"
            )
        out_stages = []
        for i, s in enumerate(stages):
            if not isinstance(s, Stage):
                # Convenience: accept (min_ways, max_loss, loss_curve) triples.
                try:
                    min_ways, max_loss, curve = s
                except (TypeError, ValueError):
                    raise ValidationError(
                        f"stages[{i}] of task {task_id!r}: expected Stage or "
                        f"(min_ways, max_loss, loss_curve) triple, got {s!r}"
                    ) from None
                s = Stage(min_ways, max_loss, curve)
            out_stages.append(s)
        if not 1 <= len(out_stages) <= MAX_STAGES:
            raise ValidationError(
                f"stages of task {task_id!r}: expected 1..{MAX_STAGES} "
                f"stage(s), got {len(out_stages)}"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "stages", tuple(out_stages))
        if migration_cost is not None:
            migration_cost = _require_int(
                migration_cost, f"migration_cost of task {task_id!r}", 0
            )
        object.__setattr__(self, "migration_cost", migration_cost)
        if max_change is not None:
            max_change = _require_int(
                max_change, f"max_change of task {task_id!r}", 1
            )
        object.__setattr__(self, "max_change", max_change)


def _normalize_tasks(tasks):
    if isinstance(tasks, (str, bytes)) or not hasattr(tasks, "__iter__"):
        raise ValidationError(
            f"tasks: expected an iterable of Task, "
            f"got {type(tasks).__name__}"
        )
    out = []
    for i, t in enumerate(tasks):
        if not isinstance(t, Task):
            # Convenience: accept (task_id, stages) pairs, validated the
            # same way (no per-task migration overrides in this short form).
            try:
                task_id, stages = t
            except (TypeError, ValueError):
                raise ValidationError(
                    f"tasks[{i}]: expected Task or (task_id, stages) pair, "
                    f"got {t!r}"
                ) from None
            t = Task(task_id, stages)
        out.append(t)
    if not 1 <= len(out) <= MAX_TASKS:
        raise ValidationError(
            f"tasks: expected 1..{MAX_TASKS} task(s), got {len(out)}"
        )
    ids = [t.task_id for t in out]
    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise ValidationError(f"tasks: duplicate task_id(s): {dupes}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskPlan:
    """Optimal plan for one task (immutable)."""

    task_id: str
    ways: tuple[int, ...]                 # chosen way count per stage
    stage_losses: tuple[int, ...]         # loss_t(ways[t]) per stage
    migration_costs: tuple[int, ...]      # cost paid entering stage t; [0] == 0
    migration_cost: int                   # fixed switching cost applied per change
    max_change: int                       # max |delta ways| between stages
    total_loss: int                       # sum of stage losses
    total_migration_cost: int             # sum of migration_costs
    total_cost: int                       # total_loss + total_migration_cost

    def stages(self):
        """Per-stage ``(ways, loss, migration_paid)`` rows, stage order."""
        return tuple(
            (w, lo, mi)
            for w, lo, mi in zip(self.ways, self.stage_losses,
                                 self.migration_costs)
        )


@dataclass(frozen=True)
class MultiStagePlanResult:
    """Result of :func:`plan_task_stages` (immutable)."""

    num_ways: int
    default_migration_cost: int
    default_max_change: int
    task_ids: tuple[str, ...]             # sorted by task id
    plans: tuple[TaskPlan, ...]           # aligned with task_ids
    total_cost: int                       # sum of per-task total costs

    def plan_map(self):
        """Task id -> :class:`TaskPlan`, insertion ordered by task id."""
        return dict(zip(self.task_ids, self.plans))


# ---------------------------------------------------------------------------
# Exact dynamic program
# ---------------------------------------------------------------------------

def _plan_one_task(task, num_ways, migration_cost, max_change):
    """Solve one task exactly by DP over (stage, way count).

    Returns a :class:`TaskPlan`.  Raises :class:`InfeasibleStageError`
    when the constraints are provably unsatisfiable.
    """
    t = len(task.stages)

    # Feasible way counts per stage: w >= min_ways and loss <= max_loss.
    feasible = []
    for ti, stage in enumerate(task.stages):
        fs = set()
        for w in range(stage.min_ways, num_ways + 1):
            if stage.loss_curve[w - 1] <= stage.max_loss:
                fs.add(w)
        if not fs:
            raise InfeasibleStageError(
                f"stage {ti} of task {task.task_id!r}: no way count in "
                f"{stage.min_ways}..{num_ways} meets the minimum-performance "
                f"requirement (loss <= {stage.max_loss}); the stage's loss "
                f"curve is {stage.loss_curve[stage.min_ways - 1:]}"
            )
        feasible.append(fs)

    # dp[w] at the current frontier = (cost_so_far, path_tuple).  Tuple
    # comparison orders by cost first and lexicographic path second, which
    # is exactly the documented tie-break.
    dp = {}
    for w in feasible[0]:
        dp[w] = (task.stages[0].loss_curve[w - 1], (w,))
    for ti in range(1, t):
        nxt = {}
        for w in feasible[ti]:
            best = None
            for prev, (prev_cost, prev_path) in dp.items():
                delta = abs(w - prev)
                if delta > max_change:
                    continue
                switch = migration_cost if w != prev else 0
                cand = (
                    prev_cost + switch + task.stages[ti].loss_curve[w - 1],
                    prev_path + (w,),
                )
                if best is None or cand < best:
                    best = cand
            if best is not None:
                nxt[w] = best
        if not nxt:
            raise InfeasibleStageError(
                f"stage {ti} of task {task.task_id!r}: no feasible way count "
                f"is reachable from stage {ti - 1} within max_change="
                f"{max_change}; feasible counts are "
                f"{sorted(feasible[ti - 1])} -> {sorted(feasible[ti])}"
            )
        dp = nxt

    total_cost, path = min(dp.values())
    stage_losses = tuple(
        task.stages[ti].loss_curve[path[ti] - 1] for ti in range(t)
    )
    migration_costs = tuple(
        0 if ti == 0 or path[ti] == path[ti - 1] else migration_cost
        for ti in range(t)
    )
    total_migration = sum(migration_costs)
    total_loss = sum(stage_losses)
    return TaskPlan(
        task_id=task.task_id,
        ways=path,
        stage_losses=stage_losses,
        migration_costs=migration_costs,
        migration_cost=migration_cost,
        max_change=max_change,
        total_loss=total_loss,
        total_migration_cost=total_migration,
        total_cost=total_loss + total_migration,
    )


def plan_task_stages(tasks, num_ways, migration_cost=0, max_change=None):
    """Plan way counts across a known stage sequence for up to 3 tasks.

    Parameters
    ----------
    tasks          : iterable of :class:`Task` (or ``(task_id, stages)``
                     pairs), 1..3 tasks with unique non-empty string ids.
                     A task may carry its own ``migration_cost`` /
                     ``max_change`` overrides.
    num_ways       : 1..8; every stage's loss curve must have exactly this
                     many entries (one per way count 1..num_ways).
    migration_cost : non-negative int, fixed switching cost charged at each
                     stage boundary where the way count changes (default 0).
    max_change     : 1..num_ways, maximum ``|delta ways|`` between
                     consecutive stages (default ``num_ways``).

    Returns a :class:`MultiStagePlanResult`; tasks are reported sorted by
    task id.  Raises :class:`ValidationError` for malformed input and
    :class:`InfeasibleStageError` for valid input whose per-stage
    performance / reachability constraints are provably unsatisfiable.
    """
    tasks_in = _normalize_tasks(tasks)
    num_ways = _require_int(num_ways, "num_ways", 1, MAX_TASK_WAYS)
    migration_cost = _require_int(
        migration_cost, "migration_cost", 0
    )
    if max_change is None:
        max_change = num_ways
    max_change = _require_int(max_change, "max_change", 1, num_ways)

    # Cross-field validation: curve lengths and per-task overrides are
    # checked for every task before any DP runs, so a failure rejects the
    # whole call up front (the library is stateless regardless, but this
    # keeps "validate, then compute" explicit).
    tasks_sorted = sorted(tasks_in, key=lambda x: x.task_id)
    for task in tasks_sorted:
        for ti, stage in enumerate(task.stages):
            if len(stage.loss_curve) != num_ways:
                raise ValidationError(
                    f"loss_curve of stage {ti} of task {task.task_id!r}: "
                    f"expected {num_ways} value(s) (one per way count "
                    f"1..{num_ways}), got {len(stage.loss_curve)}"
                )
            if stage.min_ways > num_ways:
                raise ValidationError(
                    f"min_ways of stage {ti} of task {task.task_id!r}: "
                    f"{stage.min_ways} exceeds num_ways {num_ways}"
                )
        if task.max_change is not None and task.max_change > num_ways:
            raise ValidationError(
                f"max_change of task {task.task_id!r}: {task.max_change} "
                f"exceeds num_ways {num_ways}"
            )

    plans = []
    for task in tasks_sorted:
        c = task.migration_cost if task.migration_cost is not None else migration_cost
        d = task.max_change if task.max_change is not None else max_change
        plans.append(_plan_one_task(task, num_ways, c, d))

    return MultiStagePlanResult(
        num_ways=num_ways,
        default_migration_cost=migration_cost,
        default_max_change=max_change,
        task_ids=tuple(t.task_id for t in tasks_sorted),
        plans=tuple(plans),
        total_cost=sum(p.total_cost for p in plans),
    )
