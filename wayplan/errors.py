"""Error types for the wayplan package.

Error semantics (fixed public contract):

- ``ValidationError``  — any contract violation in caller-supplied input
  (wrong type, out-of-range value, NaN/Infinity, bool passed as an integer
  parameter, duplicate app ids, mismatched lengths, ...).  The message
  always names the offending parameter so the failure is locatable.
- ``InfeasibleBudgetError`` — the input is *valid*, but no way allocation
  can satisfy every hard constraint at once (the way budget, the mandatory
  minimum of one way per non-disableable application, and every declared
  per-application miss cap).  This is a proved-infeasible result, not a
  search timeout: the planner uses an exact, complete dynamic program, so
  "search budget exhausted" cannot occur and is therefore represented by
  no exception at all.  The two situations are never conflated.

  The exception carries a diagnostic snapshot (``app_ids``, ``min_ways``,
  ``required_total``, ``budget``): each application's *independent*
  minimum way requirement (0 when the app may be disabled, ``None`` when
  its miss cap is unreachable even with ``total_ways`` ways) and the total
  budget.  This is only a necessary-condition summary to help locate the
  cause — it is not, by itself, a general infeasibility proof; the exact
  planner is the authority on infeasibility.

Both derive from ``WayPlanError`` so callers can catch one base class.
``ValidationError`` also derives from ``ValueError`` for idiomatic use.
"""


class WayPlanError(Exception):
    """Base class for all errors raised by wayplan."""


class ValidationError(WayPlanError, ValueError):
    """Caller-supplied input violates the documented contract."""


class InfeasibleBudgetError(WayPlanError):
    """Valid input, but no allocation satisfies every hard constraint.

    Diagnostic attributes (all ``None`` when raised without diagnostics):

    - ``app_ids``        — application ids, sorted, aligned with ``min_ways``
    - ``min_ways``       — per-app independent minimum way requirement:
      0 for disableable apps, the smallest way count meeting the miss cap
      otherwise, or ``None`` when the cap is unreachable at any way count
    - ``required_total`` — sum of ``min_ways`` (``None`` when any app is
      unachievable); the minimum budget any feasible plan would need
    - ``budget``         — the budget that was requested
    """

    def __init__(self, message, *, app_ids=None, min_ways=None,
                 required_total=None, budget=None):
        super().__init__(message)
        self.app_ids = app_ids
        self.min_ways = min_ways
        self.required_total = required_total
        self.budget = budget
