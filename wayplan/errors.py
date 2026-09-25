"""Error types for the wayplan package.

Error semantics (fixed public contract):

- ``ValidationError``  — any contract violation in caller-supplied input
  (wrong type, out-of-range value, NaN/Infinity, bool passed as an integer
  parameter, duplicate app ids, mismatched lengths, ...).  The message
  always names the offending parameter so the failure is locatable.
- ``InfeasibleBudgetError`` — the input is *valid*, but no allocation
  exists that satisfies every hard constraint within the way budget:
  either the budget cannot cover the mandatory minimum of one way per
  non-deactivatable application, or some application's hard miss cap
  cannot be met by any way count in 1..total_ways (and the application
  may not be deactivated), or the budget cannot cover the per-application
  minima imposed by the caps.  This is a proved-infeasible result, not a
  search timeout: the planner uses an exact, complete dynamic program, so
  "search budget exhausted" cannot occur and is therefore represented by
  no exception at all.  The two situations are never conflated.

  The exception carries a ``diagnostics`` attribute (``None`` when not
  computed): a mapping with the total budget and, per application, the
  independent minimum way count needed to satisfy its own cap (``None``
  when the cap is unachievable with 1..total_ways ways) plus whether the
  application may be deactivated.  These per-application minima are only
  a *necessary* condition — they are diagnostic information, not a proof
  procedure for general infeasibility; feasibility itself is decided by
  the exact, complete planner.

Both derive from ``WayPlanError`` so callers can catch one base class.
``ValidationError`` also derives from ``ValueError`` for idiomatic use.
"""


class WayPlanError(Exception):
    """Base class for all errors raised by wayplan."""


class ValidationError(WayPlanError, ValueError):
    """Caller-supplied input violates the documented contract."""


class InfeasibleBudgetError(WayPlanError):
    """Valid input, but no allocation satisfies all hard constraints.

    ``diagnostics`` is ``None`` or a mapping::

        {
            "budget": int,
            "total_ways": int,
            "per_app": (
                {
                    "app_id": str,
                    "min_ways_if_active": int | None,  # None = cap unachievable
                    "deactivatable": bool,
                    "penalty": int | None,
                },
                ...  # one entry per application, ordered by app id
            ),
        }

    The per-application minima are independent lower bounds (a necessary
    condition only); they are reported to help locate the bottleneck, not
    as a general feasibility proof.
    """

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics
