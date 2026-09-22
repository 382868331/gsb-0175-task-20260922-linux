"""Error types for the wayplan package.

Error semantics (fixed public contract):

- ``ValidationError``  — any contract violation in caller-supplied input
  (wrong type, out-of-range value, NaN/Infinity, bool passed as an integer
  parameter, duplicate app ids, mismatched lengths, ...).  The message
  always names the offending parameter so the failure is locatable.
- ``InfeasibleBudgetError`` — the input is *valid*, but the way budget is
  provably too small to give every application its mandatory minimum of
  one way.  This is a proved-infeasible result, not a search timeout:
  the planner uses an exact, complete dynamic program, so "search budget
  exhausted" cannot occur and is therefore represented by no exception
  at all.  The two situations are never conflated.

Both derive from ``WayPlanError`` so callers can catch one base class.
``ValidationError`` also derives from ``ValueError`` for idiomatic use.
"""


class WayPlanError(Exception):
    """Base class for all errors raised by wayplan."""


class ValidationError(WayPlanError, ValueError):
    """Caller-supplied input violates the documented contract."""


class InfeasibleBudgetError(WayPlanError):
    """Valid input, but the way budget cannot cover one way per app."""
