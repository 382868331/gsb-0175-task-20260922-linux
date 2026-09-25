"""Contract validation tests: every rejection is locatable and explicit."""

import math
import unittest

from wayplan import (
    MAX_APPS,
    MAX_SETS,
    MAX_WAYS,
    AppTrace,
    InfeasibleBudgetError,
    ValidationError,
    WayPlanError,
    miss_curve,
    plan_allocation,
)

GOOD_TRACES = [("a", [0, 1, 0]), ("b", [2, 3])]
GOOD_WEIGHTS = [1, 2]


def plan(**overrides):
    args = dict(
        traces=GOOD_TRACES, weights=GOOD_WEIGHTS,
        num_sets=2, total_ways=4, budget=3,
    )
    args.update(overrides)
    return plan_allocation(
        args["traces"], args["weights"], args["num_sets"],
        args["total_ways"], args["budget"],
    )


class NumericContract(unittest.TestCase):
    def test_nan_and_infinity_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValidationError, msg=f"weight {bad}"):
                plan(weights=[1, bad])
            with self.assertRaises(ValidationError, msg=f"budget {bad}"):
                plan(budget=bad)
            with self.assertRaises(ValidationError, msg=f"num_sets {bad}"):
                plan(num_sets=bad)
            with self.assertRaises(ValidationError, msg=f"block {bad}"):
                plan(traces=[("a", [0, bad]), ("b", [1])])

    def test_non_integral_float_rejected_not_coerced(self):
        with self.assertRaises(ValidationError):
            plan(weights=[1, 2.5])
        with self.assertRaises(ValidationError):
            plan(budget=2.0)  # even integral floats are not silently accepted

    def test_bool_rejected_for_integer_parameters(self):
        with self.assertRaises(ValidationError):
            plan(num_sets=True)
        with self.assertRaises(ValidationError):
            plan(total_ways=False)
        with self.assertRaises(ValidationError):
            plan(budget=True)
        with self.assertRaises(ValidationError):
            plan(weights=[1, True])
        with self.assertRaises(ValidationError):
            plan(traces=[("a", [0, True]), ("b", [1])])

    def test_wrong_types_rejected(self):
        with self.assertRaises(ValidationError):
            plan(num_sets="4")
        with self.assertRaises(ValidationError):
            plan(weights="12")
        with self.assertRaises(ValidationError):
            plan(traces="ab")
        with self.assertRaises(ValidationError):
            plan(traces=[("a", [0]), ("b", None)])


class RangeContract(unittest.TestCase):
    def test_interface_bounds(self):
        with self.assertRaises(ValidationError):
            plan(num_sets=0)
        with self.assertRaises(ValidationError):
            plan(num_sets=MAX_SETS + 1)
        with self.assertRaises(ValidationError):
            plan(total_ways=0)
        with self.assertRaises(ValidationError):
            plan(total_ways=MAX_WAYS + 1)
        with self.assertRaises(ValidationError):
            plan(budget=-1)
        # budget 0 is valid input since v1.1 (feasible only when every app
        # is disableable); with the default all-mandatory apps it is
        # proved infeasible, which is NOT a validation error
        with self.assertRaises(InfeasibleBudgetError):
            plan(budget=0)
        # budget may not exceed total_ways (cannot hand out ways that
        # do not exist in the cache being partitioned)
        with self.assertRaises(ValidationError):
            plan(budget=5, total_ways=4)

    def test_app_count_bounds(self):
        one = [("a", [0])]
        with self.assertRaises(ValidationError):
            plan(traces=[], weights=[])
        too_many = [(f"app{i}", [i]) for i in range(MAX_APPS + 1)]
        with self.assertRaises(ValidationError):
            plan(traces=too_many, weights=[1] * (MAX_APPS + 1))
        # boundary values themselves are accepted
        plan(traces=one, weights=[1], budget=1, total_ways=1, num_sets=1)
        many = [(f"app{i}", [i]) for i in range(MAX_APPS)]
        plan(traces=many, weights=[1] * MAX_APPS, budget=MAX_APPS,
             total_ways=MAX_WAYS, num_sets=MAX_SETS)

    def test_weights_contract(self):
        with self.assertRaises(ValidationError):  # wrong length
            plan(weights=[1])
        with self.assertRaises(ValidationError):  # zero
            plan(weights=[1, 0])
        with self.assertRaises(ValidationError):  # negative
            plan(weights=[-3, 1])

    def test_trace_contract(self):
        with self.assertRaises(ValidationError):  # duplicate ids
            plan(traces=[("a", [0]), ("a", [1])])
        with self.assertRaises(ValidationError):  # empty id
            plan(traces=[("", [0]), ("b", [1])])
        with self.assertRaises(ValidationError):  # non-str id
            plan(traces=[(1, [0]), ("b", [1])])
        with self.assertRaises(ValidationError):  # empty trace
            plan(traces=[("a", []), ("b", [1])])
        with self.assertRaises(ValidationError):  # negative block
            plan(traces=[("a", [0, -1]), ("b", [1])])
        with self.assertRaises(ValidationError):  # malformed entry
            plan(traces=[("a", [0]), 42])


class ErrorSemantics(unittest.TestCase):
    def test_error_hierarchy(self):
        self.assertTrue(issubclass(ValidationError, WayPlanError))
        self.assertTrue(issubclass(ValidationError, ValueError))
        self.assertTrue(issubclass(InfeasibleBudgetError, WayPlanError))
        self.assertFalse(issubclass(InfeasibleBudgetError, ValidationError))

    def test_messages_are_locatable(self):
        cases = [
            (dict(weights=[1, -1]), "weights[1]"),
            (dict(num_sets=99), "num_sets"),
            (dict(total_ways=0), "total_ways"),
            (dict(budget=-1), "budget"),
            (dict(traces=[("a", [0, -5]), ("b", [1])]), "blocks"),
        ]
        for kwargs, needle in cases:
            with self.assertRaises(ValidationError) as ctx:
                plan(**kwargs)
            self.assertIn(needle, str(ctx.exception))

    def test_failed_call_leaves_no_partial_state(self):
        # The library keeps no mutable module-level state: a rejected call
        # must not affect subsequent valid calls.
        with self.assertRaises(ValidationError):
            plan(weights=[1, math.nan])
        r = plan()
        self.assertEqual(sum(r.allocation), r.used_ways)
        self.assertLessEqual(r.used_ways, 3)

    def test_apptrace_validates_eagerly(self):
        with self.assertRaises(ValidationError):
            AppTrace("a", [0, -1])
        t = AppTrace("a", iter([1, 2, 3]))  # any iterable is accepted once
        self.assertEqual(t.blocks, (1, 2, 3))

    def test_miss_curve_bounds(self):
        with self.assertRaises(ValidationError):
            miss_curve([1], 1, 0)
        with self.assertRaises(ValidationError):
            miss_curve([1], 1, MAX_WAYS + 1)


if __name__ == "__main__":
    unittest.main()
