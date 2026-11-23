"""G0.9-P.1 independent analytical and Python/C23 error gates."""

import math
import os
import unittest

from vmar_ctypes import VmarDifferential, VmarError


class DifferentialInteropTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = VmarDifferential(os.environ["VMAR_FFI_LIBRARY"])

    def test_scalar_analytic_reference(self):
        field = lambda t, y: (-y[0],)
        euler = self.engine.step(field, (1.0,), step=0.1, method="euler")[0]
        rk4 = self.engine.step(field, (1.0,), step=0.1, method="rk4")[0]
        exact = math.exp(-0.1)
        self.assertAlmostEqual(euler, 0.9, places=14)
        self.assertAlmostEqual(rk4, 0.9048375, places=14)
        self.assertLess(abs(rk4 - exact), abs(euler - exact))

    def test_vector_and_repeated_time(self):
        def field(t, y):
            return (-y[0], -2 * y[1])

        for method, expected_calls in (("euler", 1), ("rk4", 4)):
            calls = []

            def tracked(t, y):
                calls.append(t)
                return field(t, y)

            value = self.engine.step(tracked, (1.0, 1.0), step=0.01, method=method)
            self.assertEqual(len(calls), expected_calls)
            self.assertEqual(len(value), 2)
            self.assertTrue(all(math.isfinite(x) for x in value))

    def test_invalid_input_and_callback(self):
        with self.assertRaises(ValueError):
            self.engine.step(lambda t, y: y, (1.0,), step=-0.1)
        with self.assertRaises(VmarError):
            self.engine.step(lambda t, y: (), (1.0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
