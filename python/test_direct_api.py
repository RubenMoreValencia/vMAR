"""Tests for pip-installed `from vmar import Differential`."""

import math
import unittest

from vmar import Differential, VmarError


class PythonAPITest(unittest.TestCase):
    def test_exact_reference_and_r_parity(self):
        model = Differential(lambda t, y: (-y[0],), initial_state=[1.0])
        euler = model.integrate(method="euler", step=0.1, steps=10)
        rk4 = model.integrate(method="rk4", step=0.1, steps=10)
        self.assertEqual(len(rk4.states), 11)
        self.assertAlmostEqual(euler.state[0], 0.3486784401, places=13)
        self.assertAlmostEqual(rk4.state[0], 0.3678797744124984, places=13)
        self.assertLess(abs(rk4.state[0] - math.exp(-1)), abs(euler.state[0] - math.exp(-1)))

    def test_vector_and_field_calls(self):
        calls = []

        def field(t, values):
            calls.append(t)
            return -values[0], -2 * values[1]

        result = Differential(field, [1, 1]).integrate(method="rk4", step=0.1, steps=2)
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(result.state), 2)
        self.assertEqual(result.times, (0.0, 0.1, 0.2))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Differential(lambda t, y: y, [float("nan")])
        with self.assertRaises(ValueError):
            Differential(lambda t, y: y, [1]).integrate(step=-0.1)
        with self.assertRaises(ValueError):
            Differential(lambda t, y: y, [1]).integrate(method="adam")
        with self.assertRaises(VmarError):
            Differential(lambda t, y: (), [1]).integrate()

    def test_zero_steps(self):
        result = Differential(lambda t, y: (-y[0],), [2]).integrate(steps=0)
        self.assertEqual(result.state, (2.0,))
        self.assertEqual(result.times, (0.0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
