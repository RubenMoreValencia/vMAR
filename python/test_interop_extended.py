"""Cross-boundary contract checks for the new native-backed Python API."""
import math
import unittest

from vmar import (Differential, Network, ControlledNeuron, NetworkMechanism,
                  ConfirmedTrace, StateLayout, activation, memory_ema,
                  memory_relaxation)


class InteropPublicContractTests(unittest.TestCase):
    def test_original_integrator_regression(self):
        r = Differential(lambda t, y: (-y[0],), (1.,)).integrate(steps=10)
        self.assertAlmostEqual(r.state[0], math.exp(-1), places=6)

    def test_activation_and_kink_convention(self):
        self.assertEqual(activation("relu", 0.), (0., 0.))
        self.assertAlmostEqual(activation("sigmoid", 0.)[0], .5)

    def test_neuronal_memory_semantics(self):
        self.assertEqual(memory_ema((0., 1.), (2., 3.), .5), (1., 2.))
        self.assertAlmostEqual(memory_relaxation((0.,), (1.,), .1, .5)[0],
                               1-math.exp(-.2), places=12)

    def test_network_gradient_and_native_rk4(self):
        n = Network((1, 1), ("identity",), (0., 0.))
        loss, gradient, prediction = n.loss_gradient([[0.]], [1.])
        self.assertEqual((loss, prediction), (1., (0.,)))
        self.assertEqual(gradient, (0., -2.))
        n.step([[0.]], [1.], method="rk4", step=.1)
        self.assertAlmostEqual(n.parameters[-1], 1-math.exp(-.2), delta=3e-6)

    def test_state_layout_and_second_moment_invariant(self):
        layout = StateLayout.make(2, "second_moment")
        self.assertEqual(layout.total_doubles, 6)
        self.assertTrue(layout.validate((0.,)*6))
        with self.assertRaises(Exception):
            layout.validate((0., 0., 0., 0., -.1, 0.))

    def test_controlled_neuron_trace(self):
        n = ControlledNeuron([[1.]], [1.], [0., 0.])
        before = n.state
        report = n.step()
        self.assertTrue(report["accepted"])
        self.assertNotEqual(before, n.state)
        self.assertEqual(report["field_evaluations"], 4)

    def test_four_network_modes_default_observational(self):
        outcomes = {}
        for mode in ("baseline", "observe", "coupled", "controlled"):
            n = Network((1, 1), ("identity",))
            process = NetworkMechanism(n, [[1.]], [1.], mode=mode,
                                       coupling=.1)
            result = process.step()
            self.assertTrue(result["accepted"])
            outcomes[mode] = result
        self.assertEqual(outcomes["baseline"]["state"][-2:], (0., 0.))
        self.assertNotEqual(outcomes["observe"]["state"][-2:], (0., 0.))
        self.assertAlmostEqual(outcomes["coupled"]["loss_after"],
                               outcomes["controlled"]["loss_after"])

    def test_confirmed_trace_rejects_bad_iteration(self):
        trace = ConfirmedTrace(2, 3)
        trace.append(0, 0., 1., 0., (0., 1.))
        with self.assertRaises(Exception):
            trace.append(2, .1, .8, .1, (1., 1.))
        self.assertEqual(len(trace.records), 1)
        trace.append(1, .1, .8, .1, (1., 1.))
        self.assertEqual(trace.records[-1].state, (1., 1.))


if __name__ == "__main__":
    unittest.main()
