"""First independent vMAR Python consumer; the mathematics executes in C23."""

from vmar import (Differential, Network, NetworkMechanism, ControlledNeuron,
                  ConfirmedTrace, StateLayout, activation, memory_ema)


def main():
    print("activation:", activation("tanh", .5))
    print("memory:", memory_ema((0., 0.), (1., 2.), .8))
    trajectory = Differential(lambda t, y: (-y[0],), [1.]).integrate(steps=10)
    print("differential:", trajectory.state)

    net = Network((2, 4, 1))
    observations = [[0., 1.], [1., 0.]]
    targets = [0., 1.]
    print("initial_loss:", net.loss_gradient(observations, targets)[0])
    net.step(observations, targets, method="rk4", step=.1)
    print("updated_loss:", net.loss_gradient(observations, targets)[0])

    mechanism = NetworkMechanism(net, observations, targets, mode="observe")
    result = mechanism.step()
    print("observational_memory:", result["memory_norm"], result["state"])

    controlled = ControlledNeuron([[1.]], [1.], [0., 0.], coupling=.1)
    decision = controlled.step()
    print("control:", decision["accepted"], decision["effective_step"])

    layout = StateLayout.make(parameters=2, kind="first_memory")
    print("state_layout:", layout)
    trace = ConfirmedTrace(dimension=2, capacity=2)
    trace.append(0, 0., 1., 0., (0., 0.))
    trace.append(1, .1, .8, .1, (.1, .2))
    print("trace:", trace.records)


if __name__ == "__main__":
    main()
