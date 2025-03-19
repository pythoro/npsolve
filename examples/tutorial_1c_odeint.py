"""Illustrates using scipy odeint solver instead of npsolve ODEIntegrator."""

from examples.tutorial_1 import get_system


import numpy as np
import npsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Unique variable names
COMP1_POS = "position1"
COMP1_VEL = "velocity1"
COMP2_VALUE = "component2_value"
COMP2_FORCE = "comp2_force"


def solve(system, t_end=10):
    t_vec = np.linspace(0, t_end, 1001)
    result = odeint(system.step, system.init_vec, t_vec)
    return t_vec, result


def run():
    system = get_system()
    inits = {COMP1_POS: 0.1, COMP1_VEL: 0.3, COMP2_VALUE: -0.1}
    system.setup(inits)
    t_vec, result = solve(system)
    return system, t_vec, result


def plot(system, t_vec, result):
    slices = system.slices
    plt.figure(1)
    for slice_name, slice in slices.items():
        plt.plot(t_vec, result[:, slice], label=slice_name)
    plt.legend()
    plt.show()


def execute():
    system, t_vec, result = run()
    plot(system, t_vec, result)


if __name__ == "__main__":
    execute()
