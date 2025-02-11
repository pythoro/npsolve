# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:48:33 2019

@author: Reuben

This example for Tutorial 1 introduces the basics of npsolve.

"""

import numpy as np
import npsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var("position1", init=0.1)
        self.add_var("velocity1", init=0.3)

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step

        Calculate acceleration based on the net component2_value.
        """
        acceleration = 1.0 * self.state["component2_value"]
        derivatives = {
            "position1": self.state["velocity1"],
            "velocity1": acceleration,
        }
        return derivatives


class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var("component2_value", init=-0.1)

    def calculate(self, t):
        """Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        """
        dc = 1.0 * np.cos(2 * t) * self.state["position1"]
        derivatives = {"component2_value": dc}
        return derivatives

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step"""
        return self.calculate(t)


class Solver(npsolve.Solver):
    def solve(self, t_end=10):
        self.npsolve_init()  # Initialise
        self.t_vec = np.linspace(0, t_end, 1001)
        result = odeint(self.step, self.npsolve_initial_values, self.t_vec)
        return result


def run():
    solver = Solver()
    partials = [Component1(), Component2()]
    solver.connect_partials(partials)
    res = solver.solve()
    return res, solver


def plot(res, solver):
    s = solver
    slices = s.npsolve_slices
    plt.figure(2)
    plt.plot(s.t_vec, res[:, slices["position1"]], label="position1")
    plt.plot(s.t_vec, res[:, slices["velocity1"]], label="velocity1")
    plt.plot(
        s.t_vec, res[:, slices["component2_value"]], label="component2_value"
    )
    plt.legend()
    plt.show()


def execute():
    res, solver = run()
    plot(res, solver)

if __name__ == '__main__':
    execute()