# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:20:43 2023

@author: Reuben

A one-module demo showing the use of dependency injection to communicate
between classes.

This used to be the preferred way to communicate between Partial objects.

"""

import npsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

from npsolve.solvers import FINAL, STOP

logger = npsolve.get_list_container("demo_logger")
status = npsolve.get_dict("demo_status")


class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var("position1", init=0.1)
        self.add_var("velocity1", init=0.3)

    def get_force(self):
        F = (
            -1 * (self.state["position1"] - 1.0)
            - 0.1 * self.state["velocity1"]
        )
        return F

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step

        Calculate acceleration based on the net force.
        """
        force = self.get_force()
        mass = 5
        acceleration = force / mass
        if status[FINAL]:
            logger["c1_force"].append(force)
        derivatives = {
            "position1": self.state["velocity1"],
            "velocity1": acceleration,
        }
        return derivatives


class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var("position2", init=0.0)
        self.add_var("velocity2", init=0.0)

    def connect(self, other: Component1):
        self._connected = other

    def calculate(self, t):
        """Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'position2'
        """
        force = -self._connected.get_force()  # Negative for reaction force
        if status[FINAL]:
            logger["c2_force"].append(force)
        derivatives = {
            "position2": self.state["velocity2"],
            "velocity2": force / 10,
        }
        return derivatives

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step"""
        return self.calculate(t)


def run():
    solver = npsolve.solvers.Integrator(
        status=status, logger=logger, framerate=60.0, x_name="time"
    )
    comp1 = Component1()
    comp2 = Component2()
    comp2.connect(comp1)  # Inject dependency
    partials = [comp1, comp2]
    solver.connect(partials)
    dct = solver.run(end=10.0)
    return dct


def plot(dct):
    plt.plot(dct["time"], dct["position1"], label="position1")
    plt.plot(dct["time"], dct["velocity1"], label="velocity1")
    plt.plot(dct["time"], dct["position2"], label="position2")
    plt.plot(dct["time"], dct["c1_force"], label="c1_force")
    plt.plot(dct["time"], dct["c2_force"], label="c2_force")
    plt.legend()


def execute():
    dct = run()
    plot(dct)
