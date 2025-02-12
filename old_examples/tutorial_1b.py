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

from tutorial_1 import Solver, plot

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var("position1", init=0.1)
        self.add_var("velocity1", init=0.3)

    def get_position(self):
        """Returns a value
        
        In this example, it is just a state variable, but it could be much
        more complex.
        """
        return self.state['position1']

    def connect(self, component2):
        """Connect with a Component2 instance"""
        self._component2 = component2

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step

        Calculate acceleration based on the net component2_value.
        """
        acceleration = 1.0 * self._component2.get_value()
        derivatives = {
            "position1": self.state["velocity1"],
            "velocity1": acceleration,
        }
        return derivatives


class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var("component2_value", init=-0.1)

    def get_value(self):
        """Returns a value
        
        In this example, it is just a state variable, but it could be much
        more complex.
        """
        return self.state['component2_value']

    def connect(self, component1):
        """Connect with a Component1 instance"""
        self._component1 = component1

    def calculate(self, t):
        """Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        """
        dc = 1.0 * np.cos(2 * t) * self._component1.get_position()
        derivatives = {"component2_value": dc}
        return derivatives

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step"""
        return self.calculate(t)


def run():
    solver = Solver()
    component1 = Component1()
    component2 = Component2()
    component1.connect(component2)  # Inject the dependency
    component2.connect(component1)  # Inject the dependency
    partials = [component1, component2]
    solver.connect_partials(partials)
    res = solver.solve()
    return res, solver


def execute():
    res, solver = run()
    plot(res, solver)

if __name__ == '__main__':
    execute()