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

# Unique variable names
COMP1_POS = 'position1'
COMP1_VEL = 'velocity1'
COMP2_VALUE = 'component2_value'
COMP2_FORCE = 'comp2_force'

class Component1():
    def set_comp2_force(self, force):
        self._comp2_force = force

    def get_pos(self, state):
        return state[COMP1_POS]

    def step(self, state, t, log):
        """Called by the solver at each time step

        Calculate acceleration based on the net component2_value.
        """
        acceleration = self._comp2_force * 1.0
        derivatives = {
            "position1": state[COMP1_VEL],
            "velocity1": acceleration,
        }
        return derivatives

class Component2:
    def get_force(self, state):
        return 1.0 * state[COMP2_VALUE]

    def set_comp1_pos(self, pos):
        self._comp1_pos = pos

    def calculate(self, state, t):
        """Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        """
        dc = 1.0 * np.cos(2 * t) * self._comp1_pos
        derivatives = {COMP2_VALUE: dc}
        return derivatives

    def step(self, state, t, log):
        """Called by the solver at each time step"""
        return self.calculate(state, t)


class Assembly:
    """Handle inter-dependencies."""
    def __init__(self, comp1, comp2):
        self.comp1 = comp1
        self.comp2 = comp2

    def precalcs(self, state, t, log):
        comp1 = self.comp1
        comp2 = self.comp2
        comp1_pos = comp1.get_pos(state)
        comp2_force = comp2.get_force(state)
        if log:
            log[COMP2_FORCE] = comp2_force
        comp1.set_comp2_force(comp2_force)
        comp2.set_comp1_pos(comp1_pos)


def get_package():
    component1 = Component1()
    component2 = Component2()
    assembly = Assembly(component1, component2)
    package = npsolve.Package()
    package.add_component(component1, 'comp1', 'step')
    package.add_component(component2, 'comp2', 'step')
    package.add_component(assembly, 'assembly', None)
    package.set_stage_calls(
        [('assembly', 'precalcs')]
    )
    return package


def solve(package, t_end=10):
    t_vec = np.linspace(0, t_end, 1001)
    result = odeint(package.step, package.init_vec, t_vec)
    return t_vec, result


def run():
    package = get_package()
    inits = {COMP1_POS: 0.1,
             COMP1_VEL: 0.3,
             COMP2_VALUE: -0.1}
    package.setup(inits)
    t_vec, result = solve(package)
    return package, t_vec, result


def plot(package, t_vec, result):
    slices = package.slices
    plt.figure(1)
    for slice_name, slice in slices.items():
        plt.plot(t_vec, result[:, slice], label=slice_name)
    plt.legend()
    plt.show()


def execute():
    package, t_vec, result = run()
    plot(package, t_vec, result)


execute()