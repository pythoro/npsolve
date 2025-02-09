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


class Component1():

    def set_comp2_acc(self, acc):
        self._comp2_acc = acc

    def get_pos(self, state_dct):
        return state_dct['position1']

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step

        Calculate acceleration based on the net component2_value.
        """
        acceleration = self._comp2_acc
        derivatives = {
            "position1": state_dct["velocity1"],
            "velocity1": acceleration,
        }
        return derivatives

class Component2:
    def get_acceleration(self, state_dct):
        return 1.0 * state_dct["component2_value"]

    def set_comp1_pos(self, pos):
        self._comp1_pos = pos

    def calculate(self, state_dct, t):
        """Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        """
        dc = 1.0 * np.cos(2 * t) * self._comp1_pos
        derivatives = {"component2_value": dc}
        return derivatives

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step"""
        return self.calculate(state_dct, t)

class Assembly:
    def __init__(self, comp1, comp2):
        self.comp1 = comp1
        self.comp2 = comp2

    def precalcs(self, state_dct, t):
        comp1 = self.comp1
        comp2 = self.comp2
        comp1_pos = comp1.get_pos(state_dct)
        comp2_acc = comp2.get_acceleration(state_dct)
        comp1.set_comp2_acc(comp2_acc)
        comp2.set_comp1_pos(comp1_pos)


def get_package():
    comp1 = Component1()
    npsolve_component1 = npsolve.Component('comp1', comp1)
    npsolve_component1.add_var("position1", init=0.1)
    npsolve_component1.add_var("velocity1", init=0.3)
    comp2 = Component2()
    npsolve_component2 = npsolve.Component('comp2', comp2)
    npsolve_component2.add_var("component2_value", init=-0.1)
    assembly = Assembly(comp1, comp2)
    npsolve_assembly = npsolve.Component('assembly', assembly)
    package = npsolve.Package()
    package.add_component(npsolve_component1, 'step')
    package.add_component(npsolve_component2, 'step')
    package.add_component(npsolve_assembly, None)
    package.add_stage_call('assembly', 'precalcs')
    package.setup()
    return package


def solve(package, t_end=10):
    t_vec = np.linspace(0, t_end, 1001)
    result = odeint(package.step, package.init_vec, t_vec)
    return t_vec, result


def run():
    package = get_package()
    t_vec, result = solve(package)
    return package, t_vec, result


def plot(package, t_vec, result):
    slices = package.slices
    plt.figure()
    plt.plot(t_vec, result[:, slices["position1"]], label="position1")
    plt.plot(t_vec, result[:, slices["velocity1"]], label="velocity1")
    plt.plot(
        t_vec, result[:, slices["component2_value"]], label="component2_value"
    )
    plt.legend()
    plt.show()


def execute():
    package, t_vec, result = run()
    plot(package, t_vec, result)

execute()