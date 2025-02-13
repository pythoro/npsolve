"""Example of all the main functionality of npsolve.

Created on Mon Sep  2 16:48:33 2019

@author: Reuben

"""

import numpy as np
import npsolve
import matplotlib.pyplot as plt

# Unique variable names
COMP1_POS = "position1"
COMP1_VEL = "velocity1"
COMP2_VALUE = "component2_value"
COMP2_FORCE = "comp2_force"


class Component1:
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
        """Inject dependencies for later calculations in 'step' methods."""
        comp1 = self.comp1
        comp2 = self.comp2
        comp1_pos = comp1.get_pos(state)
        comp2_force = comp2.get_force(state)
        if log:
            # Log whatever we want here into a dictionary.
            log[COMP2_FORCE] = comp2_force
        comp1.set_comp2_force(comp2_force)
        comp2.set_comp1_pos(comp1_pos)


def get_system():
    component1 = Component1()
    component2 = Component2()
    assembly = Assembly(component1, component2)
    system = npsolve.System()
    system.add_component(component1, "comp1", "step")
    system.add_component(component2, "comp2", "step")
    system.add_component(assembly, "assembly", None)
    system.set_stage_calls([("assembly", "precalcs")])
    return system


def run():
    system = get_system()
    inits = {COMP1_POS: 0.1, COMP1_VEL: 0.3, COMP2_VALUE: -0.1}
    system.setup(inits)
    dct = npsolve.integrate(system, t_end=10.0, framerate=60.0)
    return dct


def plot(dct):
    plt.figure(1)
    dct2 = dct.copy()
    t_vec = dct2.pop("time")
    for var_name, values in dct2.items():
        plt.plot(t_vec, values, label=var_name)
    plt.legend()
    plt.show()


def execute():
    dct = run()
    plot(dct)


if __name__ == "__main__":
    execute()
