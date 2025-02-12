# -*- coding: utf-8 -*-
"""
Created on Mon May 25 06:59:10 2020

@author: Reuben

This example for Tutorial 6 illustrates how to use the log functionality,
which includes a stop flag to stop the ODEIntegrator early.

"""

import matplotlib.pyplot as plt
import npsolve
from tutorial_4 import Slider, Pendulum, Tether, Assembly, PPOS, PVEL
from tutorial_4 import get_inits, solve

from npsolve.solvers import STOP

status = npsolve.get_status("demos_status")
logger = npsolve.get_logger("demos_logger")


class Pendulum2(Pendulum):
    def get_derivs(self, state, t, log):
        """Called by the solver at each time step
        Calculate acceleration based on the
        """
        F_net = self._F_tether + self.F_gravity()
        acceleration = F_net / self.mass
        if log:
            log["F_tether"] = self._F_tether
            log["acceleration"] = acceleration
            if self._F_tether[1] > 24.0:
                log[STOP] = True
        derivatives = {PPOS: state[PVEL], PVEL: acceleration}
        return derivatives


def get_package():
    slider = Slider()
    pendulum = Pendulum2()
    tether = Tether()
    assembly = Assembly(slider, pendulum, tether)
    package = npsolve.Package()
    package.add_component(slider, "slider", "get_derivs")
    package.add_component(pendulum, "pendulum", "get_derivs")
    package.add_component(tether, "tether", None)
    package.add_component(assembly, "assembly", None)
    package.add_stage_call("assembly", "set_tether_forces")
    return package


def run(t_end=20.0, n=100001):
    package = get_package()
    inits = get_inits(package)
    package.setup(inits)
    dct = solve(package, n, t_end)
    return dct


def plot_pivot_force(dct):
    plt.figure()
    plt.plot(dct["F_tether"][:, 0], dct["F_tether"][:, 1], label="F_tether_y")
    plt.xlabel("Force in x")
    plt.ylabel("Force in y")
    plt.legend(loc=3)
    plt.show()


def plot_F_tether_vs_time(dct):
    plt.figure()
    plt.plot(dct["time"], dct["F_tether"][:, 0], label="F_tether_x")
    plt.plot(dct["time"], dct["F_tether"][:, 1], label="F_tether_y")
    plt.xlabel("time")
    plt.ylabel("Pivot force")
    plt.legend(loc=3)
    plt.show()


def plot_acc(dct):
    plt.figure()
    plt.plot(dct["time"], dct[PVEL][:, 0], label="x_velocity")
    plt.plot(dct["time"], dct["acceleration"][:, 0], label="x_acceleration")
    plt.xlabel("time")
    plt.ylabel("x-acceleration")
    plt.legend(loc=3)
    plt.show()


def execute():
    dct = run(t_end=20.0, n=10001)
    plot_pivot_force(dct)
    plot_F_tether_vs_time(dct)
    plot_acc(dct)


if __name__ == "__main__":
    execute()
