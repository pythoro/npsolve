# -*- coding: utf-8 -*-
"""
Created on Sun May 24 07:23:55 2020

@author: Reuben

This example for Tutorial 4 illustrates how to use dependency injection to pass
values between classes.

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import npsolve

G = np.array([0, -9.80665])

SPOS = "slider_pos"
SVEL = "slider_vel"
PPOS = "pendulum_pos"
PVEL = "pendulum_vel"


class Slider:
    def __init__(self, freq=1.0, mass=1.0):
        self.freq = freq
        self.mass = mass

    def set_F_tether(self, F_tether):
        self._F_tether = F_tether

    def pos(self, state, t):
        """The location of the tether connection."""
        return state[SPOS]

    def vel(self, state, t):
        """The velocity of the tether connection."""
        return state[SVEL]

    def F_sinusoid(self, t):
        """The force to make the system do something"""
        return 10 * np.cos(2 * np.pi * (self.freq * t))

    def get_derivs(self, state, t, log):
        """Called by the solver at each time step"""
        F_tether = -self._F_tether
        F_tether_x = F_tether[0]
        F_sinusoid_x = self.F_sinusoid(t)
        F_net_x = F_tether_x + F_sinusoid_x
        acc = np.array([F_net_x / self.mass, 0])
        derivatives = {SPOS: state[SVEL], SVEL: acc}
        return derivatives


class Tether:
    def __init__(self, k=1e6, c=1e4, length=1.0):
        self.k = k
        self.c = c
        self.length = length

    def get_pendulum_init(self, slider_pos: np.ndarray[float], mass: float):
        offset = np.array([0, -self.length])
        stretch = G / self.k
        return slider_pos + offset + stretch

    def F_tether(self, slider_pos, slider_vel, pendulum_pos, pendulum_vel):
        """Work out the force on the pendulum mass"""
        rel_pos = slider_pos - pendulum_pos
        rel_vel = slider_vel - pendulum_vel
        dist = np.linalg.norm(rel_pos)
        unit_vec = rel_pos / dist
        F_spring = self.k * (dist - self.length) * unit_vec
        rel_vel_in_line = np.dot(rel_vel, unit_vec)
        F_damping = self.c * rel_vel_in_line * unit_vec
        return F_spring + F_damping


class Pendulum(npsolve.Partial):
    def __init__(self, mass=1.0):
        self.mass = mass

    def set_F_tether(self, F_tether):
        self._F_tether = F_tether

    def pos(self, state, t):
        """The location of the tether connection."""
        return state[PPOS]

    def vel(self, state, t):
        """The velocity of the tether connection."""
        return state[PVEL]

    def F_gravity(self):
        return self.mass * G

    def get_derivs(self, state, t, log):
        """Called by the solver at each time step
        Calculate acceleration based on the
        """
        F_net = self._F_tether + self.F_gravity()
        acceleration = F_net / self.mass
        derivatives = {PPOS: state[PVEL], PVEL: acceleration}
        return derivatives


class Assembly:
    def __init__(self, slider: Slider, pendulum: Pendulum, tether: Tether):
        self._slider = slider
        self._pendulum = pendulum
        self._tether = tether

    def set_tether_forces(self, state, t, log):
        slider = self._slider
        pendulum = self._pendulum
        slider_pos = slider.pos(state, t)
        slider_vel = slider.vel(state, t)
        pendulum_pos = pendulum.pos(state, t)
        pendulum_vel = pendulum.vel(state, t)
        F_tether = self._tether.F_tether(
            slider_pos, slider_vel, pendulum_pos, pendulum_vel
        )
        slider.set_F_tether(F_tether)
        pendulum.set_F_tether(F_tether)


def get_package():
    slider = Slider()
    pendulum = Pendulum()
    tether = Tether()
    assembly = Assembly(slider, pendulum, tether)
    package = npsolve.Package()
    package.add_component(slider, "slider", "get_derivs")
    package.add_component(pendulum, "pendulum", "get_derivs")
    package.add_component(tether, "tether", None)
    package.add_component(assembly, "assembly", None)
    package.add_stage_call("assembly", "set_tether_forces")
    return package


def solve(package, n=100001, t_end=1.0):
    framerate = (n - 1) / t_end
    ode_integrator = npsolve.solvers.ODEIntegrator(framerate=framerate)
    dct = ode_integrator.run(package, t_end)
    return dct


def get_inits(package):
    slider_pos = np.zeros(2)
    pend_mass = package["pendulum"].mass
    inits = {
        SPOS: slider_pos,
        SVEL: np.zeros(2),
        PPOS: package["tether"].get_pendulum_init(slider_pos, pend_mass),
        PVEL: np.zeros(2),
    }
    return inits


def run(t_end=1.0, n=100001):
    package = get_package()
    inits = get_inits(package)
    package.setup(inits)
    dct = solve(package, n, t_end)
    return dct


def plot_xs(dct):
    plt.figure()
    plt.plot(dct["time"], dct[SPOS][:, 0], label="slider")
    plt.plot(dct["time"], dct[PPOS][:, 0], label="pendulum")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.legend(loc=3)
    plt.show()


def plot_trajectories(dct):
    plt.figure()
    plt.plot(dct[SPOS][:, 0], dct[SPOS][:, 1], label="slider")
    plt.plot(dct[PPOS][:, 0], dct[PPOS][:, 1], label="pendulum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect("equal")
    plt.legend(loc=2)
    plt.show()


def plot_distance_check(dct):
    plt.figure()
    diff = dct[PPOS] - dct[SPOS]
    dist = np.linalg.norm(diff, axis=1)
    plt.plot(dct["time"], dist)
    plt.xlabel("time")
    plt.ylabel("length")
    plt.show()


def execute():
    dct = run(t_end=10.0, n=10001)
    plot_xs(dct)
    plot_trajectories(dct)
    plot_distance_check(dct)


if __name__ == "__main__":
    execute()
