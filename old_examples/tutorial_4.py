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


class Slider(npsolve.Partial):
    def __init__(self, freq=1.0, mass=1.0):
        super().__init__()  # Don't forget to call this!
        self.freq = freq
        self.mass = mass
        self.add_var("s_pos", init=np.zeros(2))
        self.add_var("s_vel", init=np.zeros(2))

    def connect_to_pendulum(self, pendulum):
        self._pendulum = pendulum
        pendulum.connect_to_slider(self)

    def pivot(self, t):
        """The location of the pivot that connects to the pendulum"""
        return self.state["s_pos"], self.state["s_vel"]

    def F_sinusoid(self, t):
        """The force to make the system do something"""
        return 10 * np.cos(2 * np.pi * (self.freq * t))

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step"""
        F_pivot = -self._pendulum.F_pivot(t)
        F_pivot_x = F_pivot[0]
        F_sinusoid_x = self.F_sinusoid(t)
        F_net_x = F_pivot_x + F_sinusoid_x
        acc = np.array([F_net_x / self.mass, 0])
        derivatives = {"s_pos": state_dct["s_vel"], "s_vel": acc}
        return derivatives


class Pendulum(npsolve.Partial):
    def __init__(self, mass=1.0, k=1e6, c=1e4, l=1.0):
        super().__init__()  # Don't forget to call this!
        self.mass = mass
        self.k = k
        self.c = c
        self.l = l
        self.add_var("p_pos", init=np.array([0, -self.l]))
        self.add_var("p_vel", init=np.array([0, 0]))

    def connect_to_slider(self, slider):
        self._slider = slider

    @npsolve.mono_cached()
    def F_pivot(self, t):
        """Work out the force on the pendulum mass"""
        pivot_pos, pivot_vel = self._slider.pivot(t)  # Will be up to date
        rel_pos = pivot_pos - self.state["p_pos"]
        rel_vel = pivot_vel - self.state["p_vel"]
        dist = np.linalg.norm(rel_pos)
        unit_vec = rel_pos / dist
        F_spring = self.k * (dist - self.l) * unit_vec
        rel_vel_in_line = np.dot(rel_vel, unit_vec)
        F_damping = self.c * rel_vel_in_line * unit_vec
        return F_spring + F_damping

    def F_gravity(self):
        return self.mass * G

    def step(self, state_dct, t, *args):
        """Called by the solver at each time step
        Calculate acceleration based on the
        """
        F_net = self.F_pivot(t) + self.F_gravity()
        acceleration = F_net / self.mass
        derivatives = {"p_pos": state_dct["p_vel"], "p_vel": acceleration}
        return derivatives


class Solver(npsolve.Solver):
    def solve(self, t_end=3.0, n=100001):
        self.npsolve_init()  # Initialise
        t_vec = np.linspace(0, t_end, n)
        solution = odeint(self.step, self.npsolve_initial_values, t_vec)
        dct = self.as_dct(solution)
        dct["time"] = t_vec
        return dct


def run(t_end=3.0, n=100001):
    slider = Slider()
    pendulum = Pendulum()
    slider.connect_to_pendulum(pendulum)
    partials = [slider, pendulum]
    solver = Solver()
    solver.connect(partials)
    return solver.solve(t_end=t_end, n=n)


def plot_xs(dct):
    plt.figure()
    plt.plot(dct["time"], dct["s_pos"][:, 0], label="slider")
    plt.plot(dct["time"], dct["p_pos"][:, 0], label="pendulum")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.legend(loc=3)


def plot_trajectories(dct):
    plt.figure()
    plt.plot(dct["s_pos"][:, 0], dct["s_pos"][:, 1], label="slider")
    plt.plot(dct["p_pos"][:, 0], dct["p_pos"][:, 1], label="pendulum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect("equal")
    plt.legend(loc=2)


def plot_distance_check(dct):
    plt.figure()
    diff = dct["p_pos"] - dct["s_pos"]
    dist = np.linalg.norm(diff, axis=1)
    plt.plot(dct["time"], dist)
    plt.xlabel("time")
    plt.ylabel("length")


def execute():
    dct = run(t_end=10.0, n=10001)
    plot_xs(dct)
    plot_trajectories(dct)
    plot_distance_check(dct)
