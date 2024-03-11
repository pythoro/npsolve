# -*- coding: utf-8 -*-
"""
Created on Sun May 24 07:23:55 2020

@author: Reuben

This example for Tutorial 4 illustrates how to set initial conditions.

"""

import numpy as np
import matplotlib.pyplot as plt
from tutorial_3 import Particle
from tutorial_4 import Pendulum, Solver
import fastwire as fw
wire_box = fw.get_wire_box('demo')


class Particle2(Particle, fw.Wired):
        
    def pivot(self, t):
        velocity = np.array([self.xts(t, der=1), self.yts(t, der=1)])
        return self.state['position'], velocity


def set_init_condition(particle, pendulum):
    init_particle_pos = particle.npsolve_vars['position']['init']
    init_pendulum_pos = init_particle_pos - np.array([0.0, 1.0])
    pendulum.set_init('p_pos', init_pendulum_pos)

def run(k=1e6, c=1e4, t_end=1.0, n=100001):
    particle = Particle2()
    pendulum = Pendulum(k=k, c=c)
    pendulum.connect_to_slider(particle)
    set_init_condition(particle, pendulum)
    partials = [particle, pendulum]
    solver = Solver()
    solver.connect(partials)
    return solver.solve(t_end=t_end, n=n)


def plot_trajectories(dct):
    plt.plot(dct['position'][:,0], dct['position'][:,1], linewidth=1.0,
             label='particle')
    plt.plot(dct['p_pos'][:,0], dct['p_pos'][:,1],  linewidth=1.0,
             label='pendulum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2.5, 12.5)
    plt.ylim(-2.5, 12.5)
    plt.gca().set_aspect('equal')
    plt.legend(loc=2)


def plot_distance_check(dct):
    diff = dct['p_pos'] - dct['position']
    dist = np.linalg.norm(diff, axis=1)
    plt.plot(dct['time'], dist)
    plt.xlabel('time')
    plt.ylabel('length')


