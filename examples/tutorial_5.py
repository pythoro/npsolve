# -*- coding: utf-8 -*-
"""
Created on Sun May 24 07:23:55 2020

@author: Reuben

This example for Tutorial 4 illustrates how to set initial conditions. It 
attaches the pendulum to a particle that travels through a trajectory.

"""

import npsolve
import numpy as np
import matplotlib.pyplot as plt
from tutorial_3 import Particle, POS
from tutorial_4 import Pendulum, solve, Tether, Assembly, PPOS, PVEL


class Particle2(Particle):

    def set_F_tether(self, F_tether):
        pass

    def pos(self, state, t):
        """The location of the tether connection."""
        return state[POS]

    def vel(self, state, t):
        """The velocity of the tether connection."""
        velocity = np.array([self._xts(t, nu=1), self._yts(t, nu=1)])
        return velocity


def get_package():
    np.random.seed(0)
    time_points = np.linspace(0, 1, 51)
    positions = np.random.rand(51, 2) * 10
    particle = Particle2(time_points, positions)
    pendulum = Pendulum()
    tether = Tether(k=1e7, c=1e4)
    assembly = Assembly(particle, pendulum, tether)
    package = npsolve.Package()
    package.add_component(particle, 'particle', 'step')
    package.add_component(pendulum, 'pendulum', 'get_derivs')
    package.add_component(tether, 'tether', None)
    package.add_component(assembly, 'assembly', None)
    package.add_stage_call('assembly', 'set_tether_forces')
    return package


def get_inits(package):
    slider_pos = package['particle'].get_init_pos()
    inits = {POS: slider_pos,
             PPOS: package['tether'].get_pendulum_init(slider_pos),
             PVEL: np.zeros(2)}
    return inits

def run(t_end=1.0, n=100001):
    package = get_package()
    inits = get_inits(package)
    package.setup(inits)
    dct = solve(package, n, t_end)
    return dct


def plot_trajectories(dct):
    plt.figure()
    plt.plot(dct[POS][:,0], dct[POS][:,1], linewidth=1.0,
             label='particle')
    plt.plot(dct[PPOS][:,0], dct[PPOS][:,1],  linewidth=1.0,
             label='pendulum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2.5, 12.5)
    plt.ylim(-2.5, 12.5)
    plt.gca().set_aspect('equal')
    plt.legend(loc=2)
    plt.show()


def plot_distance_check(dct):
    plt.figure()
    diff = dct[PPOS] - dct[POS]
    dist = np.linalg.norm(diff, axis=1)
    plt.plot(dct['time'], dist)
    plt.xlabel('time')
    plt.ylabel('length')
    plt.show()


def execute():
    dct = run()
    plot_trajectories(dct)
    plot_distance_check(dct)

if __name__ == '__main__':
    execute()
