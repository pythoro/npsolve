# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:51:52 2020

@author: Reuben

This example for Tutorial 3 illustrates interpolated data.

"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

import npsolve

POS = 'position'

class Particle():
    def __init__(self, time_points, positions):
        self.time_points = time_points
        self.positions = positions
        self._xts = make_interp_spline(time_points, positions[:,0])
        self._yts = make_interp_spline(time_points, positions[:,1])
    
    def get_init_pos(self):
        return np.array([self._xts(0.0), self._yts(0.0)])
    
    def step(self, state, t, log):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        velocity = np.array([self._xts(t, nu=1), self._yts(t, nu=1)])
        derivatives = {POS: velocity}
        return derivatives


def get_package(time_points, positions):
    particle = Particle(time_points, positions)
    package = npsolve.Package()
    package.add_component(particle, 'particle', 'step')
    return package

def solve(package, n=100001, t_end=1.0):
    framerate = n / t_end
    ode_integrator = npsolve.solvers.ODEIntegrator(framerate=framerate)
    dct = ode_integrator.run(package, t_end)
    return dct

def run(t_end=1.0, n=100001):
    np.random.seed(0)
    time_points = np.linspace(0, 1, 51)
    positions = np.random.rand(51, 2) * 10
    package = get_package(time_points, positions)
    particle: Particle = package['particle']
    inits = {POS: particle.get_init_pos()}
    package.setup(inits)
    dct = solve(package, n, t_end)
    return particle, dct


def plot(dct, particle):
    plt.figure(1)
    plt.plot(dct[POS][:,0], dct[POS][:,1], linewidth=0.5)
    plt.scatter(particle.positions[:,0], particle.positions[:,1], c='r',
                marker='.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_vs_time(dct, particle):
    fig, axes = plt.subplots(2, 1, sharex=True, num=2)
    axes[0].plot(dct['time'], dct[POS][:,0], linewidth=0.5)
    axes[0].scatter(particle.time_points, particle.positions[:,0], c='r',
                marker='.')
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('x')
    axes[1].plot(dct['time'], dct[POS][:,1], linewidth=0.5)
    axes[1].scatter(particle.time_points, particle.positions[:,1], c='r',
                marker='.')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('y')
    plt.show()
    
def execute():
    particle, dct = run()
    plot(dct, particle)
    plot_vs_time(dct, particle)
    
if __name__ == '__main__':
    execute()