# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:51:52 2020

@author: Reuben

This example for Tutorial 3 illustrates how to use the Timeseries class.

"""

import numpy as np
import matplotlib.pyplot as plt

import npsolve
from npsolve.utils import Timeseries

from tutorial_2 import run


class Particle(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
        self.time_points = np.linspace(0, 1, 51)
        np.random.seed(0)
        self.positions = np.random.rand(51, 2) * 10
        self.xts = Timeseries(self.time_points, self.positions[:,0])
        self.yts = Timeseries(self.time_points, self.positions[:,1])
        self.add_var('position', init=self.positions[0,:])
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        velocity = np.array([self.xts(t, der=1), self.yts(t, der=1)])
        derivatives = {'position': velocity}
        return derivatives


def plot(dct, particle):
    plt.plot(dct['position'][:,0], dct['position'][:,1], linewidth=0.5)
    plt.scatter(particle.positions[:,0], particle.positions[:,1], c='r',
                marker='.')
    plt.xlabel('x')
    plt.ylabel('y')

def plot_vs_time(dct, particle):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(dct['time'], dct['position'][:,0], linewidth=0.5)
    axes[0].scatter(particle.time_points, particle.positions[:,0], c='r',
                marker='.')
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('x')
    axes[1].plot(dct['time'], dct['position'][:,1], linewidth=0.5)
    axes[1].scatter(particle.time_points, particle.positions[:,1], c='r',
                marker='.')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('y')