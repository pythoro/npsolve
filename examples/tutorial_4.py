# -*- coding: utf-8 -*-
"""
Created on Sun May 24 07:23:55 2020

@author: Reuben

This example for Tutorial 4 illustrates how to use fastwire to pass
values between classes.

"""

import numpy as np
import matplotlib.pyplot as plt
import npsolve
from tutorial_2 import run

G = np.array([0, -9.80665])

import fastwire as fw
wire_box = fw.get_wire_box('demo')


class Slider(npsolve.Partial, fw.Wired):
    def __init__(self, freq=1.0, mass=1.0):
        super().__init__() # Don't forget to call this!
        self.freq = freq
        self.mass = mass
        self.add_var('s_pos', init=np.zeros(2))
        self.add_var('s_vel', init=np.zeros(2))
    
    @wire_box.supply('pivot')
    def pivot(self):
        """ The location of the pivot that connects to the pendulum """
        return self.state['s_pos'], self.state['s_vel']

    def F_sinusoid(self, t):
        """ The force to make the system do something """
        return 10 * np.cos(2 * np.pi * (self.freq * t))

    def step(self, state_dct, t, *args):
        """ Called by the solver at each time step  """
        F_pivot = -wire_box['F_pivot'].fetch(t)
        F_pivot_x = F_pivot[0]
        F_sinusoid_x = self.F_sinusoid(t)
        F_net_x = F_pivot_x + F_sinusoid_x
        acc = np.array([F_net_x / self.mass, 0])
        derivatives = {'s_pos': state_dct['s_vel'],
                       's_vel': acc}
        return derivatives


class Pendulum(npsolve.Partial, fw.Wired):
    def __init__(self, mass=1.0, k=1e6, c=1e4, l=1.0):
        super().__init__() # Don't forget to call this!
        self.mass = mass
        self.k = k
        self.c = c
        self.l = l
        self.add_var('p_pos', init=np.array([0, -self.l]))
        self.add_var('p_vel', init=np.array([0, 0]))
    
    @wire_box.supply('F_pivot')
    @npsolve.mono_cached()
    def F_pivot(self, t):
        """ Work out the force on the pendulum mass """
        pivot_pos, pivot_vel = wire_box['pivot'].fetch()
        rel_pos = pivot_pos - self.state['p_pos']
        rel_vel = pivot_vel - self.state['p_vel']
        dist = np.linalg.norm(rel_pos)
        unit_vec = rel_pos / dist
        F_spring = self.k * (dist - self.l) * unit_vec
        rel_vel_in_line = np.dot(rel_vel, unit_vec)
        F_damping = self.c * rel_vel_in_line * unit_vec
        return F_spring + F_damping
    
    def F_gravity(self):
        return self.mass * G
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        F_net = self.F_pivot(t) + self.F_gravity()
        acceleration = F_net / self.mass
        derivatives = {'p_pos': state_dct['p_vel'],
                       'p_vel': acceleration}
        return derivatives


def plot_xs(dct):
    plt.plot(dct['time'], dct['s_pos'][:,0], label='slider')
    plt.plot(dct['time'], dct['p_pos'][:,0], label='pendulum')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(loc=3)

def plot_trajectories(dct):
    plt.plot(dct['s_pos'][:,0], dct['s_pos'][:,1], label='slider')
    plt.plot(dct['p_pos'][:,0], dct['p_pos'][:,1], label='pendulum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal')
    plt.legend(loc=2)

def execute(freq):
    partials = [Slider(freq=freq), Pendulum()]
    dct = run(partials, t_end=10.0, n=10001)
    plot_xs(dct)
    plot_trajectories(dct)
