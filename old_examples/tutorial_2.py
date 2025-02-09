# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:43:17 2020

@author: Reuben

This example for Tutorial 2 illustrates how to handle discontinuities.

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import npsolve
from npsolve.soft_functions import negdiff, below

G = -9.80665
Y_SURFACE = 1.5
X_LEDGE = 2.0
Y_LEDGE = 5.0

class Ball(npsolve.Partial):
    def __init__(self, mass=1.0,
                 k_bounce=1e7,
                 c_bounce=3e3,
                 initial_vel=5.0):
        super().__init__() # Don't forget to call this!
        self.mass = mass
        self.k_bounce = k_bounce
        self.c_bounce = c_bounce
        self.add_var('position', init=np.array([0.0, Y_LEDGE]))
        self.add_var('velocity', init=np.array([initial_vel, 0.0]))

    def F_gravity(self):
        """ Force of gravity """
        return np.array([0, self.mass * G])

    def F_ledge(self):
        x_pos = self.state['position'][0]
        return -self.F_gravity() * below(x_pos, limit=X_LEDGE)

    def F_bounce(self):
        """ Force bouncing on the surface """
        y_pos = self.state['position'][1]
        y_vel = self.state['velocity'][1]
        F_spring = -self.k_bounce * negdiff(y_pos, limit=Y_SURFACE)
        c_damping = -self.c_bounce * below(y_pos, Y_SURFACE)
        F_damping = c_damping * negdiff(y_vel, limit=0)
        return np.array([0, F_spring + F_damping])
    
    def step(self, state_dct, t, *args):
        """ Called by the solver at each time step """
        F_net = self.F_gravity() + self.F_ledge() + self.F_bounce()
        acceleration = F_net / self.mass
        derivatives = {'position': self.state['velocity'],
                       'velocity': acceleration}
        return derivatives

       
class Solver(npsolve.Solver):
    def solve(self, t_end=3.0, n=100001):
        self.npsolve_init() # Initialise
        t_vec = np.linspace(0, t_end, n)
        solution = odeint(self.step, self.npsolve_initial_values, t_vec)
        dct = self.as_dct(solution)
        dct['time'] = t_vec
        return dct


def run(t_end=3.0, n=100001):
    ball = Ball()
    solver = Solver()
    solver.connect_partial(ball)
    return solver.solve(t_end=t_end, n=n)

def plot(dct):
    plt.figure()
    plt.plot(dct['position'][:,0], dct['position'][:,1], label='position')
    plt.axhline(Y_SURFACE, c='r')
    plt.plot([0, X_LEDGE], [Y_LEDGE, Y_LEDGE], 'r:')
    plt.ylim(0, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    
def execute():
    dct = run()
    plot(dct)
