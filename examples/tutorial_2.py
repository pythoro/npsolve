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
POS = 'position'
VEL = 'velocity'


class Ball:
    def __init__(self, mass=1.0,
                 k_bounce=1e7,
                 c_bounce=3e3):
        self.mass = mass
        self.k_bounce = k_bounce
        self.c_bounce = c_bounce

    def F_gravity(self):
        """ Force of gravity """
        return np.array([0, self.mass * G])

    def F_ledge(self, state):
        x_pos = state[POS][0]
        return -self.F_gravity() * below(x_pos, limit=X_LEDGE)

    def F_bounce(self, state):
        """ Force bouncing on the surface """
        y_pos = state[POS][1]
        y_vel = state[VEL][1]
        F_spring = -self.k_bounce * negdiff(y_pos, limit=Y_SURFACE)
        c_damping = -self.c_bounce * below(y_pos, Y_SURFACE)
        F_damping = c_damping * negdiff(y_vel, limit=0)
        return np.array([0, F_spring + F_damping])
    
    def step(self, state, t, log):
        """ Called by the solver at each time step """
        F_net = self.F_gravity() + self.F_ledge(state) + self.F_bounce(state)
        acceleration = F_net / self.mass
        derivatives = {POS: state[VEL],
                       VEL: acceleration}
        return derivatives


def get_package():
    ball = Ball()
    package = npsolve.Package()
    package.add_component(ball, 'ball', 'step')
    return package


def solve(package, t_end=3.0):
    framerate = 100001 / 3.0
    ode_integrator = npsolve.solvers.ODEIntegrator(framerate=framerate)
    dct = ode_integrator.run(package, t_end)
    return dct

def run(t_end=3.0, n=100001):
    package = get_package()
    inits = {POS: np.array([0.0, Y_LEDGE]),
             VEL: np.array([5.0, 0.0]),
    }
    package.setup(inits)
    dct = solve(package)
    return dct

def plot(dct):
    plt.figure()
    plt.plot(dct[POS][:,0], dct[POS][:,1], label='position')
    plt.axhline(Y_SURFACE, c='r')
    plt.plot([0, X_LEDGE], [Y_LEDGE, Y_LEDGE], 'r:')
    plt.ylim(0, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def execute():
    dct = run()
    plot(dct)

if __name__ == '__main__':
    execute()