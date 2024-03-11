# -*- coding: utf-8 -*-
"""
Created on Mon May 25 06:59:10 2020

@author: Reuben

This example for Tutorial 6 illustrates how to log values during the
solving and add them to the output. It also uses the Integrator class,
which provides some additional functionality for doing this.

"""

import matplotlib.pyplot as plt
import npsolve
from tutorial_4 import Slider, Pendulum

from npsolve.solvers import FINAL, STOP
status = npsolve.get_status('demos_status')
logger = npsolve.get_logger('demos_logger')


class Pendulum2(Pendulum):
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        F_pivot = self.F_pivot(t)
        F_gravity = self.F_gravity()
        F_net = F_pivot + F_gravity
        acceleration = F_net / self.mass
        if status[FINAL]:
            logger['F_pivot'].append(F_pivot)
            logger['acceleration'].append(acceleration)
        if F_pivot[1] > 90.0:
            status[STOP] = True
        derivatives = {'p_pos': state_dct['p_vel'],
                       'p_vel': acceleration}
        return derivatives


def run(freq, t_end=20.0, n=100001):
    slider = Slider(freq=freq)
    pendulum = Pendulum2()
    slider.connect_to_pendulum(pendulum)
    solver = npsolve.solvers.Integrator(status=status,
                                        logger=logger,
                                        framerate=n//t_end)
    partials = [slider, pendulum]
    solver.connect(partials)
    return solver.run(t_end)


def plot_F_pivot(dct):
    plt.figure()
    plt.plot(dct['time'], dct['F_pivot'][:,0], label='F_pivot_x')
    plt.plot(dct['time'], dct['F_pivot'][:,1], label='F_pivot_y')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(loc=3)

def plot_acc(dct):
    plt.figure()
    plt.plot(dct['time'], dct['p_vel'][:,0], label='x_velocity')
    plt.plot(dct['time'], dct['acceleration'][:,0], label='x_acceleration')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(loc=3)


def execute(freq):
    dct = run(freq=freq, t_end=20.0, n=10001)
    plot_F_pivot(dct)
    plot_acc(dct)

