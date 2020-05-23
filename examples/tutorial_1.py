# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:48:33 2019

@author: Reuben

This example for Tutorial 1 introduces the basics of npsolve.

"""

import numpy as np
import npsolve

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        acceleration = 1.0 * self.state['force']
        derivatives = {'position': self.state['velocity'],
                       'velocity': acceleration}
        return derivatives


class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var('force', init=-0.1)

    def calculate(self, t):
        ''' Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        '''
        dc = 1.0 * np.cos(2*t) * self.state['position']
        derivatives = {'force': dc}
        return derivatives
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step '''
        return self.calculate(t)
        
        

from scipy.integrate import odeint

class Solver(npsolve.Solver):
    def solve(self):
        self.npsolve_init() # Initialise
        self.t_vec = np.linspace(0, 10, 1001)
        result = odeint(self.step, self.npsolve_initial_values, self.t_vec)
        return result


def run():
    solver = Solver()
    partials = [Component1(), Component2()]
    solver.connect(partials)
    res = solver.solve()
    return res, solver


import matplotlib.pyplot as plt

def plot(res, s):
    slices = s.npsolve_slices
    
    plt.plot(s.t_vec, res[:,slices['position']], label='position')
    plt.plot(s.t_vec, res[:,slices['velocity']], label='velocity')
    plt.plot(s.t_vec, res[:,slices['force']], label='force')
    plt.legend()