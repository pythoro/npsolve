# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:48:33 2019

@author: Reuben
"""

import numpy as np
import npsolve

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
    def set_vectors(self, state_dct, ret_dct):
        ''' Set some state views for use during calculations '''
        self.position = state_dct['position']
        self.velocity = state_dct['velocity']
        self.force = state_dct['force']
    
    def step(self, state_dct, *args):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        acceleration = 1.0 * self.force
        derivatives = {'position': self.velocity,
                       'velocity': acceleration}
        return derivatives

class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('force', init=-0.1)

    def calculate(self, state_dct, t):
        ''' Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        '''
        dc = 1.0 * np.cos(2*t) * state_dct['position']
        derivatives = {'force': dc}
        return derivatives
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step '''
        return self.calculate(state_dct, t)
        
        
from scipy.integrate import odeint

class Solver(npsolve.Solver):
    def solve(self):
        self.t_vec = np.linspace(0, 10, 1001)
        result = odeint(self.step, self.npsolve_initial_values, self.t_vec)
        return result
    
    
def run():
    s = Solver()
    c1 = Component1()
    c2 = Component2()
    s.npsolve_init()
    # Now we can run!
    res = s.solve()
    return res, s

import matplotlib.pyplot as plt

def plot(res, s):
    slices = s.npsolve_slices
    
    plt.plot(s.t_vec, res[:,slices['position']], label='position')
    plt.plot(s.t_vec, res[:,slices['velocity']], label='velocity')
    plt.plot(s.t_vec, res[:,slices['force']], label='force')
    plt.legend()