# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:48:33 2019

@author: Reuben
"""

import numpy as np
import npsolve

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
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
        super().__init__()  # Don't forget to call this!
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
    
    def set_model(self, model):
        self.model = model
        self.connect(model)
		
    def connect(self, model):
        self.remove_signals()
        self.setup_signals()
        for k, e in model.elements.items():
            e.connect()
        self.close_signals()
    
    
class Model():
    def __init__(self):
        self.elements = {}
		
    def add_element(self, key, element):
        self.elements[key] = element
    
    
def make_model():
    m = Model()
    m.add_element('component 1', Component1())
    m.add_element('component 2', Component2())
    return m
    
def make_solver():
    return Solver()

def run():
    solver = make_solver()
    model = make_model()
    solver.set_model(model)

    # Initialise the solver
    solver.npsolve_init()
	
    # Now we can run!
    res = solver.solve()
    return res, solver

import matplotlib.pyplot as plt

def plot(res, s):
    slices = s.npsolve_slices
    
    plt.plot(s.t_vec, res[:,slices['position']], label='position')
    plt.plot(s.t_vec, res[:,slices['velocity']], label='velocity')
    plt.plot(s.t_vec, res[:,slices['force']], label='force')
    plt.legend()