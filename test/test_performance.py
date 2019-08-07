# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 07:22:13 2019

@author: Reuben
"""

import unittest
import numpy as np
import timeit

from npsolve.core import sb, Partial, Solver


class Partial_1(Partial):
    npsolve_name = 'partial_1'
    
    def __init__(self):
        super().__init__()
        self.add_var('a', 1.0)
    
    def set_vectors(self, state_dct, ret_dct):
        self.a = state_dct['a']
    
    def step(self, state_dct, *args):
        return {'a': self.a + 1.0}
        

class Test_Solver(unittest.TestCase):


    def test_create(self):
        s = Solver()
        p = Partial_1()
        s.npsolve_init()
        lst = s.fetch_partials()
        self.assertEqual(lst['partial_1'], p)
       
    def test_step(self):
        s = Solver()
        p = Partial_1()
        
        s.npsolve_init()
        vec = s.npsolve_state
        globals_dct = {'s': s, 'vec': vec}
        time = timeit.timeit('s.step(vec)',
                             globals=globals_dct,
                             number=100000)
        
        ret = vec
        def internal(a):
            return a + 1.0
        
        def step_baseline(vec, ret):
            ret[0] = internal(vec[0])
            return ret
            
        globals_dct = {'step_baseline': step_baseline, 'vec': vec, 'ret': ret}
        baseline = timeit.timeit('step_baseline(vec, ret)',
                                 globals=globals_dct,
                                 number=100000)
        print()
        print('Relative speed: ' + '{:0.3f}'.format(time/baseline))
        self.assertLess(time, baseline*8)
        
        
        