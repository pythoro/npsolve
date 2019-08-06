# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:43:48 2019

@author: Reuben
"""

import unittest
import numpy as np

from npsolve.core import sb, EMIT_VECTORS, GET_VARS, \
    VECTORS_SET, GET_CACHE_CLEAR_FUNCTIONS, Solver



class S(Solver):
    pass



class Test_Solver(unittest.TestCase):

       
    def test_create(self):
        s = S()
        
    def test_setup_vecs(self):
        s = S()
        dct = {'a': {'init': np.array([1.1])}, 'b': {'init': np.array([2.2])}}
        slices, state, ret = s._setup_vecs(dct)
        self.assertEqual((state==np.array([1.1, 2.2])).all(), True)
        self.assertEqual(slices['a'], slice(0, 1))
        self.assertEqual(slices['b'], slice(1, 2))
        
    def test_fetch_vars(self):
        s = S()
        
        def get_init_a():
            return {'a': {'init': np.array([1.1])},
                    'b': {'init': np.array([2.2])}}

        def get_init_b():
            return {'c': {'init': np.array([3.3, 4.4])}}
        
        s._signals[GET_VARS].connect(get_init_a)
        s._signals[GET_VARS].connect(get_init_b)
        
        s._fetch_vars()
        state = s.npsolve_state
        slices = s.npsolve_slices
        self.assertEqual((state==np.array([1.1, 2.2, 3.3, 4.4])).all(), True)
        self.assertEqual(slices['a'], slice(0, 1))
        self.assertEqual(slices['b'], slice(1, 2))
        self.assertEqual(slices['c'], slice(2, 4))

    def test_emit_vectors(self):
        s = S()
        s.npsolve_state = np.array([1.1, 2.2, 3.3, 4.4])
        s.npsolve_ret = np.zeros(4)
        s.npsolve_slices = {'a': slice(0, 1),
                            'b': slice(1, 2),
                            'c': slice(2, 4)}
        
        dct = {}
        def test_receiver(state, ret, slices):
            dct['state'] = state
            dct['ret'] = ret
            dct['slices'] = slices
            
        s._signals[EMIT_VECTORS].connect(test_receiver)
        
        s._emit_vectors()
    
        self.assertEqual((dct['state']==s.npsolve_state).all(), True)
        self.assertEqual((dct['ret']==s.npsolve_ret).all(), True)
        self.assertEqual(dct['slices'], s.npsolve_slices)

        