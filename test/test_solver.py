# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:43:48 2019

@author: Reuben
"""

import unittest
import numpy as np

from npsolve.core import sb, SET_VECTORS, GET_INIT, GET_VARIABLES, \
    VECTORS_SET, Solver



class S(Solver):
    pass



class Test_Solver(unittest.TestCase):

       
    def test_create(self):
        s = S()
        
    def test_setup_vecs(self):
        s = S()
        dct = {'a': np.array([1.1]), 'b': np.array([2.2])}
        slices, state, ret = s._setup_vecs(dct)
        self.assertEqual((state==np.array([1.1, 2.2])).all(), True)
        self.assertEqual(slices['a'], slice(0, 1))
        self.assertEqual(slices['b'], slice(1, 2))
        
    def test_get_init(self):
        s = S()
        
        def get_init_a():
            return {'a': np.array([1.1]), 'b': np.array([2.2])}

        def get_init_b():
            return {'c': np.array([3.3, 4.4])}
        
        s._signals[GET_INIT].connect(get_init_a)
        s._signals[GET_INIT].connect(get_init_b)
        
        s._get_init()
        state = s.npsolve_state
        slices = s.npsolve_slices
        self.assertEqual((state==np.array([1.1, 2.2, 3.3, 4.4])).all(), True)
        self.assertEqual(slices['a'], slice(0, 1))
        self.assertEqual(slices['b'], slice(1, 2))
        self.assertEqual(slices['c'], slice(2, 4))

    def test_set_vectors(self):
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
            
        s._signals[SET_VECTORS].connect(test_receiver)
        
        s._set_vectors()
    
        self.assertEqual((dct['state']==s.npsolve_state).all(), True)
        self.assertEqual((dct['ret']==s.npsolve_ret).all(), True)
        self.assertEqual(dct['slices'], s.npsolve_slices)

        