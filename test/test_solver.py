# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:43:48 2019

@author: Reuben
"""

import unittest
import numpy as np

from npsolve.core import sb, EMIT_VECTORS, GET_VARS, GET_STEP_METHODS, \
    Solver



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
        
    def test_make_dcts(self):
        s = S()
        dct = {'a': {'init': np.array([1.1])}, 'b': {'init': np.array([2.2])}}
        slices, state, ret = s._setup_vecs(dct)
        state_dct, ret_dct = s._make_dcts(slices, state, ret)
        self.assertEqual(state_dct['a'], np.array([1.1]))
        self.assertEqual(state_dct['b'], np.array([2.2]))
        self.assertEqual(state_dct['a'].flags['WRITEABLE'], False)
        self.assertEqual(state_dct['b'].flags['WRITEABLE'], False)
        
    def test_fetch_vars(self):
        s = S()
        
        def get_init_a():
            return {'a': {'init': np.array([1.1])},
                    'b': {'init': np.array([2.2])}}

        def get_init_b():
            return {'c': {'init': np.array([3.3, 4.4])}}
        
        s._signals[GET_VARS].connect(get_init_a)
        s._signals[GET_VARS].connect(get_init_b)
        
        dct = s._fetch_vars()
        slices, state, ret = s._setup_vecs(dct)
        self.assertEqual((state==np.array([1.1, 2.2, 3.3, 4.4])).all(), True)
        self.assertEqual(slices['a'], slice(0, 1))
        self.assertEqual(slices['b'], slice(1, 2))
        self.assertEqual(slices['c'], slice(2, 4))

    def test_emit_vectors(self):
        s = S()
        state = np.array([1.1, 2.2, 3.3, 4.4])
        ret = np.zeros(4)
        a_arr = state[0:1]
        a_arr.flags['WRITEABLE'] = False
        b_arr = state[1:2]
        b_arr.flags['WRITEABLE'] = False
        c_arr = state[2:4]
        c_arr.flags['WRITEABLE'] = False
        state_dct = {'a': a_arr, 'b': b_arr, 'c': c_arr}
        ret_dct = {'a': ret[0:1], 'b': ret[1:2], 'c': ret[2:4]}

        s.npsolve_state_dct = state_dct
        s.npsolve_ret_dct = ret_dct
        
        dct = {}
        def test_receiver(state_dct, ret_dct):
            dct['state_dct'] = state_dct
            dct['ret_dct'] = ret_dct
            
        s._signals[EMIT_VECTORS].connect(test_receiver)
        
        s._emit_vectors()
    
        self.assertEqual(dct['state_dct'], s.npsolve_state_dct)
        self.assertEqual(dct['ret_dct'], s.npsolve_ret_dct)

    def test_fetch_step_methods(self):
        s = S()
        
        def step_a():
            pass

        def step_b():
            pass
        
        def get_step_a():
            return step_a

        def get_step_b():
            return step_b
        
        s._signals[GET_STEP_METHODS].connect(get_step_a)
        s._signals[GET_STEP_METHODS].connect(get_step_b)
        
        lst = s._fetch_step_methods()
        self.assertEqual(lst, [step_a, step_b])

    def test_step(self):
        s = S()
        
        def step(state_dct):
            return {'a': state_dct['a'] * 2}
        
        s._step_methods = [step]
        
        state = np.array([1.1])
        ret = np.zeros(1)
        a_arr = state[0:1]
        a_arr.flags['WRITEABLE'] = False
        state_dct = {'a': a_arr}
        ret_dct = {'a': ret[0:1]}
        s.npsolve_state = state
        s.npsolve_ret = ret
        s.npsolve_state_dct = state_dct
        s.npsolve_ret_dct = ret_dct
        
        vec = np.array([3.3])
        ret_arr = s.step(vec)
        
        self.assertEqual(ret_arr, np.array([6.6]))
        self.assertEqual(s.npsolve_ret, np.array([6.6]))