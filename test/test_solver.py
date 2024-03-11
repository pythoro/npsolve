# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:43:48 2019

@author: Reuben
"""

import unittest
import numpy as np

from npsolve.core import Solver


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
        
        class MockPartial:
            def __init__(self, dct):
                self.dct = dct
            
            def _get_vars(self):
                return self.dct
        
        dct_a = {'a': {'init': np.array([1.1])},
                    'b': {'init': np.array([2.2])}}

        dct_b = {'c': {'init': np.array([3.3, 4.4])}}
        
        mock_a = MockPartial(dct_a)
        mock_b = MockPartial(dct_b)
        
        s.connect_partial(mock_a)
        s.connect_partial(mock_b)
        
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
        
        class MockPartial:
            def __init__(self):
                self.dct = {}
            
            def set_vectors(self, state_dct, ret_dct):
                self.dct = {'state_dct': state_dct,
                            'ret_dct': ret_dct}
        
        p = MockPartial()            
        s.connect_partial(p)
        s._emit_vectors()
        self.assertEqual(p.dct['state_dct'], s.npsolve_state_dct)
        self.assertEqual(p.dct['ret_dct'], s.npsolve_ret_dct)

    def test_fetch_step_methods(self):
        s = S()
        
        class MockPartial:
            def step(self, state_dct, *args):
                pass
            
            def _get_step_method(self):
                return self.step
        
        p_a = MockPartial()
        p_b = MockPartial()
        
        s.connect_partial(p_a)
        s.connect_partial(p_b)
        
        lst = s._fetch_step_methods()
        self.assertEqual(lst, [p_a.step, p_b.step])

    def test_fetch_partials(self):
        s = S()
        
        class MockPartial:
            def __init__(self, name):
                self.npsolve_name = name
        
        p_a = MockPartial('dummy')
        p_b = MockPartial('mock')
        
        s.connect_partial(p_a)
        s.connect_partial(p_b)
        dct = s.fetch_partials()
        self.assertEqual(dct, {'dummy': p_a, 'mock': p_b})

    def test_step(self):
        s = S()
        
        class MockPartial:
            def step(self, state_dct):
                return {'a': state_dct['a'] * 2}
        
        p = MockPartial()
        
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
        s._partials = [p]
        s._step_methods = [p.step]
        
        vec = np.array([3.3])
        ret_arr = s.step(vec)
        
        self.assertEqual(ret_arr, np.array([6.6]))
        self.assertEqual(s.npsolve_ret, np.array([6.6]))