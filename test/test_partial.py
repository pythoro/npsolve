# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:43:48 2019

@author: Reuben
"""

import unittest
import numpy as np

from npsolve.core import sb, SET_VECTORS, GET_INIT, GET_VARIABLES, Partial

def make_signals():
    sb.get_active().clear()
    s_names = [SET_VECTORS, GET_INIT, GET_VARIABLES]
    signals = {name: sb.get(name) for name in s_names}
    return signals

class P(Partial):
    def __init__(self):
        super().__init__()
        self._names = ['a', 'b']
        self.a = 0.7
        self.b = 5.0


def make_p():
    signals = make_signals()
    p = P()
    ID = str(id(p))
    a = ID + 'a'
    b = ID + 'b'
    slices = {a: slice(0, 1), b: slice(1, 2)}
    state = np.array([1.3, 5.6])
    ret = np.array([7.0, 6.3])
    signals[SET_VECTORS].emit(state=state, ret=ret, slices=slices)
    return p, state, ret, slices


class Test_Partial(unittest.TestCase):

    def test_create_before_solver(self):
        def test_fun():
            sb.get_active().clear()
            p = Partial()    
        self.assertRaises(KeyError, test_fun)
        
    def test_create(self):
        signals = make_signals()
        p = P()
        
    def test_get_init(self):
        signals = make_signals()
        p = P()
        dicts = signals[GET_INIT].fetch_all()
        ID = str(id(p))
        dct = {ID + 'a': np.array([0.7]),
               ID + 'b': np.array([5.0])}
        self.assertEqual(dicts[0], dct)
        
    def test_set_vectors(self):
        p, state, ret, slices = make_p()
        self.assertEqual(p.a, 1.3)
        self.assertEqual(p.a.flags['OWNDATA'], False)
        self.assertEqual(p.b, 5.6)
        self.assertEqual(p.b.flags['OWNDATA'], False)

    def test_update_state(self):
        p, state, ret, slices = make_p()
        state[:] = [33.0, 66.0]
        self.assertEqual(p.a, 33.0)
        self.assertEqual(p.b, 66.0)

    def test_set_return(self):
        p, state, ret, slices = make_p()
        p.set_return('a', 55.1)
        p.set_return('b', 300.0)
        self.assertEqual(ret[0], 55.1)
        self.assertEqual(ret[1], 300.0)
