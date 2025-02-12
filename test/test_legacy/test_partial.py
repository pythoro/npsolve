# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:43:48 2019

@author: Reuben
"""

import unittest
import numpy as np

from npsolve import Partial
from npsolve import multi_cached, mono_cached


class P(Partial):
    def __init__(self):
        super().__init__()
        self.add_var("a", init=0.7)
        self.add_var("b", init=5.0)

    def step(self, state_dct, *args):
        a = state_dct["a"]
        return {"a": a * 2}


class Cached(P):
    @mono_cached()
    def mono(self, a):
        return np.array([a])

    @mono_cached()
    def mono_b(self, a):
        return np.array([a])

    @multi_cached()
    def multi(self, a):
        return np.array([a])

    @multi_cached()
    def multi_b(self, a):
        return np.array([a])


def make_partial(cls=P):
    p = cls()
    state = np.array([1.3, 5.6])
    ret = np.zeros(2)
    a_arr = state[0:1]
    a_arr.flags["WRITEABLE"] = False
    b_arr = state[1:2]
    b_arr.flags["WRITEABLE"] = False
    state_dct = {"a": a_arr, "b": b_arr}
    ret_dct = {"a": ret[0:1], "b": ret[1:2]}
    p.set_vectors(state_dct=state_dct, ret_dct=ret_dct)
    p.a = state_dct["a"]
    p.b = state_dct["b"]
    return p, state, ret, state_dct, ret_dct


class Test_Partial(unittest.TestCase):
    def test_create(self):
        p = P()

    def test_set_init(self):
        p = P()
        p.set_init("a", 0.8)
        p.set_init("b", 55.1)
        dct = {"a": {"init": np.array([0.8])}, "b": {"init": np.array([55.1])}}
        self.assertEqual(p.npsolve_vars, dct)

    def test_get_init(self):
        p = P()
        var_dct = p._get_vars()
        dct = {"a": {"init": np.array([0.7])}, "b": {"init": np.array([5.0])}}
        self.assertEqual(var_dct, dct)

    def test_set_vectors(self):
        p, state, ret, state_dct, ret_dct = make_partial()
        self.assertEqual(p.a, np.array([1.3]))
        self.assertEqual(p.a.flags["OWNDATA"], False)
        self.assertEqual(p.b, np.array([5.6]))
        self.assertEqual(p.b.flags["OWNDATA"], False)

    def test_update_state(self):
        p, state, ret, state_dct, ret_dct = make_partial()
        state[:] = [33.0, 66.0]
        self.assertEqual(p.a, 33.0)
        self.assertEqual(p.b, 66.0)

    def test_get_state(self):
        p, state, ret, state_dct, ret_dct = make_partial()
        a = state_dct["a"]
        b = state_dct["b"]
        self.assertEqual(a, 1.3)
        self.assertEqual(b, 5.6)

    def test_fetch_step(self):
        p, state, ret, state_dct, ret_dct = make_partial()
        p_step = p._get_step_method()
        self.assertEqual(p_step, p.step)

    def test_step(self):
        p, state, ret, state_dct, ret_dct = make_partial()
        a = state_dct["a"]
        a_ret = p.step(state_dct)
        self.assertEqual(a_ret, {"a": a * 2})


class Test_Partial_Mono_Caching(unittest.TestCase):
    def test_mono_cache_init(self):
        p = Cached()

    def test_mono_cache_after_call(self):
        p = Cached()
        p.mono.cache_enable()
        ret_1 = p.mono(65.1)
        self.assertEqual(ret_1, 65.1)

    def test_mono_cache_not_enabled(self):
        p = Cached()
        ret_1 = p.mono(65.1)
        ret_2 = p.mono(31.2)
        self.assertEqual(ret_1, 65.1)
        self.assertEqual(ret_2, 31.2)

    def test_mono_cache_disabled(self):
        p = Cached()
        p.mono.cache_enable()
        p.mono.cache_disable()
        ret_1 = p.mono(65.1)
        ret_2 = p.mono(31.2)
        self.assertEqual(ret_1, 65.1)
        self.assertEqual(ret_2, 31.2)

    def test_mono_cache_after_second_call(self):
        p = Cached()
        p.mono.cache_enable()
        ret_1 = p.mono(65.1)
        ret_2 = p.mono(31.2)
        self.assertEqual(ret_1, 65.1)
        self.assertEqual(ret_2, 65.1)

    def test_mono_cache_clear(self):
        p = Cached()
        p.mono.cache_enable()
        ret_1 = p.mono(65.1)
        p.mono.cache_clear()
        ret_2 = p.mono(31.2)
        self.assertEqual(ret_2, np.array(31.2))

    def test_mono_cache_separate_caches(self):
        p = Cached()
        p.mono.cache_enable()
        p.mono.cache_clear()
        ret_1 = p.mono(65.1)
        ret_2 = p.mono(15)
        self.assertEqual(ret_2, 65.1)
        ret_3 = p.mono_b(100)
        self.assertEqual(ret_3, 100)

    def test_get_cache_clear_functions(self):
        p = Cached()
        lst = p._get_cache_clear_functions()
        self.assertEqual(len(lst), 4)
        for f in lst:
            self.assertEqual(callable(f), True)

    def test_get_cached_methods(self):
        p = Cached()
        lst = p._get_cached_methods()
        self.assertEqual(len(lst), 4)
        for f in lst:
            self.assertEqual(callable(f), True)

    def test_set_caching(self):
        p = Cached()
        p._set_caching(enable=True)
        ret = p.mono(5)
        ret2 = p.mono(5)
        self.assertEqual(ret, ret2)


class Test_Partial_Multi_Caching(unittest.TestCase):
    def test_multi_cache_init(self):
        p = Cached()

    def test_multi_cache_after_call(self):
        p = Cached()
        p.multi.cache_enable()
        ret_1 = p.multi(65.1)
        self.assertEqual(ret_1, 65.1)

    def test_multi_cache_not_enabled(self):
        p = Cached()
        ret_1 = p.multi(65.1)
        ret_2 = p.multi(31.2)
        self.assertEqual(ret_1, 65.1)
        self.assertEqual(ret_2, 31.2)

    def test_multi_cache_disabled(self):
        p = Cached()
        p.multi.cache_enable()
        p.multi.cache_disable()
        ret_1 = p.multi(65.1)
        ret_2 = p.multi(31.2)
        self.assertEqual(ret_1, 65.1)
        self.assertEqual(ret_2, 31.2)

    def test_multi_cache_after_second_call(self):
        p = Cached()
        p.multi.cache_enable()
        ret_1 = p.multi(65.1)
        ret_2 = p.multi(31.2)
        self.assertEqual(ret_1, 65.1)
        self.assertEqual(ret_2, 31.2)

    def test_multi_cache_clear(self):
        p = Cached()
        p.multi.cache_enable()
        p.multi.cache_clear()
        ret_1 = p.multi(65.1)
        self.assertEqual(len(p.multi.__closure__[0].cell_contents), 1)
        p.multi.cache_clear()
        self.assertEqual(len(p.multi.__closure__[0].cell_contents), 0)

    def test_multi_cache_separate_caches(self):
        p = Cached()
        p.multi.cache_enable()
        p.multi.cache_clear()
        p.multi_b.cache_enable()
        p.multi_b.cache_clear()
        ret_1 = p.multi(65.1)
        self.assertEqual(len(p.multi.__closure__[0].cell_contents), 1)
        self.assertEqual(len(p.multi_b.__closure__[0].cell_contents), 0)
        ret_2 = p.multi_b(5.5)
        self.assertEqual(len(p.multi.__closure__[0].cell_contents), 1)
        self.assertEqual(len(p.multi_b.__closure__[0].cell_contents), 1)

    def test_cache_clear(self):
        p = Cached()
        p.multi.cache_enable()
        p.multi.cache_clear()
        ret_1 = p.multi(65.1)
        self.assertEqual(len(p.multi.__closure__[0].cell_contents), 1)
        p.cache_clear()
        self.assertEqual(len(p.multi.__closure__[0].cell_contents), 0)
