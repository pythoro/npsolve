# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 07:22:13 2019

@author: Reuben
"""

import unittest
import numpy as np
import timeit

from npsolve.legacy.legacy_core import Partial, Solver


class Partial_1(Partial):
    npsolve_name = "partial_1"

    def __init__(self):
        super().__init__()
        self.add_var("a", np.linspace(0, 3, 3))
        self.add_var("b", np.linspace(3, 6, 3))
        self.add_var("c", np.linspace(6, 9, 3))
        self.add_var("d", np.linspace(9, 12, 3))

    def set_vectors(self, state_dct, ret_dct):
        self.a = state_dct["a"]
        self.b = state_dct["b"]
        self.c = state_dct["c"]
        self.d = state_dct["d"]

    def step(self, state_dct, *args):
        return {
            "a": self.a + 1.0,
            "b": self.b + 1.0,
            "c": self.c + 1.0,
            "d": self.d + 1.0,
        }


class Test_Solver(unittest.TestCase):
    def test_create(self):
        s = Solver()
        p = Partial_1()
        p.connect_solver(s)
        s.npsolve_init()
        lst = s.fetch_partials()
        self.assertEqual(lst["partial_1"], p)

    def test_step(self):
        s = Solver()
        p = Partial_1()
        p.connect_solver(s)
        s.npsolve_init()
        vec = s.npsolve_state
        globals_dct = {"s": s, "vec": vec}
        time = timeit.timeit("s.step(vec)", globals=globals_dct, number=100000)

        ret = vec

        def step_baseline(vec, ret):
            ret[0:3] = vec[0:3] + 1.0
            ret[3:6] = vec[3:6] + 1.0
            ret[6:9] = vec[6:9] + 1.0
            ret[9:12] = vec[9:12] + 1.0
            return ret

        globals_dct = {"step_baseline": step_baseline, "vec": vec, "ret": ret}
        baseline = timeit.timeit(
            "step_baseline(vec, ret)", globals=globals_dct, number=100000
        )
        print()
        print("Relative speed: " + "{:0.3f}".format(time / baseline))
        self.assertLess(time, baseline * 1.15)
